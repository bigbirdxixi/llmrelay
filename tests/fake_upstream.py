"""测试用的假上游：同时提供 OpenAI 兼容与 Anthropic Messages 两套接口。

启动方式由 tests/conftest.py 控制（线程里跑 uvicorn）。
所有收到的请求体都记在 RECEIVED 里，方便断言中转站有没有正确注入/翻译参数。
"""
from __future__ import annotations

import json
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI()
RECEIVED: list[dict] = []

USAGE = {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}


def _record(path: str, body: dict) -> None:
    RECEIVED.append({"path": path, "body": body, "ts": time.time()})


def _thinking_off(body: dict) -> bool:
    """模拟"遵守关闭思考指令"的上游：只有显式关闭字段才不思考。"""
    if body.get("enable_thinking") is False:
        return True
    th = body.get("thinking")
    if isinstance(th, dict) and th.get("type") == "disabled":
        return True
    ctk = body.get("chat_template_kwargs")
    if isinstance(ctk, dict) and ctk.get("enable_thinking") is False:
        return True
    if str(body.get("reasoning_effort") or "").lower() in ("none", "minimal"):
        return True
    return False


def _anthropic_thinking_on(body: dict) -> bool:
    """真实 Anthropic 语义：thinking 只在 type=enabled/adaptive 时才思考，省略＝关。"""
    th = body.get("thinking")
    if not isinstance(th, dict):
        return False
    return str(th.get("type") or "").lower() in ("enabled", "adaptive")


def _sse(obj) -> str:
    if isinstance(obj, str):
        return f"data: {obj}\n\n"
    return "data: " + json.dumps(obj, ensure_ascii=False) + "\n\n"


def _aevent(name: str, obj: dict) -> str:
    return f"event: {name}\ndata: " + json.dumps(obj, ensure_ascii=False) + "\n\n"


# ---------------------------------------------------------------- OpenAI 兼容
@app.get("/v1/models")
async def models():
    return {"object": "list", "data": [{"id": "fake-gpt"}, {"id": "fake-gpt-2"}]}


@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = await request.json()
    _record("/v1/chat/completions", body)
    if body.get("fail"):
        return JSONResponse(status_code=400, content={
            "error": {"message": "fake upstream rejected: " + str(body["fail"]), "type": "invalid_request_error"}})

    model = body.get("model")
    want_tool = bool(body.get("tools"))
    want_reasoning = ("reasoner" in (model or "")) and not _thinking_off(body)
    stream = bool(body.get("stream"))

    if not stream:
        msg = {"role": "assistant", "content": "hello from fake openai"}
        if want_reasoning:
            msg["reasoning_content"] = "fake thinking process"
        if want_tool:
            msg = {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call_abc", "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city":"Beijing"}'}}]}
        return {
            "id": "chatcmpl-fake", "object": "chat.completion", "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "message": msg,
                         "finish_reason": "tool_calls" if want_tool else "stop"}],
            "usage": USAGE,
        }

    include_usage = bool((body.get("stream_options") or {}).get("include_usage"))

    async def gen():
        base = {"id": "chatcmpl-fake", "object": "chat.completion.chunk",
                "created": int(time.time()), "model": model}
        yield _sse({**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""},
                                         "finish_reason": None}]})
        if want_reasoning:
            for rp in ("思考", " 一下", " 再说"):
                yield _sse({**base, "choices": [{"index": 0,
                                                 "delta": {"reasoning_content": rp},
                                                 "finish_reason": None}]})
        for piece in ("hello", " from", " fake"):
            yield _sse({**base, "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]})
        if want_tool:
            yield _sse({**base, "choices": [{"index": 0, "delta": {"tool_calls": [{
                "index": 0, "id": "call_abc", "type": "function",
                "function": {"name": "get_weather", "arguments": ""}}]}, "finish_reason": None}]})
            yield _sse({**base, "choices": [{"index": 0, "delta": {"tool_calls": [{
                "index": 0, "function": {"arguments": '{"city":"Beijing"}'}}]}, "finish_reason": None}]})
        yield _sse({**base, "choices": [{"index": 0, "delta": {},
                                         "finish_reason": "tool_calls" if want_tool else "stop"}]})
        if include_usage:
            yield _sse({**base, "choices": [], "usage": USAGE})
        yield _sse("[DONE]")

    return StreamingResponse(gen(), media_type="text/event-stream")


# ---------------------------------------------------------------- Anthropic
@app.post("/v1/messages")
async def messages(request: Request):
    body = await request.json()
    _record("/v1/messages", body)
    if body.get("fail"):
        return JSONResponse(status_code=400, content={
            "type": "error", "error": {"type": "invalid_request_error",
                                       "message": "fake anthropic rejected"}})
    model = body.get("model")
    want_tool = bool(body.get("tools"))
    want_thinking = ("thinking" in (model or "")) and _anthropic_thinking_on(body)
    stream = bool(body.get("stream"))
    usage = {"input_tokens": 12, "output_tokens": 5}

    if not stream:
        content = ([{"type": "text", "text": "hello from fake claude"}] if not want_tool else
                   [{"type": "tool_use", "id": "toolu_1", "name": "get_weather",
                     "input": {"city": "Beijing"}}])
        if want_thinking:
            content = [{"type": "thinking", "thinking": "fake claude thinking"}] + content
        return {"id": "msg_fake", "type": "message", "role": "assistant", "model": model,
                "content": content,
                "stop_reason": "tool_use" if want_tool else "end_turn",
                "stop_sequence": None, "usage": usage}

    async def gen():
        yield _aevent("message_start", {
            "type": "message_start",
            "message": {"id": "msg_fake", "type": "message", "role": "assistant", "model": model,
                        "content": [], "stop_reason": None, "stop_sequence": None,
                        "usage": {"input_tokens": 12, "output_tokens": 1, "cache_read_input_tokens": 4}}})
        block = 0
        if want_thinking:
            yield _aevent("content_block_start", {
                "type": "content_block_start", "index": block,
                "content_block": {"type": "thinking", "thinking": "", "signature": ""}})
            for tp in ("先", "想一想"):
                yield _aevent("content_block_delta", {
                    "type": "content_block_delta", "index": block,
                    "delta": {"type": "thinking_delta", "thinking": tp}})
            yield _aevent("content_block_stop", {"type": "content_block_stop", "index": block})
            block += 1
        if want_tool:
            yield _aevent("content_block_start", {
                "type": "content_block_start", "index": block,
                "content_block": {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {}}})
            for part in ('{"city"', ': "Beijing"}'):
                yield _aevent("content_block_delta", {
                    "type": "content_block_delta", "index": block,
                    "delta": {"type": "input_json_delta", "partial_json": part}})
        else:
            yield _aevent("content_block_start", {
                "type": "content_block_start", "index": block,
                "content_block": {"type": "text", "text": ""}})
            for piece in ("hello", " from", " fake claude"):
                yield _aevent("content_block_delta", {
                    "type": "content_block_delta", "index": block,
                    "delta": {"type": "text_delta", "text": piece}})
        yield _aevent("content_block_stop", {"type": "content_block_stop", "index": block})
        yield _aevent("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use" if want_tool else "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 5}})
        yield _aevent("message_stop", {"type": "message_stop"})

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/v1/messages/count_tokens")
async def count_tokens(request: Request):
    await request.json()
    return {"input_tokens": 12}


# ---------------------------------------------------------------- 假的"网页版聊天"页面
WEB_CHAT_PAGE = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><title>Fake Web Chat</title></head>
<body>
  <h3>Fake Web Chat</h3>
  <div id="answer"></div>
  <textarea id="prompt" placeholder="发消息"></textarea>
  <button id="send">发送</button>
  <button id="newchat">新对话</button>
<script>
const ta = document.getElementById('prompt');
const box = document.getElementById('answer');
document.getElementById('newchat').onclick = () => { box.textContent = ''; };
async function send(){
  const q = ta.value.trim();
  if(!q) return;
  ta.value = '';
  box.textContent = '';
  const reply = 'ECHO: ' + q;
  for (const ch of reply){
    box.textContent += ch;
    await new Promise(r => setTimeout(r, 15));
  }
}
document.getElementById('send').onclick = send;
ta.addEventListener('keydown', e => { if(e.key === 'Enter' && !e.shiftKey){ e.preventDefault(); send(); } });
</script>
</body></html>"""


@app.get("/webchat")
async def webchat():
    from fastapi.responses import HTMLResponse
    return HTMLResponse(WEB_CHAT_PAGE)
