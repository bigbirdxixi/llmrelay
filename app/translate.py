"""OpenAI Chat Completions <-> Anthropic Messages 协议互转（含 SSE 流式状态机）。

下游可能是 OpenAI 协议客户端，也可能是 Anthropic 协议客户端（Claude Code / Anthropic SDK）；
上游也可能是两种协议之一。4 种组合都由这里负责翻译，用量统一折算成 Usage。
"""
from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Iterable

# ---------------------------------------------------------------- 用量


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    estimated: bool = False

    def finish(self) -> "Usage":
        if not self.total_tokens:
            self.total_tokens = self.prompt_tokens + self.completion_tokens
        return self


_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af\uf900-\ufaff]")


def estimate_tokens(text: str) -> int:
    """无 usage 时的兜底估算：CJK 约 1.5 字/token，其它约 4 字符/token。"""
    if not text:
        return 0
    cjk = len(_CJK.findall(text))
    other = len(text) - cjk
    return max(1, int(round(cjk / 1.5 + other / 4.0)))


def estimate_tokens_from_chars(n: int) -> int:
    """只知道字符数时的粗估（混合中英文按 3.2 字符/token）。"""
    if n <= 0:
        return 0
    return max(1, int(round(n / 3.2)))



def usage_from_openai(u: dict | None) -> Usage:
    u = u or {}
    det = u.get("prompt_tokens_details") or {}
    cdet = u.get("completion_tokens_details") or {}
    return Usage(
        prompt_tokens=int(u.get("prompt_tokens") or 0),
        completion_tokens=int(u.get("completion_tokens") or 0),
        total_tokens=int(u.get("total_tokens") or 0),
        cached_tokens=int(det.get("cached_tokens") or 0),
        reasoning_tokens=int(cdet.get("reasoning_tokens") or 0),
    ).finish()


def usage_from_anthropic(u: dict | None) -> Usage:
    u = u or {}
    cache_read = int(u.get("cache_read_input_tokens") or 0)
    cache_write = int(u.get("cache_creation_input_tokens") or 0)
    inp = int(u.get("input_tokens") or 0)
    return Usage(
        prompt_tokens=inp + cache_read + cache_write,
        completion_tokens=int(u.get("output_tokens") or 0),
        cached_tokens=cache_read,
        cache_write_tokens=cache_write,
    ).finish()


def merge_usage(dst: Usage, src: Usage) -> Usage:
    dst.prompt_tokens = max(dst.prompt_tokens, src.prompt_tokens)
    dst.completion_tokens = max(dst.completion_tokens, src.completion_tokens)
    dst.cached_tokens = max(dst.cached_tokens, src.cached_tokens)
    dst.cache_write_tokens = max(dst.cache_write_tokens, src.cache_write_tokens)
    dst.reasoning_tokens = max(dst.reasoning_tokens, src.reasoning_tokens)
    dst.total_tokens = dst.prompt_tokens + dst.completion_tokens
    dst.estimated = dst.estimated or src.estimated
    return dst


# ---------------------------------------------------------------- SSE 解析


class SSEParser:
    """增量解析 SSE 文本流 -> (event, [data lines], [comments])。"""

    def __init__(self) -> None:
        self.buf = ""
        self._event: str | None = None
        self._data: list[str] = []
        self._comments: list[str] = []

    def feed(self, text: str) -> list[tuple[str | None, list[str], list[str]]]:
        self.buf += text
        out: list[tuple[str | None, list[str], list[str]]] = []
        while True:
            nl = self.buf.find("\n")
            if nl < 0:
                break
            line = self.buf[:nl]
            if line.endswith("\r"):
                line = line[:-1]
            self.buf = self.buf[nl + 1:]
            if line == "":
                if self._data or self._comments:
                    out.append((self._event, self._data, self._comments))
                self._event, self._data, self._comments = None, [], []
                continue
            if line.startswith(":"):
                self._comments.append(line)
                continue
            field, _, value = line.partition(":")
            if value.startswith(" "):
                value = value[1:]
            if field == "event":
                self._event = value
            elif field == "data":
                self._data.append(value)
            # id / retry 忽略
        return out

    def flush(self) -> list[tuple[str | None, list[str], list[str]]]:
        out: list[tuple[str | None, list[str], list[str]]] = []
        if self._data or self._comments:
            out.append((self._event, self._data, self._comments))
            self._event, self._data, self._comments = None, [], []
        return out


def sse_frame(data: str, event: str | None = None) -> str:
    head = f"event: {event}\n" if event else ""
    if data == "":
        return head + "\n"
    return head + "".join(f"data: {line}\n" for line in data.split("\n")) + "\n"


# ---------------------------------------------------------------- 请求：OpenAI -> Anthropic


def _text_of(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict):
                if p.get("type") in ("text", "input_text") and p.get("text"):
                    parts.append(str(p["text"]))
                elif p.get("text"):
                    parts.append(str(p["text"]))
        return "".join(parts)
    return str(content)


def _image_part_to_anthropic(part: dict) -> dict | None:
    url = ""
    iu = part.get("image_url")
    if isinstance(iu, dict):
        url = iu.get("url") or ""
    elif isinstance(iu, str):
        url = iu
    if not url and part.get("type") == "image" and isinstance(part.get("source"), dict):
        # 已经是 Anthropic 形状，原样返回
        return part
    if not url:
        return None
    m = re.match(r"data:([^;,]+);base64,(.*)$", url, re.S)
    if m:
        return {"type": "image", "source": {"type": "base64", "media_type": m.group(1), "data": m.group(2)}}
    return {"type": "image", "source": {"type": "url", "url": url}}


def _content_to_anthropic(content: Any) -> list[dict]:
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    blocks: list[dict] = []
    if isinstance(content, list):
        for p in content:
            if isinstance(p, str):
                if p:
                    blocks.append({"type": "text", "text": p})
                continue
            if not isinstance(p, dict):
                continue
            t = p.get("type")
            if t in ("text", "input_text"):
                txt = p.get("text") or ""
                if txt:
                    blocks.append({"type": "text", "text": txt})
            elif t in ("image_url", "image", "input_image"):
                blk = _image_part_to_anthropic(p)
                if blk:
                    blocks.append(blk)
            else:
                txt = _text_of(p)
                if txt:
                    blocks.append({"type": "text", "text": txt})
    return blocks


def _json_loads_maybe(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        val = json.loads(raw)
        return val if isinstance(val, dict) else {"value": val}
    except (TypeError, ValueError):
        return {"raw": str(raw)}


def _merge_same_role(msgs: list[dict]) -> list[dict]:
    out: list[dict] = []
    for m in msgs:
        if out and out[-1]["role"] == m["role"]:
            out[-1]["content"].extend(m["content"])
        else:
            out.append({"role": m["role"], "content": list(m["content"])})
    return out


def openai_to_anthropic(body: dict, upstream_model: str, default_max_tokens: int = 8192) -> dict:
    max_tokens = body.get("max_completion_tokens") or body.get("max_tokens") or default_max_tokens
    msgs: list[dict] = []
    system_parts: list[str] = []

    for m in body.get("messages") or []:
        role = m.get("role")
        content = m.get("content")
        if role in ("system", "developer"):
            txt = _text_of(content)
            if txt:
                system_parts.append(txt)
            continue
        if role == "user":
            blocks = _content_to_anthropic(content)
            if blocks:
                msgs.append({"role": "user", "content": blocks})
            continue
        if role == "assistant":
            blocks = _content_to_anthropic(content)
            for tc in (m.get("tool_calls") or []):
                fn = tc.get("function") or {}
                blocks.append({
                    "type": "tool_use",
                    "id": tc.get("id") or "toolu_" + uuid.uuid4().hex[:20],
                    "name": fn.get("name") or "tool",
                    "input": _json_loads_maybe(fn.get("arguments")),
                })
            if blocks:
                msgs.append({"role": "assistant", "content": blocks})
            continue
        if role in ("tool", "function"):
            msgs.append({"role": "user", "content": [{
                "type": "tool_result",
                "tool_use_id": m.get("tool_call_id") or m.get("name") or "",
                "content": _text_of(content),
            }]})
            continue

    msgs = _merge_same_role(msgs)
    if not msgs:
        msgs = [{"role": "user", "content": [{"type": "text", "text": "(empty)"}]}]

    out: dict[str, Any] = {
        "model": upstream_model,
        "max_tokens": int(max_tokens),
        "messages": msgs,
    }
    if system_parts:
        out["system"] = "\n\n".join(system_parts)

    temp = body.get("temperature")
    if isinstance(temp, (int, float)):
        out["temperature"] = max(0.0, min(float(temp), 1.0))
    top_p = body.get("top_p")
    if isinstance(top_p, (int, float)):
        out["top_p"] = float(top_p)

    stop = body.get("stop")
    if isinstance(stop, str):
        stop = [stop]
    if isinstance(stop, list) and stop:
        seqs = [s for s in stop if isinstance(s, str) and s.strip()] or [s for s in stop if isinstance(s, str)]
        if seqs:
            out["stop_sequences"] = seqs[:8]

    tools = body.get("tools") or []
    anth_tools = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        if t.get("type") not in (None, "function"):
            continue
        fn = t.get("function") or t
        name = fn.get("name")
        if not name:
            continue
        anth_tools.append({
            "name": name,
            "description": fn.get("description") or "",
            "input_schema": fn.get("parameters") or {"type": "object", "properties": {}},
        })

    tool_choice = body.get("tool_choice")
    if anth_tools:
        if tool_choice == "none":
            anth_tools = []
        else:
            out["tools"] = anth_tools
            if tool_choice == "required":
                out["tool_choice"] = {"type": "any"}
            elif isinstance(tool_choice, dict):
                name = (tool_choice.get("function") or {}).get("name") or tool_choice.get("name")
                if name:
                    out["tool_choice"] = {"type": "tool", "name": name}
                else:
                    out["tool_choice"] = {"type": "auto"}
            elif tool_choice == "auto":
                out["tool_choice"] = {"type": "auto"}

    if body.get("stream"):
        out["stream"] = True
    user = body.get("user")
    if user:
        out["metadata"] = {"user_id": str(user)}
    extra = body.get("extra_body")
    if isinstance(extra, dict):
        out.update({k: v for k, v in extra.items() if k not in out})
    return out


# ---------------------------------------------------------------- 请求：Anthropic -> OpenAI


def anthropic_to_openai(body: dict, upstream_model: str) -> dict:
    out: dict[str, Any] = {"model": upstream_model, "messages": []}
    sys = body.get("system")
    if sys:
        txt = sys if isinstance(sys, str) else _text_of(sys)
        if txt:
            out["messages"].append({"role": "system", "content": txt})

    for m in body.get("messages") or []:
        role = m.get("role") or "user"
        content = m.get("content")
        if isinstance(content, str):
            out["messages"].append({"role": role, "content": content})
            continue
        blocks = content or []
        texts: list[str] = []
        parts: list[dict] = []
        tool_calls: list[dict] = []
        tool_results: list[dict] = []
        for b in blocks:
            if not isinstance(b, dict):
                continue
            t = b.get("type")
            if t == "text":
                txt = b.get("text") or ""
                if txt:
                    texts.append(txt)
                    parts.append({"type": "text", "text": txt})
            elif t == "image":
                src = b.get("source") or {}
                if src.get("type") == "base64":
                    url = f"data:{src.get('media_type', 'image/png')};base64,{src.get('data', '')}"
                else:
                    url = src.get("url") or ""
                if url:
                    parts.append({"type": "image_url", "image_url": {"url": url}})
            elif t == "tool_use":
                tool_calls.append({
                    "id": b.get("id") or "call_" + uuid.uuid4().hex[:12],
                    "type": "function",
                    "function": {
                        "name": b.get("name") or "tool",
                        "arguments": json.dumps(b.get("input") or {}, ensure_ascii=False),
                    },
                })
            elif t == "tool_result":
                tool_results.append({
                    "role": "tool",
                    "tool_call_id": b.get("tool_use_id") or "",
                    "content": _text_of(b.get("content")),
                })
        if tool_results:
            if texts:
                out["messages"].append({"role": "user", "content": "\n".join(texts)})
            out["messages"].extend(tool_results)
            continue
        if role == "assistant" and tool_calls:
            msg: dict[str, Any] = {"role": "assistant", "content": "\n".join(texts) if texts else None}
            msg["tool_calls"] = tool_calls
            out["messages"].append(msg)
            continue
        if parts and any(p["type"] == "image_url" for p in parts) and len(parts) > 1:
            out["messages"].append({"role": role, "content": parts})
        elif texts:
            out["messages"].append({"role": role, "content": "\n".join(texts)})
        elif parts:
            out["messages"].append({"role": role, "content": parts})
        # 空内容消息直接丢弃

    if not out["messages"]:
        out["messages"] = [{"role": "user", "content": "(empty)"}]

    mt = body.get("max_tokens")
    if mt:
        out["max_tokens"] = int(mt)
    for k_src, k_dst in (("temperature", "temperature"), ("top_p", "top_p")):
        v = body.get(k_src)
        if isinstance(v, (int, float)):
            out[k_dst] = float(v)
    stop = body.get("stop_sequences")
    if isinstance(stop, list) and stop:
        out["stop"] = stop
    elif isinstance(stop, str) and stop:
        out["stop"] = [stop]

    tools = []
    for t in body.get("tools") or []:
        if not isinstance(t, dict):
            continue
        name = t.get("name")
        if not name:
            continue
        tools.append({
            "type": "function",
            "function": {
                "name": name,
                "description": t.get("description") or "",
                "parameters": t.get("input_schema") or {"type": "object", "properties": {}},
            },
        })
    if tools:
        out["tools"] = tools
        tc = body.get("tool_choice")
        if isinstance(tc, dict):
            tct = tc.get("type")
            if tct == "any":
                out["tool_choice"] = "required"
            elif tct == "tool" and tc.get("name"):
                out["tool_choice"] = {"type": "function", "function": {"name": tc["name"]}}
            elif tct == "auto":
                out["tool_choice"] = "auto"
            elif tct == "none":
                out["tool_choice"] = "none"
    if body.get("stream"):
        out["stream"] = True
    return out


# ---------------------------------------------------------------- 非流式响应翻译

STOP_TO_OPENAI = {
    "end_turn": "stop",
    "max_tokens": "length",
    "stop_sequence": "stop",
    "tool_use": "tool_calls",
    "pause_turn": "stop",
    "refusal": "content_filter",
}
STOP_TO_ANTHROPIC = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "content_filter": "end_turn",
}


def anthropic_response_to_openai(resp: dict, client_model: str) -> dict:
    text_parts: list[str] = []
    reasoning: list[str] = []
    tool_calls: list[dict] = []
    for b in resp.get("content") or []:
        t = (b or {}).get("type")
        if t == "text":
            text_parts.append(b.get("text") or "")
        elif t == "thinking":
            reasoning.append(b.get("thinking") or "")
        elif t == "tool_use":
            tool_calls.append({
                "id": b.get("id") or "call_" + uuid.uuid4().hex[:12],
                "type": "function",
                "function": {"name": b.get("name") or "tool",
                             "arguments": json.dumps(b.get("input") or {}, ensure_ascii=False)},
            })
    usage = usage_from_anthropic(resp.get("usage"))
    stop = resp.get("stop_reason")
    message: dict[str, Any] = {"role": "assistant", "content": "".join(text_parts)}
    if reasoning:
        message["reasoning_content"] = "".join(reasoning)
    if tool_calls:
        message["tool_calls"] = tool_calls
        if not message["content"]:
            message["content"] = None
    return {
        "id": resp.get("id") or "chatcmpl-" + uuid.uuid4().hex[:20],
        "object": "chat.completion",
        "created": int(time.time()),
        "model": client_model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": STOP_TO_OPENAI.get(stop or "", "stop"),
        }],
        "usage": {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
            "prompt_tokens_details": {"cached_tokens": usage.cached_tokens},
        },
    }


def openai_response_to_anthropic(resp: dict, client_model: str) -> dict:
    choices = resp.get("choices") or [{}]
    msg = (choices[0] or {}).get("message") or {}
    blocks: list[dict] = []
    txt = _text_of(msg.get("content"))
    if txt:
        blocks.append({"type": "text", "text": txt})
    for tc in (msg.get("tool_calls") or []):
        fn = tc.get("function") or {}
        blocks.append({
            "type": "tool_use",
            "id": tc.get("id") or "toolu_" + uuid.uuid4().hex[:20],
            "name": fn.get("name") or "tool",
            "input": _json_loads_maybe(fn.get("arguments")),
        })
    if not blocks:
        blocks = [{"type": "text", "text": ""}]
    usage = usage_from_openai(resp.get("usage"))
    return {
        "id": "msg_" + uuid.uuid4().hex[:24],
        "type": "message",
        "role": "assistant",
        "model": client_model,
        "content": blocks,
        "stop_reason": STOP_TO_ANTHROPIC.get((choices[0] or {}).get("finish_reason") or "", "end_turn"),
        "stop_sequence": None,
        "usage": {
            "input_tokens": max(0, usage.prompt_tokens - usage.cached_tokens),
            "output_tokens": usage.completion_tokens,
            "cache_read_input_tokens": usage.cached_tokens,
        },
    }


# ---------------------------------------------------------------- SSE 桥


class Bridge:
    """把上游 SSE 事件翻译成下游 SSE 帧。

    hide_reasoning=True 时，思考/推理内容不会透给下游（只用于"关不掉"的模型）。
    """

    hide_reasoning: bool = False

    def __init__(self) -> None:
        self.usage = Usage()
        self.completion_chars = 0
        self.text_chars = 0
        self.finished = False

    def push(self, event: str | None, data: str) -> list[str]:
        raise NotImplementedError

    def close(self) -> list[str]:
        return []


class OpenAIPassthrough(Bridge):
    """OpenAI 下游 <-> OpenAI 上游：语义透传，只旁路解析 usage。

    swallow_usage=True 时，只含 usage 的尾包（choices 为空）会被吞掉，
    这样我们可以主动向上游要 usage 拿精确 token 数，又不改变客户端看到的流。
    """

    def __init__(self, include_usage: bool = False, swallow_usage: bool = False) -> None:
        super().__init__()
        self.include_usage = include_usage
        self.swallow_usage = swallow_usage

    def push(self, event: str | None, data: str) -> list[str]:
        payload = data.strip()
        if payload == "[DONE]":
            self.finished = True
            frames: list[str] = []
            if self.include_usage and not self.usage.total_tokens:
                frames.append(sse_frame(json.dumps(self._usage_chunk(), ensure_ascii=False)))
            frames.append(sse_frame("[DONE]"))
            return frames
        try:
            obj = json.loads(payload)
        except ValueError:
            return [sse_frame(data)]
        if isinstance(obj, dict):
            if obj.get("usage"):
                merge_usage(self.usage, usage_from_openai(obj["usage"]))
                if self.swallow_usage and not obj.get("choices") and not obj.get("error"):
                    return []  # 只含 usage 的尾包，客户端没要求，吞掉
            dropped = False
            for ch in obj.get("choices") or []:
                if not isinstance(ch, dict):
                    continue
                delta = ch.get("delta") or {}
                if isinstance(delta, dict):
                    if self.hide_reasoning and any(
                            k in delta for k in ("reasoning_content", "reasoning", "reasoning_details")):
                        for k in ("reasoning_content", "reasoning", "reasoning_details"):
                            delta.pop(k, None)
                        dropped = True
                    for key in ("content", "reasoning_content", "reasoning"):
                        v = delta.get(key)
                        if isinstance(v, str) and v:
                            self.text_chars += len(v)
                fr = ch.get("finish_reason")
                if fr:
                    self.finished = True
            if dropped:
                return [sse_frame(json.dumps(obj, ensure_ascii=False))]
        return [sse_frame(data)]

    def _usage_chunk(self) -> dict:
        u = self.usage.finish()
        return {
            "id": "chatcmpl-" + uuid.uuid4().hex[:20],
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": "",
            "choices": [],
            "usage": {
                "prompt_tokens": u.prompt_tokens,
                "completion_tokens": u.completion_tokens,
                "total_tokens": u.total_tokens,
            },
        }


class AnthropicPassthrough(Bridge):
    """Anthropic 下游 <-> Anthropic 上游（Claude Code 那条路）。"""

    def __init__(self) -> None:
        super().__init__()
        self.hidden_blocks: set[int] = set()

    def push(self, event: str | None, data: str) -> list[str]:
        payload = data.strip()
        if payload:
            try:
                obj = json.loads(payload)
            except ValueError:
                obj = None
            if isinstance(obj, dict):
                t = obj.get("type") or event
                idx = int(obj.get("index") or 0)
                if t == "message_start":
                    merge_usage(self.usage, usage_from_anthropic((obj.get("message") or {}).get("usage")))
                elif t == "content_block_start":
                    if self.hide_reasoning and (obj.get("content_block") or {}).get("type") in (
                            "thinking", "redacted_thinking"):
                        self.hidden_blocks.add(idx)
                        return []
                elif t == "content_block_stop":
                    if idx in self.hidden_blocks:
                        self.hidden_blocks.discard(idx)
                        return []
                elif t == "message_delta":
                    merge_usage(self.usage, usage_from_anthropic(obj.get("usage")))
                    if (obj.get("delta") or {}).get("stop_reason"):
                        self.finished = True
                elif t == "content_block_delta":
                    if idx in self.hidden_blocks:
                        return []
                    d = obj.get("delta") or {}
                    if d.get("type") == "text_delta":
                        self.text_chars += len(d.get("text") or "")
                    elif d.get("type") == "thinking_delta":
                        self.text_chars += len(d.get("thinking") or "")
        return [sse_frame(data, event)]


class AnthropicToOpenAIStream(Bridge):
    """Anthropic 上游事件 -> OpenAI chat.completion.chunk。"""

    def __init__(self, client_model: str, include_usage: bool = False) -> None:
        super().__init__()
        self.model = client_model
        self.include_usage = include_usage
        self.id = "chatcmpl-" + uuid.uuid4().hex[:24]
        self.created = int(time.time())
        self.tool_idx = -1
        self.block_tool: dict[int, int] = {}
        self.block_kind: dict[int, str] = {}
        self.finish_reason = "stop"
        self.started = False

    def _chunk(self, delta: dict, finish_reason: str | None = None) -> str:
        return sse_frame(json.dumps({
            "id": self.id, "object": "chat.completion.chunk", "created": self.created,
            "model": self.model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }, ensure_ascii=False))

    def _usage_frame(self) -> str:
        u = self.usage.finish()
        return sse_frame(json.dumps({
            "id": self.id, "object": "chat.completion.chunk", "created": self.created,
            "model": self.model, "choices": [],
            "usage": {
                "prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens,
                "total_tokens": u.total_tokens,
            },
        }, ensure_ascii=False))

    def push(self, event: str | None, data: str) -> list[str]:
        payload = data.strip()
        if not payload:
            return []
        try:
            obj = json.loads(payload)
        except ValueError:
            return []
        t = obj.get("type") or event or ""
        frames: list[str] = []

        if t == "message_start":
            merge_usage(self.usage, usage_from_anthropic((obj.get("message") or {}).get("usage")))
            if not self.started:
                self.started = True
                frames.append(self._chunk({"role": "assistant", "content": ""}))
            return frames

        if t == "content_block_start":
            idx = int(obj.get("index") or 0)
            cb = obj.get("content_block") or {}
            cbt = cb.get("type")
            self.block_kind[idx] = cbt or "text"
            if cbt in ("thinking", "redacted_thinking"):
                if cb.get("thinking"):
                    self.text_chars += len(cb["thinking"])
                    frames.append(self._chunk({"reasoning_content": cb["thinking"]}))
                return frames
            if cbt == "tool_use":
                self.tool_idx += 1
                self.block_tool[idx] = self.tool_idx
                frames.append(self._chunk({"tool_calls": [{
                    "index": self.tool_idx, "id": cb.get("id") or "call_" + uuid.uuid4().hex[:12],
                    "type": "function",
                    "function": {"name": cb.get("name") or "tool", "arguments": ""},
                }]}))
            elif cbt == "text" and cb.get("text"):
                self.text_chars += len(cb["text"])
                frames.append(self._chunk({"content": cb["text"]}))
            return frames

        if t == "content_block_delta":
            idx = int(obj.get("index") or 0)
            d = obj.get("delta") or {}
            dt = d.get("type")
            if dt == "text_delta":
                txt = d.get("text") or ""
                self.text_chars += len(txt)
                if txt:
                    frames.append(self._chunk({"content": txt}))
            elif dt == "thinking_delta":
                txt = d.get("thinking") or ""
                self.text_chars += len(txt)
                if txt:
                    frames.append(self._chunk({"reasoning_content": txt}))
            elif dt == "input_json_delta":
                ti = self.block_tool.get(idx, 0)
                frames.append(self._chunk({"tool_calls": [{
                    "index": ti, "function": {"arguments": d.get("partial_json") or ""},
                }]}))
            return frames

        if t == "message_delta":
            merge_usage(self.usage, usage_from_anthropic(obj.get("usage")))
            stop = (obj.get("delta") or {}).get("stop_reason")
            if stop:
                self.finish_reason = STOP_TO_OPENAI.get(stop, "stop")
            return frames

        if t == "message_stop":
            frames.append(self._chunk({}, self.finish_reason))
            if self.include_usage:
                frames.append(self._usage_frame())
            frames.append(sse_frame("[DONE]"))
            self.finished = True
            return frames

        if t == "error":
            err = obj.get("error") or obj
            frames.append(sse_frame(json.dumps({"error": err}, ensure_ascii=False)))
            frames.append(sse_frame("[DONE]"))
            self.finished = True
            return frames

        return frames

    def close(self) -> list[str]:
        if self.finished:
            return []
        frames = [self._chunk({}, self.finish_reason)]
        if self.include_usage:
            frames.append(self._usage_frame())
        frames.append(sse_frame("[DONE]"))
        self.finished = True
        return frames


class OpenAIToAnthropicStream(Bridge):
    """OpenAI 上游 chunk -> Anthropic Messages SSE 事件。"""

    def __init__(self, client_model: str, est_input_tokens: int = 0) -> None:
        super().__init__()
        self.model = client_model
        self.id = "msg_" + uuid.uuid4().hex[:24]
        self.est_input = est_input_tokens
        self.block_index = -1
        self.block_open = False
        self.block_kind = ""
        self.tool_block: dict[int, int] = {}
        self.finish_reason = "end_turn"
        self.started = False

    def start(self) -> list[str]:
        if self.started:
            return []
        self.started = True
        return [sse_frame(json.dumps({
            "type": "message_start",
            "message": {
                "id": self.id, "type": "message", "role": "assistant", "model": self.model,
                "content": [], "stop_reason": None, "stop_sequence": None,
                "usage": {"input_tokens": self.est_input, "output_tokens": 1},
            },
        }, ensure_ascii=False), "message_start")]

    def _open_block(self, kind: str, meta: dict | None = None) -> list[str]:
        frames = self._close_block()
        self.block_index += 1
        self.block_open = True
        self.block_kind = kind
        block: dict[str, Any] = {"type": kind}
        if kind == "tool_use":
            block.update({"id": (meta or {}).get("id") or "toolu_" + uuid.uuid4().hex[:20],
                          "name": (meta or {}).get("name") or "tool", "input": {}})
        elif kind == "thinking":
            block.update({"thinking": "", "signature": ""})
        else:
            block.update({"text": ""})
        frames.append(sse_frame(json.dumps({
            "type": "content_block_start", "index": self.block_index, "content_block": block,
        }, ensure_ascii=False), "content_block_start"))
        return frames

    def _close_block(self) -> list[str]:
        if not self.block_open:
            return []
        self.block_open = False
        return [sse_frame(json.dumps({
            "type": "content_block_stop", "index": self.block_index,
        }, ensure_ascii=False), "content_block_stop")]

    def _delta(self, delta: dict) -> list[str]:
        if self.block_kind == "tool_use":
            return [sse_frame(json.dumps({
                "type": "content_block_delta", "index": self.block_index,
                "delta": {"type": "input_json_delta", "partial_json": delta.get("partial_json", "")},
            }, ensure_ascii=False), "content_block_delta")]
        dtype = "thinking_delta" if self.block_kind == "thinking" else "text_delta"
        key = "thinking" if dtype == "thinking_delta" else "text"
        return [sse_frame(json.dumps({
            "type": "content_block_delta", "index": self.block_index,
            "delta": {"type": dtype, key: delta.get(key, "")},
        }, ensure_ascii=False), "content_block_delta")]

    def push(self, event: str | None, data: str) -> list[str]:
        payload = data.strip()
        if not self.started:
            frames = self.start()
        else:
            frames = []
        if payload == "[DONE]":
            frames.extend(self.close())
            return frames
        try:
            obj = json.loads(payload)
        except ValueError:
            return frames
        if not isinstance(obj, dict):
            return frames
        if obj.get("error"):
            frames.append(sse_frame(json.dumps({
                "type": "error", "error": obj["error"],
            }, ensure_ascii=False), "error"))
            self.finished = True
            return frames
        if obj.get("usage"):
            merge_usage(self.usage, usage_from_openai(obj["usage"]))
        for ch in obj.get("choices") or []:
            if not isinstance(ch, dict):
                continue
            delta = ch.get("delta") or {}
            reasoning = delta.get("reasoning_content") or delta.get("reasoning")
            if isinstance(reasoning, str) and reasoning:
                if self.block_kind != "thinking":
                    frames.extend(self._open_block("thinking"))
                self.text_chars += len(reasoning)
                frames.extend(self._delta({"thinking": reasoning}))
            content = delta.get("content")
            if isinstance(content, str) and content:
                if self.block_kind != "text":
                    frames.extend(self._open_block("text"))
                self.text_chars += len(content)
                frames.extend(self._delta({"text": content}))
            for tc in (delta.get("tool_calls") or []):
                if not isinstance(tc, dict):
                    continue
                ti = int(tc.get("index") or 0)
                fn = tc.get("function") or {}
                if ti not in self.tool_block:
                    frames.extend(self._open_block("tool_use", {"id": tc.get("id"), "name": fn.get("name")}))
                    self.tool_block[ti] = self.block_index
                elif self.block_kind != "tool_use":
                    frames.extend(self._open_block("tool_use", {"id": tc.get("id"), "name": fn.get("name")}))
                args = fn.get("arguments")
                if args:
                    frames.extend(self._delta({"partial_json": args}))
            fr = ch.get("finish_reason")
            if fr:
                self.finish_reason = STOP_TO_ANTHROPIC.get(fr, "end_turn")
        return frames

    def close(self) -> list[str]:
        if self.finished:
            return []
        self.finished = True
        frames = self._close_block()
        u = self.usage.finish()
        out_tokens = u.completion_tokens or estimate_tokens("x" * self.text_chars)
        if not u.prompt_tokens:
            self.usage.prompt_tokens = self.est_input
        frames.append(sse_frame(json.dumps({
            "type": "message_delta",
            "delta": {"stop_reason": self.finish_reason, "stop_sequence": None},
            "usage": {"output_tokens": out_tokens},
        }, ensure_ascii=False), "message_delta"))
        frames.append(sse_frame(json.dumps({"type": "message_stop"}, ensure_ascii=False), "message_stop"))
        return frames


def flatten_messages_text(body: dict) -> str:
    """把请求里的所有文本拼起来，用于 fallback 估算 prompt tokens。"""
    chunks: list[str] = []
    sys = body.get("system")
    if sys:
        chunks.append(sys if isinstance(sys, str) else _text_of(sys))
    for m in body.get("messages") or []:
        if isinstance(m, dict):
            c = m.get("content")
            chunks.append(c if isinstance(c, str) else _text_of(c))
            for tc in (m.get("tool_calls") or []):
                chunks.append(str((tc.get("function") or {}).get("arguments") or ""))
    return "\n".join(x for x in chunks if x)


def iter_json_events(lines: Iterable[str]):
    for line in lines:
        if line.startswith("data:"):
            yield line[5:].strip()
