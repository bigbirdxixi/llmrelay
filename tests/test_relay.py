"""端到端门禁测试：真 HTTP 打到假上游，验证转发、翻译、用量落库。"""
from __future__ import annotations

import json

import pytest

from tests import conftest  # noqa: E402  (fixture 运行后才有的账号 key 要按模块属性取)
from tests.conftest import sse_parse  # noqa: E402


def _latest():
    from app.main import store
    rows = store.recent(1)
    assert rows, "没有写入任何用量记录"
    return rows[0]


# ---------------------------------------------------------------- R1 鉴权
def test_r1_auth_missing_and_bad_key(client):
    r = client.post("/v1/chat/completions", json={"model": "fake/fake-gpt", "messages": []})
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "invalid_api_key"

    r = client.post("/v1/messages", json={"model": "fake/fake-gpt", "messages": []},
                    headers={"x-api-key": "wrong"})
    assert r.status_code == 401
    assert r.json()["type"] == "error"
    assert r.json()["error"]["type"] == "authentication_error"

    ok = client.get("/v1/models", headers={"authorization": f"Bearer {conftest._TEST_API_KEY}"})
    assert ok.status_code == 200


# ---------------------------------------------------------------- R2 非流式透传
def test_r2_openai_nonstream_passthrough(client, hdr):
    r = client.post("/v1/chat/completions", headers=hdr, json={
        "model": "fake/fake-gpt", "messages": [{"role": "user", "content": "hi"}],
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["choices"][0]["message"]["content"] == "hello from fake openai"
    assert r.headers["x-relay-platform"] == "fake"

    rec = _latest()
    assert rec["platform"] == "fake" and rec["model"] == "fake-gpt"
    assert rec["prompt_tokens"] == 11 and rec["completion_tokens"] == 7
    assert rec["total_tokens"] == 18 and rec["estimated"] == 0
    assert rec["status"] == "ok" and rec["stream"] == 0


# ---------------------------------------------------------------- R3 流式透传（默认吞 usage 尾包）
def test_r3_openai_stream_passthrough(client, hdr):
    body = {"model": "fake/fake-gpt", "stream": True,
            "messages": [{"role": "user", "content": "hi"}]}
    with client.stream("POST", "/v1/chat/completions", headers=hdr, json=body) as r:
        assert r.status_code == 200
        events = sse_parse(r.iter_lines())
    datas = [d for _, d in events]
    assert datas[-1] == "[DONE]"
    text = ""
    for d in datas[:-1]:
        obj = json.loads(d)
        for ch in obj.get("choices") or []:
            text += (ch.get("delta") or {}).get("content") or ""
    assert text == "hello from fake"
    # 客户端没要 include_usage，就不该看到只含 usage 的尾包
    assert not any(json.loads(d).get("usage") for d in datas[:-1] if json.loads(d).get("choices") == [])

    rec = _latest()
    assert rec["stream"] == 1
    assert rec["total_tokens"] == 18, "应该拿到上游精确 usage 而不是估算"
    assert rec["estimated"] == 0
    assert rec["ttft_ms"] >= 0


# ---------------------------------------------------------------- R4 客户端要 include_usage
def test_r4_openai_stream_include_usage(client, hdr):
    body = {"model": "fake/fake-gpt", "stream": True, "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": "hi"}]}
    with client.stream("POST", "/v1/chat/completions", headers=hdr, json=body) as r:
        datas = [d for _, d in sse_parse(r.iter_lines())]
    usage_chunks = [json.loads(d) for d in datas if d != "[DONE]" and json.loads(d).get("usage")]
    assert usage_chunks, "客户端要求了 include_usage，必须收到 usage 尾包"
    assert usage_chunks[-1]["usage"]["total_tokens"] == 18


# ---------------------------------------------------------------- R5 OpenAI 下游 -> Anthropic 上游（流式翻译）
def test_r5_openai_downstream_anthropic_upstream_stream(client, hdr):
    body = {"model": "fakeclaude/fake-claude", "stream": True,
            "messages": [{"role": "system", "content": "be nice"},
                         {"role": "user", "content": "hi"}]}
    with client.stream("POST", "/v1/chat/completions", headers=hdr, json=body) as r:
        assert r.status_code == 200, r.text
        datas = [d for _, d in sse_parse(r.iter_lines())]

    from tests import fake_upstream
    sent = [x for x in fake_upstream.RECEIVED if x["path"] == "/v1/messages"][-1]["body"]
    assert sent["system"] == "be nice"
    assert sent["model"] == "fake-claude"
    assert sent["max_tokens"] > 0
    assert sent["stream"] is True

    assert datas[-1] == "[DONE]"
    text = ""
    for d in datas[:-1]:
        obj = json.loads(d)
        assert obj["object"] == "chat.completion.chunk"
        for ch in obj.get("choices") or []:
            text += (ch.get("delta") or {}).get("content") or ""
    assert text == "hello from fake claude"

    rec = _latest()
    # anthropic usage: input 12 + cache_read 4, output 5
    assert rec["prompt_tokens"] == 16 and rec["completion_tokens"] == 5
    assert rec["cached_tokens"] == 4


# ---------------------------------------------------------------- R6 Anthropic 下游 -> Anthropic 上游（透传）
def test_r6_anthropic_downstream_anthropic_upstream(client, hdr):
    body = {"model": "fakeclaude/fake-claude", "max_tokens": 64, "stream": True,
            "messages": [{"role": "user", "content": "hi"}]}
    with client.stream("POST", "/v1/messages", headers=hdr, json=body) as r:
        assert r.status_code == 200, r.text
        events = sse_parse(r.iter_lines())
    names = [e for e, _ in events]
    assert names[0] == "message_start"
    assert "content_block_delta" in names and names[-1] == "message_stop"
    assert any(e == "message_delta" for e in names)
    text = ""
    for name, data in events:
        if name == "content_block_delta":
            text += json.loads(data)["delta"]["text"]
    assert text == "hello from fake claude"

    rec = _latest()
    assert rec["in_format"] == "anthropic"
    assert rec["total_tokens"] == 21


# ---------------------------------------------------------------- R7 Anthropic 下游 -> OpenAI 上游（流式翻译）
def test_r7_anthropic_downstream_openai_upstream_stream(client, hdr):
    body = {"model": "fake/fake-gpt", "max_tokens": 64, "stream": True,
            "messages": [{"role": "user", "content": "hi"}]}
    with client.stream("POST", "/v1/messages", headers=hdr, json=body) as r:
        assert r.status_code == 200, r.text
        events = sse_parse(r.iter_lines())
    names = [e for e, _ in events]
    assert names[0] == "message_start", names
    assert names[-1] == "message_stop"
    assert "content_block_start" in names and "content_block_stop" in names
    text = ""
    for name, data in events:
        if name == "content_block_delta":
            d = json.loads(data)["delta"]
            assert d["type"] == "text_delta"
            text += d["text"]
    assert text == "hello from fake"
    # message_delta 里带真实 output_tokens
    md = [json.loads(d) for n, d in events if n == "message_delta"][0]
    assert md["delta"]["stop_reason"] == "end_turn"
    assert md["usage"]["output_tokens"] == 7

    rec = _latest()
    assert rec["in_format"] == "anthropic" and rec["platform"] == "fake"


# ---------------------------------------------------------------- R8 非流式双向翻译
def test_r8_nonstream_bidirectional_translation(client, hdr):
    # OpenAI 下游 -> Anthropic 上游
    r = client.post("/v1/chat/completions", headers=hdr, json={
        "model": "fakeclaude/fake-claude", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "hello from fake claude"
    assert body["choices"][0]["finish_reason"] == "stop"

    # Anthropic 下游 -> OpenAI 上游
    r = client.post("/v1/messages", headers=hdr, json={
        "model": "fake/fake-gpt", "max_tokens": 32, "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["type"] == "message" and body["role"] == "assistant"
    assert body["content"][0] == {"type": "text", "text": "hello from fake openai"}
    assert body["stop_reason"] == "end_turn"
    assert body["usage"]["input_tokens"] == 11


# ---------------------------------------------------------------- R9 工具调用翻译
def test_r9_tool_call_translation(client, hdr):
    tools = [{"type": "function", "function": {
        "name": "get_weather", "description": "weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}]

    # OpenAI 格式请求 -> Anthropic 上游：工具定义要变成 input_schema
    r = client.post("/v1/chat/completions", headers=hdr, json={
        "model": "fakeclaude/fake-claude", "tools": tools,
        "messages": [{"role": "user", "content": "北京天气"}]})
    assert r.status_code == 200, r.text
    from tests import fake_upstream
    sent = [x for x in fake_upstream.RECEIVED if x["path"] == "/v1/messages"][-1]["body"]
    assert sent["tools"][0]["name"] == "get_weather"
    assert sent["tools"][0]["input_schema"]["properties"]["city"]["type"] == "string"
    tc = r.json()["choices"][0]["message"]["tool_calls"][0]
    assert tc["function"]["name"] == "get_weather"
    assert json.loads(tc["function"]["arguments"]) == {"city": "Beijing"}
    assert r.json()["choices"][0]["finish_reason"] == "tool_calls"

    # Anthropic 格式请求 -> OpenAI 上游：工具要变成 function 形状
    r = client.post("/v1/messages", headers=hdr, json={
        "model": "fake/fake-gpt", "max_tokens": 64,
        "tools": [{"name": "get_weather", "description": "weather",
                   "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}}],
        "messages": [{"role": "user", "content": [{"type": "text", "text": "北京天气"}]}]})
    assert r.status_code == 200, r.text
    sent = [x for x in fake_upstream.RECEIVED if x["path"] == "/v1/chat/completions"][-1]["body"]
    assert sent["tools"][0]["type"] == "function"
    assert sent["tools"][0]["function"]["name"] == "get_weather"
    block = r.json()["content"][0]
    assert block["type"] == "tool_use" and block["input"] == {"city": "Beijing"}
    assert r.json()["stop_reason"] == "tool_use"


# ---------------------------------------------------------------- R10 模型列表
def test_r10_models_listing(client, hdr):
    r = client.get("/v1/models", headers=hdr)
    assert r.status_code == 200
    ids = [m["id"] for m in r.json()["data"]]
    assert "fake/fake-gpt" in ids and "fake-gpt" in ids and "fakeclaude/fake-claude" in ids
    assert [m for m in r.json()["data"] if m["id"] == "fake/fake-gpt"][0]["owned_by"] == "fake"

    # Anthropic 形状
    r = client.get("/v1/models", headers={"authorization": f"Bearer {conftest._TEST_API_KEY}",
                                          "anthropic-version": "2023-06-01"})
    assert r.status_code == 200
    assert r.json()["data"][0]["type"] == "model"
    assert r.json()["has_more"] is False


# ---------------------------------------------------------------- R11 count_tokens
def test_r11_count_tokens(client, hdr):
    r = client.post("/v1/messages/count_tokens", headers=hdr, json={
        "model": "fakeclaude/fake-claude", "messages": [{"role": "user", "content": "hello"}]})
    assert r.status_code == 200
    assert r.json()["input_tokens"] == 12

    r = client.post("/v1/messages/count_tokens", headers=hdr, json={
        "model": "fake/fake-gpt", "messages": [{"role": "user", "content": "hello world"}]})
    assert r.status_code == 200
    assert r.json()["input_tokens"] > 0 and r.json()["estimated"] is True


# ---------------------------------------------------------------- R12 错误处理
def test_r12_error_paths(client, hdr):
    r = client.post("/v1/chat/completions", headers=hdr, json={
        "model": "nope/nothing", "messages": [{"role": "user", "content": "x"}]})
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "model_not_found"

    r = client.post("/v1/chat/completions", headers=hdr, json={
        "model": "fake/fake-gpt", "fail": "boom", "messages": [{"role": "user", "content": "x"}]})
    assert r.status_code == 400
    assert "boom" in json.dumps(r.json())
    rec = _latest()
    assert rec["status"] == "error" and rec["http_status"] == 400


# ---------------------------------------------------------------- R13 统计聚合
def test_r13_stats_aggregation(client, hdr):
    r = client.get("/api/stats/summary?days=1&group_by=platform", headers=hdr)
    assert r.status_code == 200
    rows = {x["k"]: x for x in r.json()["data"]}
    assert "fake" in rows and "fakeclaude" in rows
    assert rows["fake"]["requests"] > 0 and rows["fake"]["total_tokens"] > 0

    r = client.get("/api/stats/summary?days=1&group_by=model", headers=hdr)
    keys = [x["k"] for x in r.json()["data"]]
    assert any(k.startswith("fake/") for k in keys)

    r = client.get("/api/stats/series?days=7", headers=hdr)
    body = r.json()
    assert len(body["rows"]) == 7 and set(body["platforms"]) >= {"fake", "fakeclaude"}
    assert sum(row["total"] for row in body["rows"]) > 0

    r = client.get("/api/overview", headers=hdr)
    assert r.json()["totals"]["total_requests"] > 0

    r = client.get("/api/stats/export.csv?days=7", headers=hdr)
    assert r.status_code == 200 and "platform" in r.text.splitlines()[0]


# ---------------------------------------------------------------- R14 上游收到注入的 usage 参数
def test_r14_stream_usage_injection(client, hdr):
    from tests import fake_upstream
    body = {"model": "fake/fake-gpt", "stream": True,
            "messages": [{"role": "user", "content": "hi"}]}
    with client.stream("POST", "/v1/chat/completions", headers=hdr, json=body) as r:
        list(r.iter_lines())
    sent = [x for x in fake_upstream.RECEIVED if x["path"] == "/v1/chat/completions"][-1]["body"]
    assert (sent.get("stream_options") or {}).get("include_usage") is True, \
        "中转站应主动要求上游返回 usage，才能精确记账"


# ---------------------------------------------------------------- R16 管理 API
def test_r16_admin_crud(client, hdr):
    r = client.get("/api/providers", headers=hdr)
    assert r.status_code == 200
    platforms = [p["platform"] for p in r.json()["data"]]
    assert "fake" in platforms
    fake = [p for p in r.json()["data"] if p["platform"] == "fake"][0]
    assert "…" in fake["api_keys"][0], "列表里的 key 必须是掩码"
    assert fake["key_count"] == 1

    # 提交掩码 -> 保留原 key
    payload = {"platform": "fake", "name": "Fake OpenAI", "type": "openai",
               "base_url": fake["base_url"], "api_keys": fake["api_keys"],
               "models": fake["models"], "enabled": True}
    r = client.post("/api/providers", headers=hdr, json=payload)
    assert r.status_code == 200, r.text
    from app.main import store, tenants
    reg = tenants.get(store.first_admin()["id"])
    up = reg.find("fake")
    assert up.api_keys == ["sk-fake-openai"], "掩码回传不能覆盖真实 key"

    # 新增一个平台再删掉
    r = client.post("/api/providers", headers=hdr, json={
        "platform": "tmp-prov", "name": "临时", "type": "openai",
        "base_url": "https://example.com/v1", "api_keys": ["sk-x"], "models": ["m1"]})
    assert r.status_code == 200
    assert reg.find("tmp-prov") is not None
    assert client.delete("/api/providers/tmp-prov", headers=hdr).status_code == 200
    assert reg.find("tmp-prov") is None

    # 真实探测 / prov/测试
    r = client.post("/api/providers/fake/refresh-models", headers=hdr)
    assert r.status_code == 200 and r.json()["count"] == 2
    r = client.post("/api/providers/fake/test", headers=hdr)
    assert r.status_code == 200 and r.json()["ok"] is True


# ---------------------------------------------------------------- R17 控制台页面
def test_r17_dashboard_page_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "LLM Relay" in r.text
    assert "api/overview" in r.text


# ---------------------------------------------------------------- R18 model_map 别名
def test_r18_model_alias_routing(client, hdr):
    from tests import fake_upstream

    # ① platform/别名
    r = client.post("/v1/chat/completions", headers=hdr, json={
        "model": "fake/gpt-cheap", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200, r.text
    assert fake_upstream.RECEIVED[-1]["body"]["model"] == "fake-gpt", "别名应被翻成真实模型名"

    # ② 裸别名（不用前缀，也不用把别名写进 models）
    r = client.post("/v1/chat/completions", headers=hdr, json={
        "model": "gpt-cheap", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200, r.text
    assert fake_upstream.RECEIVED[-1]["body"]["model"] == "fake-gpt"

    # ③ 别名也要能在 /v1/models 里被发现
    ids = [m["id"] for m in client.get("/v1/models", headers=hdr).json()["data"]]
    assert "fake/gpt-cheap" in ids and "gpt-cheap" in ids


# ---------------------------------------------------------------- R19 关闭思考：OpenAI 上游（字段上显式关闭）
def test_r19_reasoning_off_openai_upstream(client, hdr):
    from tests import fake_upstream

    body = {"model": "fakereason/fake-reasoner", "stream": True,
            "reasoning_effort": "high",  # 客户端想思考，平台策略强制关闭
            "messages": [{"role": "user", "content": "hi"}]}
    with client.stream("POST", "/v1/chat/completions", headers=hdr, json=body) as r:
        datas = [d for _, d in sse_parse(r.iter_lines())]
    reasoning, text = "", ""
    for d in datas[:-1]:
        for ch in (json.loads(d).get("choices") or []):
            delta = ch.get("delta") or {}
            reasoning += delta.get("reasoning_content") or ""
            text += delta.get("content") or ""
    assert text == "hello from fake", "正文不能丢"
    assert reasoning == "", "上游已经不思考了，自然没有思考内容"

    sent = fake_upstream.RECEIVED[-1]["body"]
    assert sent.get("reasoning_effort") == "none", "必须显式写关闭字段，而不是只把客户端的删掉"
    rec = _latest()
    assert rec["reasoning_policy"] == "off"


# ---------------------------------------------------------------- R20 关闭思考：Anthropic 上游（字段上显式关闭）
def test_r20_reasoning_off_anthropic_upstream(client, hdr):
    from tests import fake_upstream

    body = {"model": "fakethink/fake-claude-thinking", "stream": True,
            "messages": [{"role": "user", "content": "hi"}]}
    with client.stream("POST", "/v1/chat/completions", headers=hdr, json=body) as r:
        datas = [d for _, d in sse_parse(r.iter_lines())]
    reasoning, text = "", ""
    for d in datas[:-1]:
        for ch in (json.loads(d).get("choices") or []):
            delta = ch.get("delta") or {}
            reasoning += delta.get("reasoning_content") or ""
            text += delta.get("content") or ""
    assert text == "hello from fake claude"
    assert reasoning == ""
    sent = fake_upstream.RECEIVED[-1]["body"]
    assert "thinking" not in sent, "Anthropic 上游省略 thinking 就是关闭，不该发 disabled"


# ---------------------------------------------------------------- R21 关闭思考：Anthropic 上下游都是 Anthropic
def test_r21_reasoning_off_anthropic_passthrough(client, hdr):
    from tests import fake_upstream

    body = {"model": "fakethink/fake-claude-thinking", "max_tokens": 64, "stream": True,
            "messages": [{"role": "user", "content": "hi"}]}
    with client.stream("POST", "/v1/messages", headers=hdr, json=body) as r:
        events = sse_parse(r.iter_lines())
    sent = fake_upstream.RECEIVED[-1]["body"]
    assert "thinking" not in sent, "Anthropic 原生协议省略 thinking 即为关闭"
    starts = [json.loads(d) for n, d in events if n == "content_block_start"]
    assert starts and all(blk["content_block"]["type"] == "text" for blk in starts), \
        "上游不思考了，就不该有思考块"
    text = ""
    for n, d in events:
        if n == "content_block_delta":
            text += json.loads(d)["delta"].get("text", "")
    assert text == "hello from fake claude"


# ---------------------------------------------------------------- R22 开启思考（medium）
def test_r22_reasoning_on_anthropic_upstream(client, hdr):
    from tests import fake_upstream

    body = {"model": "fakethinkon/fake-claude-thinking", "stream": True,
            "temperature": 0.3, "max_tokens": 64,
            "messages": [{"role": "user", "content": "hi"}]}
    with client.stream("POST", "/v1/chat/completions", headers=hdr, json=body) as r:
        datas = [d for _, d in sse_parse(r.iter_lines())]
    sent = fake_upstream.RECEIVED[-1]["body"]
    assert sent["thinking"] == {"type": "enabled", "budget_tokens": 4096}
    assert "temperature" not in sent, "开思考时 temperature 必须移除"
    reasoning = ""
    for d in datas[:-1]:
        for ch in (json.loads(d).get("choices") or []):
            reasoning += (ch.get("delta") or {}).get("reasoning_content") or ""
    assert reasoning, "策略不是 off 时思考内容应正常下发"


# ---------------------------------------------------------------- R23 记录来源 IP
def test_r23_client_ip_recorded(client, hdr):
    client.post("/v1/chat/completions", headers=hdr, json={
        "model": "fake/fake-gpt", "messages": [{"role": "user", "content": "hi"}]})
    rec = _latest()
    assert rec["client_ip"], "应记录来源 IP（反代场景靠 X-Forwarded-For 还原）"


# ---------------------------------------------------------------- R24 客户端硬要思考也要被覆盖
def test_r24_client_thinking_overridden_when_off(client, hdr):
    from tests import fake_upstream

    body = {"model": "fakethink/fake-claude-thinking", "max_tokens": 128, "stream": True,
            "thinking": {"type": "enabled", "budget_tokens": 2000},  # 客户端强行要思考
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]}
    with client.stream("POST", "/v1/messages", headers=hdr, json=body) as r:
        events = sse_parse(r.iter_lines())
    sent = fake_upstream.RECEIVED[-1]["body"]
    assert "thinking" not in sent, "客户端传的思考必须被覆盖成关闭（Anthropic 省略即关）"
    starts = [json.loads(d) for n, d in events if n == "content_block_start"]
    assert starts and all(b["content_block"]["type"] == "text" for b in starts)


# ---------------------------------------------------------------- R25 模型级思考覆盖
def test_r25_model_level_reasoning_off(client, hdr):
    """同一平台里只关某个模型的思考，别的模型照旧。"""
    from tests import fake_upstream

    # ① 被单独关掉的模型：出站必须带关闭字段
    with client.stream("POST", "/v1/chat/completions", headers=hdr, json={
            "model": "fakemodel/fake-reasoner", "stream": True,
            "reasoning_effort": "high",
            "messages": [{"role": "user", "content": "hi"}]}) as r:
        datas = [d for _, d in sse_parse(r.iter_lines())]
    sent = fake_upstream.RECEIVED[-1]["body"]
    assert sent.get("reasoning_effort") == "none", "模型级策略应覆盖客户端的 effort"
    reasoning = "".join(
        (ch.get("delta") or {}).get("reasoning_content") or ""
        for d in datas[:-1] for ch in (json.loads(d).get("choices") or []))
    assert reasoning == ""
    assert _latest()["reasoning_policy"] == "off"

    # ② 同平台的另一个模型：不受影响，客户端值原样透传
    #    （这个平台 extra 里还躺着 "reasoning": "off"，必须被忽略 —— 平台级开关已取消）
    with client.stream("POST", "/v1/chat/completions", headers=hdr, json={
            "model": "fakemodel/fake-gpt", "stream": True,
            "reasoning_effort": "high",
            "messages": [{"role": "user", "content": "hi"}]}) as r:
        list(r.iter_lines())
    sent = fake_upstream.RECEIVED[-1]["body"]
    assert sent.get("reasoning_effort") == "high", "平台级 thinking 已废弃，不该影响任何模型"
    assert _latest()["reasoning_policy"] == "auto"


# ---------------------------------------------------------------- R31 模型级可指定"关闭写法"
def test_r31_model_level_off_field(client, hdr):
    from tests import fake_upstream

    with client.stream("POST", "/v1/chat/completions", headers=hdr, json={
            "model": "fakefield/fake-gpt", "stream": True,
            "messages": [{"role": "user", "content": "hi"}]}) as r:
        list(r.iter_lines())
    sent = fake_upstream.RECEIVED[-1]["body"]
    assert sent.get("enable_thinking") is False, "模型级指定的关闭写法应生效"
    assert "reasoning_effort" not in sent, "不该同时用自动选出来的 reasoning_effort"


# ---------------------------------------------------------------- R32 按平台自动挑关闭字段（同 Hermes）
def test_r32_auto_off_payload_per_platform():
    from types import SimpleNamespace

    from app import reasoning

    def up(platform, url, name=""):
        return SimpleNamespace(platform=platform, name=name or platform, base_url=url)

    cases = [
        ("ollama", "https://ollama.com/v1", {"reasoning_effort": "none", "think": False}),
        ("local", "http://127.0.0.1:11434/v1", {"reasoning_effort": "none", "think": False}),
        ("zhipu", "https://open.bigmodel.cn/api/paas/v4", {"thinking": {"type": "disabled"}}),
        ("qwen", "https://dashscope.aliyuncs.com/compatible-mode/v1", {"enable_thinking": False}),
        ("openrouter", "https://openrouter.ai/api/v1", {"reasoning": {"enabled": False, "effort": "none"}}),
        ("kimi", "https://api.moonshot.cn/v1", {"thinking": {"type": "disabled"}}),
        ("agnes", "https://apihub.agnes-ai.com/v1", {"reasoning_effort": "none"}),
    ]
    for platform, url, want in cases:
        got = reasoning.auto_off_payload(up(platform, url), "openai")
        assert got == want, f"{platform} 的关闭字段不对: {got} != {want}"

    # Anthropic 原生协议没有 disabled 类型，省略 thinking 就是关闭
    assert reasoning.auto_off_payload(up("claude", "https://api.anthropic.com"), "anthropic") == {}
    # GLM-5.3 拒绝 disabled，只能降到最低档
    assert reasoning.auto_off_payload(
        up("zhipu", "https://api.z.ai/api/paas/v4"), "openai", "glm-5.3") == {"reasoning_effort": "low"}


# ---------------------------------------------------------------- R33 客户端开思考，上游收到的是关闭（端到端）
def test_r33_client_thinking_on_upstream_gets_off(client, hdr):
    from tests import fake_upstream

    with client.stream("POST", "/v1/chat/completions", headers=hdr, json={
            "model": "fakeollama/beam", "stream": True, "reasoning_effort": "high",
            "messages": [{"role": "user", "content": "hi"}]}) as r:
        list(r.iter_lines())
    sent = fake_upstream.RECEIVED[-1]["body"]
    assert sent.get("reasoning_effort") == "none", f"上游应收到 none，实际 {sent.get('reasoning_effort')}"
    assert sent.get("think") is False, "Ollama 还要发 think:false"


# ---------------------------------------------------------------- R35 web 会话自动按对话隔离


# ---------------------------------------------------------------- R36 用量统计带 API Key 维度
def test_r36_usage_group_by_api_key(client, hdr):
    from app.gateway import mask_key

    client.post("/v1/chat/completions", headers=hdr, json={
        "model": "fakemulti/multi-gpt", "messages": [{"role": "user", "content": "hi"}]})
    rec = _latest()
    assert rec["key_hint"], "每条记录都要写明用了哪个上游 key"
    assert rec["key_hint"] == mask_key("sk-multi-aaaa1111")
    assert "…" in rec["key_hint"], "key 要打码，不能明文入库"

    r = client.get("/api/stats/summary?days=1&group_by=key", headers=hdr).json()
    rows = [x for x in r["data"] if "fakemulti" in x["k"]]
    assert rows, f"按 key 聚合必须有 fakemulti 的行: {r['data']}"
    assert any(rec["key_hint"] in x["k"] for x in rows), f"聚合结果要带上 key: {rows}"

    # 两个 key 在配置里，统计能区分出当前用的是第一个
    ids = [m["id"] for m in client.get("/v1/models", headers=hdr).json()["data"]]
    assert "fakemulti/multi-gpt" in ids


# ---------------------------------------------------------------- R37 曲线支持按维度取（前端多选联动用）
def test_r37_series_group_by(client, hdr):
    client.post("/v1/chat/completions", headers=hdr, json={
        "model": "fake/fake-gpt", "messages": [{"role": "user", "content": "hi"}]})
    client.post("/v1/chat/completions", headers=hdr, json={
        "model": "fakemulti/multi-gpt", "messages": [{"role": "user", "content": "hi"}]})

    by_model = client.get("/api/stats/series?days=1&group_by=model", headers=hdr).json()
    assert by_model["keys"], by_model
    assert "fakemulti/multi-gpt" in by_model["keys"]
    assert len(by_model["rows"]) == 1 and by_model["rows"][0]["day"]
    # rows 里每个 key 都是一列，前端直接画
    for k in by_model["keys"]:
        assert k in by_model["rows"][0]

    by_plat = client.get("/api/stats/series?days=1&group_by=platform", headers=hdr).json()
    assert "fake" in by_plat["keys"] and "fakemulti" in by_plat["keys"]
    assert by_plat["platforms"] == by_plat["keys"], "platforms 是 keys 的别名，老代码继续可用"

    by_key = client.get("/api/stats/series?days=1&group_by=key", headers=hdr).json()
    assert any("sk-" in k or "·" in k for k in by_key["keys"]), by_key["keys"]

    total = client.get("/api/stats/series?days=1&group_by=total", headers=hdr).json()
    assert total["keys"] == ["合计"]

    # 未知维度回落到按平台，不报错
    fallback = client.get("/api/stats/series?days=1&group_by=whatever", headers=hdr).json()
    assert "fake" in fallback["keys"]


# ---------------------------------------------------------------- R38 反代子路径（/llmrelay/v1）
def test_r38_base_path_prefix(client, hdr):
    from app import config

    old = config.get_base_path()
    try:
        assert config.set_base_path("/llmrelay/") == "/llmrelay", "要自动规整成 /llmrelay"
        # 带前缀：模型列表 / 健康检查 / 控制台
        assert client.get("/llmrelay/v1/models", headers=hdr).status_code == 200
        assert client.get("/llmrelay/healthz").status_code == 200
        assert client.get("/llmrelay/").status_code == 200
        # 不带前缀照样能用（本地直连不受影响）
        assert client.get("/v1/models", headers=hdr).status_code == 200
        assert client.get("/healthz").status_code == 200
        # /llmrelay 没有斜杠 → 307 到 /llmrelay/（否则控制台相对路径算错）
        r = client.get("/llmrelay", follow_redirects=False)
        assert r.status_code in (307, 308) and r.headers["location"].endswith("/llmrelay/")
        # 转发照常
        r = client.post("/llmrelay/v1/chat/completions", headers=hdr, json={
            "model": "fake/fake-gpt", "messages": [{"role": "user", "content": "hi"}]})
        assert r.status_code == 200
    finally:
        config.set_base_path(old)
    # 清空后前缀失效
    assert client.get("/llmrelay/v1/models", headers=hdr).status_code == 404




# ---------------------------------------------------------------- R39 多账号：注册登录 + 数据隔离
def _session_cookie(client, email, password):
    """登录拿会话 cookie，并清空 client 自带的 cookie jar（免得污染别的测试）。"""
    r = client.post("/api/auth/login", json={"email": email, "password": password})
    assert r.status_code == 200, r.text
    tok = r.cookies.get("relay_session") or ""
    client.cookies.clear()
    assert tok, "登录必须下发会话 cookie"
    return {"relay_session": tok}


def test_r39_accounts_register_login_isolation(client, hdr):
    fake = [p for p in client.get("/api/providers", headers=hdr).json()["data"]
            if p["platform"] == "fake"][0]

    # 1) 注册（库里已有 admin，所以 bob 是普通账号，且需邮箱确认）
    r = client.post("/api/auth/register", json={"email": "bob@test.local", "password": "bob-password"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["user"]["role"] == "user", "第一个之后的注册者不是管理员"
    assert body["pending"] is True and body["api_key"] == "", "未确认账号：无会话、不发 Key"
    assert not r.cookies.get("relay_session")
    # 点邮件里的确认链接（magic link）→ active + 默认 Key，再登录拿会话
    import app.auth as _A
    store = client.app.state.store
    bob = store.get_user_by_email("bob@test.local")
    link = _A.build_email_link(store, bob, "http://testserver")
    assert client.get(link).status_code == 200
    assert store.get_user_by_email("bob@test.local")["status"] == "active"
    bob_key = store.list_api_keys(bob["id"])[0]["key"]
    assert bob_key.startswith("sk-relay-")
    r = client.post("/api/auth/login", json={"email": "bob@test.local", "password": "bob-password"})
    assert r.status_code == 200, "确认后正常登录"
    cb = {"relay_session": r.cookies.get("relay_session")}
    client.cookies.clear()

    # 2) 会话生效
    st = client.get("/api/auth/status", cookies=cb).json()
    assert st["user"] and st["user"]["email"] == "bob@test.local"

    # 3) 上游平台隔离
    bob_platforms = [x["platform"] for x in client.get("/api/providers", cookies=cb).json()["data"]]
    assert "fake" not in bob_platforms, f"平台必须隔离: {bob_platforms}"

    # 4) 下游 key 隔离
    bh = {"authorization": f"Bearer {bob_key}"}
    ids = [m["id"] for m in client.get("/v1/models", headers=bh).json()["data"]]
    assert "fake/fake-gpt" not in ids, f"模型必须隔离: {ids}"
    r = client.post("/v1/chat/completions", headers=bh, json={
        "model": "fake/fake-gpt", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 404, "不能跨账号调用别人的上游"

    # 5) bob 自己加平台后就能用
    r = client.post("/api/providers", cookies=cb, json={
        "platform": "bobplat", "name": "Bob 的", "type": "openai",
        "base_url": fake["base_url"], "api_keys": ["sk-bob-up"], "models": ["bob-model"]})
    assert r.status_code == 200, r.text
    assert "bobplat/bob-model" in [m["id"] for m in client.get("/v1/models", headers=bh).json()["data"]]
    r = client.post("/v1/chat/completions", headers=bh, json={
        "model": "bobplat/bob-model", "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200, r.text

    # 6) 普通账号不能调管理员接口
    assert client.get("/api/users", cookies=cb).status_code == 403

    # 7) 管理员能看到所有账号，并能按账号看数据
    users = client.get("/api/users", headers=hdr).json()["data"]
    emails = [u["email"] for u in users]
    assert "bob@test.local" in emails and len(users) >= 2
    bob_id = [u["id"] for u in users if u["email"] == "bob@test.local"][0]
    mine = client.get(f"/api/stats/summary?days=1&group_by=platform&tenant={bob_id}",
                      headers=hdr).json()
    assert any("bobplat" in row["k"] for row in mine["data"]), mine
    allrows = client.get("/api/stats/summary?days=1&group_by=platform&tenant=all",
                         headers=hdr).json()
    assert len(allrows["data"]) >= len(mine["data"])


# ---------------------------------------------------------------- R40 账号管理：建号/禁用/改密
def test_r40_admin_user_management(client, hdr):
    r = client.post("/api/users", headers=hdr, json={
        "email": "carol@test.local", "password": "carol-password", "role": "user"})
    assert r.status_code == 200, r.text
    carol_key = r.json()["api_key"]
    uid = r.json()["user"]["id"]

    # 登录 / 退出 / 错密码
    cc = {"relay_session": ""}
    r = client.post("/api/auth/login", json={"email": "carol@test.local", "password": "carol-password"})
    assert r.status_code == 200, r.text
    cc["relay_session"] = r.cookies.get("relay_session")
    client.cookies.clear()
    assert client.get("/api/auth/status", cookies=cc).json()["user"]["email"] == "carol@test.local"
    assert client.post("/api/auth/login", json={
        "email": "carol@test.local", "password": "wrong-password"}).status_code == 401
    client.cookies.clear()
    assert client.post("/api/auth/logout", cookies=cc).status_code == 200
    assert client.get("/api/auth/status", cookies=cc).json()["user"] is None

    # 禁用 → key 立即失效
    assert client.patch(f"/api/users/{uid}", headers=hdr,
                        json={"status": "disabled"}).status_code == 200
    assert client.get("/v1/models", headers={"authorization": f"Bearer {carol_key}"}).status_code == 401
    assert client.patch(f"/api/users/{uid}", headers=hdr,
                        json={"status": "active"}).status_code == 200
    assert client.get("/v1/models", headers={"authorization": f"Bearer {carol_key}"}).status_code == 200

    # 重置密码
    assert client.patch(f"/api/users/{uid}", headers=hdr,
                        json={"password": "carol-new-password"}).status_code == 200
    r = client.post("/api/auth/login", json={
        "email": "carol@test.local", "password": "carol-new-password"})
    assert r.status_code == 200
    client.cookies.clear()

    # 不能把自己禁用/删除
    me = client.get("/api/auth/status", headers=hdr).json()["user"]["id"]
    assert client.patch(f"/api/users/{me}", headers=hdr,
                        json={"status": "disabled"}).status_code == 400
    assert client.delete(f"/api/users/{me}", headers=hdr).status_code == 400


# ---------------------------------------------------------------- R41 我的 API Key 管理
def test_r41_api_key_management(client, hdr):
    keys = client.get("/api/keys", headers=hdr).json()["data"]
    assert keys, "每个账号至少有一把 key"
    assert "…" in keys[0]["key"], "列表里的 key 必须打码"

    r = client.post("/api/keys", headers=hdr, json={"label": "测试用"})
    assert r.status_code == 200
    newkey = r.json()["key"]
    kid = r.json()["id"]
    assert newkey.startswith("sk-relay-")
    assert client.get("/v1/models", headers={"authorization": f"Bearer {newkey}"}).status_code == 200

    assert client.delete(f"/api/keys/{kid}", headers=hdr).status_code == 200
    assert client.get("/v1/models", headers={"authorization": f"Bearer {newkey}"}).status_code == 401


# ---------------------------------------------------------------- R42 复制 / 删除 key
def test_r42_key_copy_and_delete(client, hdr):
    r = client.post("/api/keys", headers=hdr, json={"label": "要复制的"})
    kid, plain = r.json()["id"], r.json()["key"]

    # 列表里是掩码
    listed = [k for k in client.get("/api/keys", headers=hdr).json()["data"] if k["id"] == kid][0]
    assert "…" in listed["key"] and listed["key"] != plain

    # reveal 取回明文，能直接用来调下游
    rv = client.get(f"/api/keys/{kid}/reveal", headers=hdr)
    assert rv.status_code == 200 and rv.json()["key"] == plain
    assert client.get("/v1/models", headers={"authorization": f"Bearer {rv.json()['key']}"}).status_code == 200

    # 别的账号看不到这把 key（普通账号 → 404）
    r = client.post("/api/users", headers=hdr, json={
        "email": "dave@test.local", "password": "dave-password"})
    dave_id = r.json()["user"]["id"]
    dave_key = r.json()["api_key"]
    # 用 dave 的 key 当 Bearer 打 /api（/api 也认 Bearer）
    assert client.get(f"/api/keys/{kid}/reveal",
                      headers={"authorization": f"Bearer {dave_key}"}).status_code == 404

    # 真删除：记录消失、reveal 404、key 立即失效
    assert client.delete(f"/api/keys/{kid}", headers=hdr).status_code == 200
    assert kid not in [k["id"] for k in client.get("/api/keys", headers=hdr).json()["data"]]
    assert client.get(f"/api/keys/{kid}/reveal", headers=hdr).status_code == 404
    assert client.get("/v1/models", headers={"authorization": f"Bearer {plain}"}).status_code == 401
    # 重复删除 → 404
    assert client.delete(f"/api/keys/{kid}", headers=hdr).status_code == 404


# ---------------------------------------------------------------- R43 新账号不能继承管理员的上游
def test_r43_new_account_upstreams_isolated_after_restart(client, hdr):
    """模拟重启：新建一个 TenantRegistries，确认新账号拿到的是空配置。

    回归的是这个 bug：认领标记原来是内存变量，服务重启后归零，
    下一个新注册的账号会被塞一份管理员的上游配置。
    """
    from app import config
    from app.main import store, tenants
    from app.tenants import TenantRegistries

    admin = store.first_admin()
    assert admin, "测试库里应该有管理员"
    assert tenants.get(admin["id"]).upstreams, "管理员应该有平台"

    r = client.post("/api/users", headers=hdr, json={
        "email": "eve@test.local", "password": "eve-password"})
    assert r.status_code == 200, r.text
    uid = r.json()["user"]["id"]

    # 关键：全新的池子 = 重启后的状态
    fresh = TenantRegistries(config.CONFIG_DIR / "tenants", config.UPSTREAMS_FILE, store)
    reg = fresh.get(uid)
    assert reg.upstreams == [], \
        f"新账号不该继承管理员的上游，实际拿到: {[u.platform for u in reg.upstreams]}"

    # 认领标记落盘了，之后任何新账号都不会再被塞
    assert (config.CONFIG_DIR / "tenants" / ".legacy_claimed").exists()
    # 管理员的配置不受影响
    assert [u.platform for u in tenants.get(admin["id"]).upstreams]


# ---------------------------------------------------------------- R44 管理员视角：按账号维度
def test_r44_account_dimension(client, hdr):
    from app.main import store

    admin = store.first_admin()
    admin_email = admin["email"]

    # 先制造一条属于 admin 的记录（下面用 admin 账号 key 发一条到假上游）
    client.post("/v1/chat/completions", headers=hdr, json={
        "model": "fake/fake-gpt", "messages": [{"role": "user", "content": "hi"}]})

    # 1) summary 按账号：k 是邮箱，不是 id
    r = client.get("/api/stats/summary?days=1&group_by=account&tenant=all", headers=hdr).json()
    keys = [row["k"] for row in r["data"]]
    assert admin_email in keys, f"按账号聚合要显示邮箱: {keys}"
    assert not any(k.isdigit() for k in keys), f"不该是裸 id: {keys}"

    # 2) series 按账号：keys 也是邮箱，rows 里同名列对得上
    d = client.get("/api/stats/series?days=1&group_by=account&tenant=all", headers=hdr).json()
    assert admin_email in d["keys"], d["keys"]
    assert all(k in d["rows"][0] for k in d["keys"]), "rows 里必须有同名数据列"

    # 3) 请求日志带账号邮箱
    recs = client.get("/api/stats/records?limit=5", headers=hdr).json()["data"]
    assert recs and recs[0]["user_email"] == admin_email, recs[0] if recs else None

    # 4) 普通账号看不到别人：自己查自己，账号维度只有他自己
    r2 = client.post("/api/users", headers=hdr, json={
        "email": "frank@test.local", "password": "frank-password"})
    frank_key = r2.json()["api_key"]
    mine = client.get("/api/stats/summary?days=1&group_by=account",
                      headers={"authorization": f"Bearer {frank_key}"}).json()
    assert [row["k"] for row in mine["data"]] in ([], ["frank@test.local"]), mine


# ---------------------------------------------------------------- R45 账号页权限：自己 / 最后一个管理员
def test_r45_user_management_permissions(client, hdr):
    me = client.get("/api/auth/status", headers=hdr).json()["user"]
    # 不能禁用自己、不能降级自己
    assert client.patch(f"/api/users/{me['id']}", headers=hdr,
                        json={"status": "disabled"}).status_code == 400
    assert client.patch(f"/api/users/{me['id']}", headers=hdr,
                        json={"role": "user"}).status_code == 400

    # 可以给别人升管理员，再降回来
    r = client.post("/api/users", headers=hdr, json={
        "email": "gina@test.local", "password": "gina-password"})
    gid = r.json()["user"]["id"]
    assert client.patch(f"/api/users/{gid}", headers=hdr,
                        json={"role": "admin"}).status_code == 200
    assert [u for u in client.get("/api/users", headers=hdr).json()["data"]
            if u["id"] == gid][0]["role"] == "admin"
    assert client.patch(f"/api/users/{gid}", headers=hdr,
                        json={"role": "user"}).status_code == 200


# ---------------------------------------------------------------- R46 流式也要记下用的是哪把上游 key
def test_r46_stream_records_key_hint(client, hdr):
    """回归：流式路径的 record() 漏传 resp，导致 key_hint 一直是空串（前端显示 "-"）。"""
    r = client.post("/v1/chat/completions", headers=hdr, json={
        "model": "fake/fake-gpt", "stream": True,
        "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200, r.text

    recs = client.get("/api/stats/records?limit=5", headers=hdr).json()["data"]
    streamed = [x for x in recs if x["platform"] == "fake" and x["stream"]]
    assert streamed, "应该有一条流式记录"
    for rec in streamed[:3]:
        assert rec["key_hint"], f"流式记录丢了 key_hint: {rec}"
        assert "…" in rec["key_hint"], rec["key_hint"]


# ---------------------------------------------------------------- R47 空的环境变量不能盖掉 settings.json
def test_r47_empty_env_base_path(tmp_path, monkeypatch):
    """docker compose 里 RELAY_BASE_PATH: ${RELAY_BASE_PATH:-} 会注入空串。

    空串必须当成「没设」，否则控制台里设置 wrote settings.json 也不生效
    （表现为：设置页空白、当前生效显示根目录，但文件里明明有值）。
    """
    from app import config

    sf = tmp_path / "settings.json"
    monkeypatch.setattr(config, "SETTINGS_FILE", sf)
    config.set_base_path("/llmrelay")
    assert sf.exists()

    # 1) 环境变量存在但为空串 → 用文件里的值
    monkeypatch.setenv("RELAY_BASE_PATH", "")
    assert config.get_base_path() == "/llmrelay"
    assert config.base_path_env_locked() is False

    # 2) 环境变量是纯空格 → 同上
    monkeypatch.setenv("RELAY_BASE_PATH", "   ")
    assert config.get_base_path() == "/llmrelay"

    # 3) 环境变量有值 → 以环境变量为准，并标记为锁定
    monkeypatch.setenv("RELAY_BASE_PATH", "/other")
    assert config.get_base_path() == "/other"
    assert config.base_path_env_locked() is True

# 4) 环境变量真的不存在 → 用文件
    monkeypatch.delenv("RELAY_BASE_PATH", raising=False)
    assert config.get_base_path() == "/llmrelay"


# ------------------------------------------------- R48 访问路径只有管理员能看能改
def test_r48_base_path_admin_only(client, hdr):
    """普通账号设置页不该出现「访问路径」，后端也必须挡住（不能只靠前端隐藏）。"""
    from app import config

    old = config.get_base_path()
    r = client.post("/api/auth/register", json={"email": "erin@test.local",
                                                "password": "erin-password"})
    assert r.status_code == 200, r.text
    assert r.json()["pending"] is True
    # 点确认链接激活后登录拿会话
    import app.auth as _A
    store = client.app.state.store
    erin = store.get_user_by_email("erin@test.local")
    assert client.get(_A.build_email_link(store, erin, "http://testserver")).status_code == 200
    r = client.post("/api/auth/login", json={"email": "erin@test.local",
                                             "password": "erin-password"})
    assert r.status_code == 200
    cb = {"relay_session": r.cookies.get("relay_session")}
    client.cookies.clear()

    # 1) 前端据 is_admin 整块隐藏
    st = client.get("/api/settings", cookies=cb).json()
    assert st["is_admin"] is False

    # 2) 普通账号直接打接口 → 403，且配置没被改
    r = client.post("/api/settings/base-path", cookies=cb, json={"base_path": "/hack"})
    assert r.status_code == 403, r.text
    assert config.get_base_path() == old, "普通账号不能改访问路径"

    # 3) 未登录同样打不进去
    client.cookies.clear()
    assert client.post("/api/settings/base-path",
                       json={"base_path": "/hack"}).status_code in (401, 403)
    assert config.get_base_path() == old

    # 4) 管理员照旧能改（改完还原）
    try:
        r = client.post("/api/settings/base-path", headers=hdr, json={"base_path": "/llmrelay"})
        assert r.status_code == 200, r.text
        assert r.json()["base_path"] == "/llmrelay"
        assert client.get("/api/settings", headers=hdr).json()["base_path"] == "/llmrelay"
    finally:
        config.set_base_path(old)


# ---------------------------------------------------------------- R49 模型下拉：只读拉取（不写回配置）
def test_r49_available_models_readonly(client, hdr):
    from app.main import store, tenants

    reg = tenants.get(store.first_admin()["id"])
    fake = [u for u in reg.upstreams if u.platform == "fake"][0]
    original = list(fake.models)
    reg.set_models("fake", ["fake-gpt"])   # 故意少登一个，验证只读拉取不会写回

    r = client.get("/api/providers/fake/available-models", headers=hdr)
    assert r.status_code == 200, r.text
    assert set(r.json()["models"]) == {"fake-gpt", "fake-gpt-2"}

    reg.reload()
    fake_after = [u for u in reg.upstreams if u.platform == "fake"][0]
    assert list(fake_after.models) == ["fake-gpt"], "只读拉取不得写回配置"

    reg.set_models("fake", original)        # 还原
    reg.reload()


# ---------------------------------------------------------------- R50 model_map 保存往返
def test_r50_provider_model_map_roundtrip(client, hdr):
    data = client.get("/api/providers", headers=hdr).json()["data"]
    fake = [p for p in data if p["platform"] == "fake"][0]
    payload = {"platform": "fake", "name": fake["name"], "type": fake["type"],
               "base_url": fake["base_url"], "api_keys": fake["api_keys"],
               "models": fake["models"], "model_map": {"fast": "fake-gpt"},
               "enabled": True, "timeout_s": fake["timeout_s"]}
    r = client.post("/api/providers", headers=hdr, json=payload)
    assert r.status_code == 200, r.text
    data2 = client.get("/api/providers", headers=hdr).json()["data"]
    fake2 = [p for p in data2 if p["platform"] == "fake"][0]
    assert fake2["model_map"] == {"fast": "fake-gpt"}


# ---------------------------------------------------------------- R51 日志分页（offset 与全量切片一致）
def test_r51_records_pagination(client, hdr):
    for i in range(5):
        r = client.post("/v1/chat/completions", headers=hdr,
                        json={"model": "fake-gpt",
                              "messages": [{"role": "user", "content": f"page {i}"}]})
        assert r.status_code == 200, r.text
    full = client.get("/api/stats/records?limit=100&offset=0", headers=hdr).json()["data"]
    assert len(full) >= 5
    page2 = client.get("/api/stats/records?limit=2&offset=3", headers=hdr)
    assert page2.status_code == 200, page2.text
    body = page2.json()
    assert len(body["data"]) == 2
    assert [x["id"] for x in body["data"]] == [x["id"] for x in full[3:5]], "offset 分页必须与全量切片一致"
    assert body["total"] >= 5


# ---------------------------------------------------------------- R52 近 N 天活动（概览看板）
def test_r52_stats_hourly(client, hdr):
    from datetime import datetime
    ok = client.post("/v1/chat/completions", headers=hdr,
                     json={"model": "fake-gpt", "messages": [{"role": "user", "content": "hourly ok"}]})
    assert ok.status_code == 200, ok.text
    # 假上游对 body 里带 fail 字段的请求回 400（openai→openai 全量透传），落一条 error 记录
    err = client.post("/v1/chat/completions", headers=hdr,
                      json={"model": "fake-gpt", "fail": "deliberate",
                            "messages": [{"role": "user", "content": "hourly err"}]})
    assert err.status_code == 400, err.text

    # 今日按小时：24 桶，当前小时必须包含刚打的 ok+err 两条
    data = client.get("/api/stats/hourly?days=1&group=hour", headers=hdr).json()
    buckets = data["buckets"]
    assert len(buckets) == 24, "今日必须补齐 24 个小时桶"
    assert [h["h"] for h in buckets[:3]] == ["00", "01", "02"]
    # 刚打的 ok+err 落在当前小时或跨整点的下一小时，用后缀断言避免边界竞态
    h0 = int(datetime.now().strftime("%H"))
    assert sum(b["requests"] for b in buckets[h0:]) >= 2, "当前小时起的桶必须包含刚打的两条"
    assert sum(b["errors"] for b in buckets[h0:]) >= 1
    t = data["today"]
    assert t["requests"] == sum(h["requests"] for h in buckets), "今日合计 = 各小时之和"
    assert t["requests"] >= 2 and t["errors"] >= 1
    assert t["tokens"] >= 0 and isinstance(t["avg_latency_ms"], (int, float))

    # 近 7 天按小时：168 桶，合计与今日口径一致（今日是 7 天的子集）
    d7 = client.get("/api/stats/hourly?days=7&group=hour", headers=hdr).json()
    assert len(d7["buckets"]) == 7 * 24
    assert d7["total"]["requests"] >= t["requests"]
    assert d7["today"] is None, "非今日口径不应带 today 字段"

    # 近 3 天按天：3 个桶（YYYY-MM-DD），合计 = 各天之和
    d3 = client.get("/api/stats/hourly?days=3&group=day", headers=hdr).json()
    assert len(d3["buckets"]) == 3
    for b in d3["buckets"]:
        assert len(b["h"]) == 10 and b["h"][4] == "-"
    assert d3["total"]["requests"] == sum(b["requests"] for b in d3["buckets"])


# ---------------------------------------------------------------- R53 用量统计多维端点（period/latency/heatmap/mix/errors）
def test_r53_stats_analytics_endpoints(client, hdr):
    ok = client.post("/v1/chat/completions", headers=hdr,
                     json={"model": "fake-gpt", "messages": [{"role": "user", "content": "p ok"}]})
    assert ok.status_code == 200, ok.text
    st = client.post("/v1/chat/completions", headers=hdr,
                     json={"model": "fake-gpt", "stream": True,
                            "messages": [{"role": "user", "content": "p stream"}]})
    assert st.status_code == 200, st.text
    err = client.post("/v1/chat/completions", headers=hdr,
                      json={"model": "fake-gpt", "fail": "deliberate",
                            "messages": [{"role": "user", "content": "p err"}]})
    assert err.status_code == 400, err.text

    p = client.get("/api/stats/period?days=1", headers=hdr).json()
    assert p["requests"] >= 3 and p["errors"] >= 1
    assert p["stream_requests"] >= 1
    assert p["avg_latency_ms"] >= 0 and p["p95_latency_ms"] >= p["avg_latency_ms"]
    assert 0 <= p["cache_rate"] <= 1 and 0 <= p["success_rate"] <= 1
    assert 0 <= p["stream_rate"] <= 1

    lat = client.get("/api/stats/latency?days=1", headers=hdr).json()
    assert lat["keys"] == ["avg", "p95", "ttft"] and len(lat["rows"]) == 1
    assert lat["rows"][0]["requests"] >= 2 and lat["rows"][0]["ttft"] >= 0

    hm = client.get("/api/stats/heatmap?days=1", headers=hdr).json()
    assert len(hm["matrix"]) == 7 and all(len(r) == 24 for r in hm["matrix"])
    assert sum(v for r in hm["matrix"] for v in r) >= 3

    mx = client.get("/api/stats/mix?days=1", headers=hdr).json()
    assert sum(mx["by_stream"].values()) == p["requests"]
    assert mx["by_stream"].get("流式", 0) >= 1, "流式请求必须单独成组（GROUP BY 不能漏）"
    assert mx["by_status"].get("ok", 0) >= 2
    assert str(400) in (mx["by_http_error"] or {}), "故意制造的 400 应进错误码分布"

    es = client.get("/api/stats/errors?days=1&limit=8", headers=hdr).json()["data"]
    assert es and "fake upstream rejected" in es[0]["reason"]
    assert es[0]["http_status"] == 400 and es[0]["c"] >= 1

def test_r54_logs_global_filters(client, hdr):
    """日志筛选作用于全量记录（跨页）+ 日历日期区间 + facets 全量选项。"""
    from datetime import datetime, timedelta
    from app.main import store

    # admin 不带 tenant 参数时数据视角=自己，合成记录要挂到该账号名下
    uid = client.get("/api/settings", headers=hdr).json()["user"]["id"]
    now = datetime.now().astimezone()
    three_days_ago = (now - timedelta(days=3)).isoformat(timespec="seconds")
    kw = {"user_id": uid}
    store.record(ts=three_days_ago, platform="zt", model="zt-a", stream=1,
                 status="ok", client_ip="10.1.1.1", key_hint="zt-key",
                 prompt_tokens=1, completion_tokens=2, total_tokens=3, **kw)
    store.record(ts=three_days_ago, platform="zt", model="zt-b", stream=0,
                 status="ok", client_ip="10.1.1.2", key_hint="zt-key",
                 prompt_tokens=1, completion_tokens=1, total_tokens=2, **kw)
    store.record(ts=now.isoformat(timespec="seconds"), platform="zt", model="zt-a",
                 stream=1, status="500", http_status=500, error="boom",
                 client_ip="10.1.1.3", key_hint="zt-other",
                 prompt_tokens=1, completion_tokens=0, total_tokens=1, **kw)
    store.record(ts=now.isoformat(timespec="seconds"), platform="other-zt", model="zt-c",
                 stream=0, status="ok", client_ip="10.1.1.4", key_hint="zt-key",
                 prompt_tokens=1, completion_tokens=1, total_tokens=2, **kw)

    f = client.get("/api/stats/logs/facets", headers=hdr).json()
    assert "zt" in f["platform"] and "zt-key" in f["key_hint"]
    assert "10.1.1.1" in f["client_ip"] and f["total"] >= 4

    # 平台过滤：跨页全量生效，total 与筛选后条数一致
    r = client.get("/api/stats/records?platform=zt", headers=hdr).json()
    assert r["total"] == 3 and len(r["data"]) == 3
    assert all(x["platform"] == "zt" for x in r["data"])

    # 多选 + 类型 + 日期区间（日历控件传 YYYY-MM-DD）
    today = now.strftime("%Y-%m-%d")
    past = (now - timedelta(days=3)).strftime("%Y-%m-%d")
    r2 = client.get("/api/stats/records?platform=zt,other-zt&stream=0,1&from="
                    + past + "&to=" + today, headers=hdr).json()
    assert r2["total"] == 4

    r3 = client.get("/api/stats/records?platform=zt&stream=1&from="
                    + today, headers=hdr).json()   # 只要今天的流式
    assert r3["total"] == 1 and r3["data"][0]["status"] == "500"

    r4 = client.get("/api/stats/records?status=ok,500&key=zt-key", headers=hdr).json()
    assert all(x["key_hint"] == "zt-key" for x in r4["data"])
    assert r4["total"] >= 3

def test_r55_hourly_split(client, hdr):
    """近期活动图的平台/模型维度拆分：每桶 by_platform / by_model，合计与 total 对得上。"""
    r = client.get("/api/stats/hourly?days=1&group=hour&split=platform", headers=hdr).json()
    assert all("by_platform" in b for b in r["buckets"])
    tot = sum(v for b in r["buckets"] for v in b["by_platform"].values())
    assert tot == r["total"]["tokens"], "各桶平台拆分之和应等于区间 tokens 总量"
    m = client.get("/api/stats/hourly?days=7&group=day&split=model", headers=hdr).json()
    assert all("by_model" in b for b in m["buckets"])
    assert sum(v for b in m["buckets"] for v in b["by_model"].values()) == m["total"]["tokens"]
    n = client.get("/api/stats/hourly?days=1&group=hour&split=bogus", headers=hdr).json()
    assert "by_platform" not in n["buckets"][0]


# ---------------------------------------------------------------- R56 Google models/ 前缀
def test_r56_google_models_prefix_strip(client, hdr):
    """Google AI Studio 的 OpenAI 兼容 /models 返回带 models/ 前缀的 id，但 chat 只收无前缀名字：
    线边界自动剥离（带前缀上游直接 400）；「重新拉取」的列表也存无前缀。"""
    import asyncio

    from tests import fake_upstream

    fake = [p for p in client.get("/api/providers", headers=hdr).json()["data"]
            if p["platform"] == "fake"][0]
    r = client.post("/api/providers", headers=hdr, json={
        "platform": "gwire", "name": "Google Wire", "type": "openai",
        "base_url": fake["base_url"], "api_keys": ["sk-fake"],
        "models": ["models/fake-gpt"], "enabled": True, "timeout_s": fake["timeout_s"]})
    assert r.status_code == 200, r.text

    r2 = client.post("/v1/chat/completions", headers=hdr, json={
        "model": "gwire/models/fake-gpt",
        "messages": [{"role": "user", "content": "hi"}]})
    assert r2.status_code == 200, r2.text
    assert fake_upstream.RECEIVED, "假上游应收到请求"
    assert fake_upstream.RECEIVED[-1]["body"]["model"] == "fake-gpt", \
        "发给上游的模型名不能带 models/ 前缀"

    # 拉取模型列表同样去前缀（下拉框/配置里不能存带前缀的名字）
    from app.registry import Upstream, fetch_models

    class _Resp:
        def raise_for_status(self): pass
        def json(self): return {"data": [{"id": "models/gemini-x"}, {"id": "gemini-y"}]}

    class _Cli:
        async def get(self, *a, **k): return _Resp()

    up = Upstream(platform="gtest", base_url="http://127.0.0.1:9", api_key="k")
    assert asyncio.run(fetch_models(up, _Cli())) == ["gemini-x", "gemini-y"]


# ---------------------------------------------------------------- R57 平台级出口代理
def test_r57_upstream_proxy(client, hdr):
    """平台配 proxy 后：转发/拉模型走该代理的独立 client；代理死端口时上游收不到请求；
    非法 scheme 拒绝；socks5h 允许（依赖 httpx[socks] extra）；GET 只读不写回。"""
    import socksio  # noqa: F401  # 硬依赖：缺了 socks5h 代理就不可用，测试直接红
    from tests import fake_upstream
    from app.main import tenants, store

    fake = [p for p in client.get("/api/providers", headers=hdr).json()["data"]
            if p["platform"] == "fake"][0]

    # 1) 保存 + 回读
    r = client.post("/api/providers", headers=hdr, json={
        "platform": "gproxy", "name": "Proxy Wire", "type": "openai",
        "base_url": fake["base_url"], "api_keys": ["sk-fake"],
        "proxy": "http://127.0.0.1:1", "models": ["fake-gpt"],
        "enabled": True, "timeout_s": fake["timeout_s"]})
    assert r.status_code == 200, r.text
    assert r.json()["data"]["proxy"] == "http://127.0.0.1:1"

    # 2) 只读端点不写回租户配置文件
    tpath = tenants.path(store.first_admin()["id"])
    before = tpath.read_text(encoding="utf-8")
    client.get("/api/providers", headers=hdr)
    assert tpath.read_text(encoding="utf-8") == before

    # 3) 死代理：请求必须走代理路径（假上游收不到），错误 502 且带上平台名
    r2 = client.post("/v1/chat/completions", headers=hdr, json={
        "model": "gproxy/fake-gpt", "messages": [{"role": "user", "content": "hi"}]})
    assert r2.status_code == 502, r2.text
    assert "gproxy" in r2.text and "连接上游失败" in r2.text
    assert not fake_upstream.RECEIVED

    # 4) 非法 scheme 拒绝
    r3 = client.post("/api/providers", headers=hdr, json={
        "platform": "gproxy", "base_url": fake["base_url"], "proxy": "ftp://1.2.3.4"})
    assert r3.status_code == 400, r3.text

    # 5) socks5h 允许；清空代理
    r4 = client.post("/api/providers", headers=hdr, json={
        "platform": "gproxy", "base_url": fake["base_url"],
        "proxy": "socks5h://127.0.0.1:1"})
    assert r4.status_code == 200, r4.text
    assert r4.json()["data"]["proxy"] == "socks5h://127.0.0.1:1"
    r5 = client.post("/api/providers", headers=hdr, json={
        "platform": "gproxy", "base_url": fake["base_url"], "proxy": ""})
    assert r5.status_code == 200, r5.text
    assert r5.json()["data"]["proxy"] == ""


# ---------------------------------------------------------------- R58 reasoning_effort=max 线边界降级
def test_r58_reasoning_effort_max_compat(client, hdr):
    """Google 等 OpenAI 兼容端点不接受 reasoning_effort=max（Google 400 INVALID_ARGUMENT，
    只收 high/low/medium/minimal/none），线边界统一降级 high；OpenAI 官方 base 保持不动；
    extra.reasoning_effort_map 可覆盖。"""
    from tests import fake_upstream
    from app.gateway import normalize_reasoning_effort
    from app.registry import Upstream

    fake = [p for p in client.get("/api/providers", headers=hdr).json()["data"]
            if p["platform"] == "fake"][0]
    r = client.post("/v1/chat/completions", headers=hdr, json={
        "model": "fake/fake-gpt", "messages": [{"role": "user", "content": "hi"}],
        "reasoning_effort": "max"})
    assert r.status_code == 200, r.text
    assert fake_upstream.RECEIVED[-1]["body"].get("reasoning_effort") == "high", \
        "第三方 OpenAI 兼容上游应收到降级后的 high"

    # OpenAI 官方 base 保持 max
    up = Upstream(platform="oai", base_url="https://api.openai.com/v1",
                  api_key="k", models=["gpt-x"])
    p = {"reasoning_effort": "max"}
    normalize_reasoning_effort(up, p)
    assert p["reasoning_effort"] == "max"

    # extra.reasoning_effort_map 覆盖
    r2 = client.post("/api/providers", headers=hdr, json={
        "platform": "fakemax", "name": "Max Map", "type": "openai",
        "base_url": fake["base_url"], "api_keys": ["sk-fake"],
        "models": ["fake-gpt"], "extra": {"reasoning_effort_map": {"max": "medium"}},
        "enabled": True})
    assert r2.status_code == 200, r2.text
    r3 = client.post("/v1/chat/completions", headers=hdr, json={
        "model": "fakemax/fake-gpt", "messages": [{"role": "user", "content": "hi"}],
        "reasoning_effort": "max"})
    assert r3.status_code == 200, r3.text
    assert fake_upstream.RECEIVED[-1]["body"].get("reasoning_effort") == "medium"


# ---------------------------------------------------------------- R59 日志时间边界（到秒）与相对快捷区间
def test_r59_log_time_boundaries(client, hdr):
    """日志页时间筛选支持到秒的日期时间（YYYY-MM-DDTHH:MM:SS）与服务端相对快捷区间
    range=15m/30m/1h/6h/today（日志页默认近 1 小时）；纯日期自动退回天粒度。
    用 client_ip=r59 标记记录，隔离其他测试的数据。"""
    from datetime import datetime, timedelta
    from urllib.parse import quote

    store = client.app.state.store
    admin = next(u for u in store.list_users() if u.get("role") == "admin")
    uid = admin["id"]
    now = datetime.now()
    old_ts = (now - timedelta(hours=3)).astimezone().isoformat(timespec="seconds")
    new_ts = now.astimezone().isoformat(timespec="seconds")
    store.record(platform="fake", model="fake-gpt", status="ok", user_id=uid,
                 ts=old_ts, client_ip="r59", prompt_tokens=1, completion_tokens=1)
    store.record(platform="fake", model="fake-gpt", status="ok", user_id=uid,
                 ts=new_ts, client_ip="r59", prompt_tokens=1, completion_tokens=1)

    def total(qs=""):
        r = client.get("/api/stats/records?ip=r59" + qs, headers=hdr)
        assert r.status_code == 200, r.text
        return r.json()["total"]

    assert total() == 2, "无时间条件时两条 r59 记录都应命中"

    new_dt = datetime.fromisoformat(new_ts)
    old_dt = datetime.fromisoformat(old_ts)
    nm = lambda dt: dt.astimezone().replace(tzinfo=None).isoformat(timespec="seconds")   # 无时区本地墙钟，与前端 datetime-local 同格式

    # 到秒的 from/to 边界
    assert total(f"&to={quote(nm(new_dt - timedelta(seconds=1)))}") == 1, "to=新记录前1秒 → 只含旧记录"
    assert total(f"&to={quote(nm(new_dt))}") == 2, "to=新记录所在秒（闭区间）→ 两条都在"
    assert total(f"&from={quote(nm(old_dt + timedelta(seconds=1)))}") == 1, "from=旧记录后1秒 → 只含新记录"
    assert total(f"&from={quote(nm(new_dt + timedelta(seconds=1)))}") == 0, "from=新记录后1秒 → 全排除"

    # 纯日期退回天粒度（起边界 00:00:00 / 终边界 23:59:59）
    yest = (now - timedelta(days=1)).strftime("%Y-%m-%d")
    today = now.strftime("%Y-%m-%d")
    assert total(f"&from={yest}") == 2
    assert total(f"&to={today}") == 2
    assert total(f"&to={yest}") == (1 if now.hour < 3 else 0)

    # 相对快捷区间（锚定服务端当前时间）
    assert total("&range=15m") == 1
    assert total("&range=30m") == 1
    assert total("&range=1h") == 1      # 日志页默认值
    assert total("&range=6h") == 2
    assert total("&range=today") == (2 if now.hour >= 3 else 1)
    assert total("&range=99x") == 2, "未知 range 不施加时间边界"


# ---------------------------------------------------------------- R60 按月活跃统计（选年月）
def test_r60_stats_month(client, hdr):
    """/api/stats/month 按指定年月统计（概览页「本月活跃情况」日历 + 年月下拉框）：
    返回该月 1 号~月末逐日桶（无数据补 0）、当月合计、active_days，以及该用户
    有数据的所有月份 months。按 ts 年月前缀（substr(ts,1,7)）精确取数，
    不受「近 N 天」窗口影响。用独立非管理员账号隔离数据、可断言精确值。"""
    from datetime import datetime, timedelta
    from calendar import monthrange
    import app.auth as auth_mod

    store = client.app.state.store
    u = store.create_user("r60@test.local", auth_mod.hash_password("r60"), role="user")
    key = auth_mod.new_api_key()
    store.create_api_key(u["id"], key, label="r60")
    h2 = {"authorization": f"Bearer {key}"}

    now = datetime.now()
    cy, cm, cd = now.year, now.month, now.day
    prev = now.replace(day=1) - timedelta(days=1)
    py, pm = prev.year, prev.month
    prev_last = monthrange(py, pm)[1]
    pday = min(cd, prev_last)
    store.record(platform="fake", model="fake-gpt", status="ok", user_id=u["id"],
                 ts=f"{cy:04d}-{cm:02d}-{cd:02d}T08:00:00{now.astimezone().tzinfo}",
                 client_ip="r60", prompt_tokens=1, completion_tokens=1)
    store.record(platform="fake", model="fake-gpt", status="ok", user_id=u["id"],
                 ts=f"{cy:04d}-{cm:02d}-{cd:02d}T09:00:00{now.astimezone().tzinfo}",
                 client_ip="r60", prompt_tokens=1, completion_tokens=1)
    store.record(platform="fake", model="fake-gpt", status="error", user_id=u["id"],
                 ts=f"{py:04d}-{pm:02d}-{pday:02d}T08:00:00{prev.astimezone().tzinfo}",
                 client_ip="r60", prompt_tokens=1, completion_tokens=1)

    def q(y, m):
        r = client.get(f"/api/stats/month?year={y}&month={m}", headers=h2)
        assert r.status_code == 200, r.text
        return r.json()

    d = q(cy, cm)
    assert d["ym"] == f"{cy:04d}-{cm:02d}"
    assert len(d["buckets"]) == monthrange(cy, cm)[1], "桶数 = 当月天数"
    assert d["total_requests"] == 2 and d["total_tokens"] == 4
    assert d["active_days"] == 1
    today = next(b for b in d["buckets"] if b["h"].endswith(f"{cd:02d}"))
    assert today["requests"] == 2
    # 空日期补 0
    first = d["buckets"][0]
    assert first["requests"] == (2 if cd == 1 else 0)
    # 下拉框候选：有数据的两个月份都在
    assert f"{cy:04d}-{cm:02d}" in d["months"] and f"{py:04d}-{pm:02d}" in d["months"]

    d2 = q(py, pm)
    assert d2["total_requests"] == 1 and d2["total_tokens"] == 2
    assert d2["active_days"] == 1 and d2["total_requests"] + d["total_requests"] == 3
    assert len(d2["buckets"]) == prev_last
    assert d2["buckets"][d2["buckets"].index(next(b for b in d2["buckets"] if b["requests"]))]["errors"] == 1

    # 没有数据的年月：全 0 桶
    d3 = q(cy + 1, 1)
    assert d3["total_requests"] == 0 and d3["active_days"] == 0
    assert len(d3["buckets"]) == 31

    # 非法月份 → 400
    r = client.get(f"/api/stats/month?year={cy}&month=13", headers=h2)
    assert r.status_code == 400


# ---------------------------------------------------------------- R61 按年活跃统计（星期 × 月份，选年份）
def test_r61_stats_year(client, hdr):
    """/api/stats/year 按「星期 × 月份」年度统计（概览页「年度活跃情况」日历 + 年份下拉框）：
    12 列（1~12 月）× 7 行（周一~周日），格子 = 该月该星期几的请求合计；
    按 ts 年份前缀精确取数。复用 R60 的独立账号思路隔离数据。"""
    from datetime import datetime, timedelta
    import calendar
    import app.auth as auth_mod

    store = client.app.state.store
    u = store.create_user("r61@test.local", auth_mod.hash_password("r61"), role="user")
    key = auth_mod.new_api_key()
    store.create_api_key(u["id"], key, label="r61")
    h2 = {"authorization": f"Bearer {key}"}

    now = datetime.now()
    cy, cm, cd = now.year, now.month, now.day
    py = cy - 1   # 显式取去年（11 月的「上月」还在今年，不能用 prev.year）
    dow_today = datetime(cy, cm, cd).weekday()
    # 今年：今天 2 条；本月另一天 1 条（与今天错开，且星期不同，避免同月同星期的格子撞车）；去年 1 条
    last = calendar.monthrange(cy, cm)[1]
    day2 = next((c for c in range(1, min(last, 28) + 1)
                 if c != cd and datetime(cy, cm, c).weekday() != dow_today), 1)
    tz = now.astimezone().tzinfo
    store.record(platform="fake", model="fake-gpt", status="ok", user_id=u["id"],
                 ts=f"{cy:04d}-{cm:02d}-{cd:02d}T08:00:00{tz}", client_ip="r61",
                 prompt_tokens=1, completion_tokens=1)
    store.record(platform="fake", model="fake-gpt", status="ok", user_id=u["id"],
                 ts=f"{cy:04d}-{cm:02d}-{cd:02d}T09:00:00{tz}", client_ip="r61",
                 prompt_tokens=1, completion_tokens=1)
    store.record(platform="fake", model="fake-gpt", status="error", user_id=u["id"],
                 ts=f"{cy:04d}-{cm:02d}-{day2:02d}T08:00:00{tz}", client_ip="r61",
                 prompt_tokens=1, completion_tokens=1)
    store.record(platform="fake", model="fake-gpt", status="ok", user_id=u["id"],
                 ts=f"{py:04d}-06-15T08:00:00{tz}", client_ip="r61",
                 prompt_tokens=1, completion_tokens=1)

    def q(y):
        r = client.get(f"/api/stats/year?year={y}", headers=h2)
        assert r.status_code == 200, r.text
        return r.json()

    d = q(cy)
    assert d["year"] == cy
    assert len(d["cells"]) == 12 and all(len(row) == 7 for row in d["cells"]), "12×7 完整网格"
    assert d["cells"][cm - 1][dow_today]["requests"] == 2, "今天所在 (月,星期) 格子 = 2"
    dow2 = datetime(cy, cm, day2).weekday()
    c2 = d["cells"][cm - 1][dow2]
    assert c2["requests"] == (3 if day2 == cd else 1) and c2["errors"] == (1 if day2 != cd else 0)
    assert d["total_requests"] == 3 and d["total_tokens"] == 6, "今年 3 条（去年的不计）"
    assert d["active_days"] == (1 if day2 == cd else 2), "今年有请求的自然日天数（去年的不计）"
    assert str(cy) in d["years"] and str(py) in d["years"], "下拉框候选含两个年份"

    d2 = q(py)
    assert d2["total_requests"] == 1 and d2["total_tokens"] == 2
    assert d2["cells"][5][datetime(py, 6, 15).weekday()]["requests"] == 1, "去年 6-15（周一）格子命中"

    d3 = q(cy + 1)
    assert d3["total_requests"] == 0 and d3["active_days"] == 0
    assert all(c["requests"] == 0 for row in d3["cells"] for c in row)

    r = client.get("/api/stats/year?year=1800", headers=h2)
    assert r.status_code == 400


# ---------------------------------------------------------------- R62 邮箱确认（magic link）
def test_r62_email_verify_flow(client, hdr):
    """自助注册（第二个账号起）pending → 未确认不能登录；点确认链接
    （HMAC 签名 + 30 分钟时间戳 + 邮箱一致）后 active 并发默认 Key；
    篡改/过期链接拒绝；重发限频；首个 admin 豁免（零配置引导）。"""
    import tempfile
    from pathlib import Path
    import app.auth as A
    from app.storage import Store

    store = client.app.state.store
    email, pw = "r62@test.local", "r62pass123"
    r = client.post("/api/auth/register", json={"email": email, "password": pw})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["pending"] is True and body["api_key"] == "", "未确认账号不建会话、不发 Key"
    u = store.get_user_by_email(email)
    assert u["status"] == "pending"
    assert store.list_api_keys(u["id"]) == [], "pending 不发 Key（数据面权限在确认后才有）"

    # 未确认不能登录
    r = client.post("/api/auth/login", json={"email": email, "password": pw})
    assert r.status_code == 403 and "email_unverified" in r.json()["detail"]

    # 点确认链接：签名 + 未过期 + 邮箱一致 → active + 默认 Key
    link = A.build_email_link(store, u, "http://verify.test")
    t = link.rsplit("t=", 1)[1]
    r = client.get(f"/api/auth/verify-email?t={t}")
    assert r.status_code == 200, r.text
    assert "邮箱确认成功" in r.text
    assert store.get_user_by_email(email)["status"] == "active"
    assert len(store.list_api_keys(u["id"])) == 1, "确认后发默认 Key"
    r = client.post("/api/auth/login", json={"email": email, "password": pw})
    assert r.status_code == 200, "确认后正常登录"
    # 重复点已确认链接：幂等
    r = client.get(f"/api/auth/verify-email?t={t}")
    assert r.status_code == 200 and "已确认" in r.text

    # 篡改签名 → 无效
    r2 = client.post("/api/auth/register", json={"email": "r62b@test.local", "password": pw})
    assert r2.json()["pending"] is True
    u2 = store.get_user_by_email("r62b@test.local")
    secret = A.get_email_secret(store)
    t2 = A.make_email_token(secret, u2)
    tampered = t2[:-1] + ("0" if t2[-1] != "0" else "1")
    r = client.get(f"/api/auth/verify-email?t={tampered}")
    assert r.status_code == 400 and "无效" in r.text
    assert store.get_user_by_email("r62b@test.local")["status"] == "pending"

    # 过期（时间戳窗口 30 分钟）
    expired = A.make_email_token(secret, u2, ttl=-60)
    r = client.get(f"/api/auth/verify-email?t={expired}")
    assert r.status_code == 400 and "失效" in r.text

    # 重发：首次成功 → 60s 冷却内拒发；未知邮箱不泄露存在性
    r = client.post("/api/auth/resend-verify", json={"email": "r62b@test.local"})
    assert r.status_code == 200 and "已发往" in r.json()["message"]
    r = client.post("/api/auth/resend-verify", json={"email": "r62b@test.local"})
    assert "频繁" in r.json()["message"], "60 秒冷却"
    r = client.post("/api/auth/resend-verify", json={"email": "nobody@test.local"})
    assert r.json()["message"].startswith("如果该账号存在"), "不泄露账号是否存在"

    # 首个 admin 豁免：全新 store 里第一个账号直接 active + 有 Key
    with tempfile.TemporaryDirectory() as td:
        s2 = Store(Path(td) / "x.db")
        u3 = A.register(s2, "first@adm.in", "firstpass123")
        assert u3["role"] == "admin" and u3["status"] == "active"
        assert s2.list_api_keys(u3["id"]), "首个 admin 有默认 Key（零配置引导）"
        u4 = A.register(s2, "second@adm.in", "secondpass123")
        assert u4["status"] == "pending", "第二个账号起必须邮箱确认"
        s2.close()


# ---------------------------------------------------------------- R63 邮箱发信设置页（SMTP）
def test_r63_mail_settings(client, hdr, monkeypatch):
    """SMTP 配置可在控制台保存（meta 表）；控制台设置 > 环境变量；
    密码打码不落明文、空密码保留旧值；非管理员 403；测试邮件端点。"""
    import app.auth as A
    import app.mailer as M

    store = client.app.state.store
    # 造一个普通账号（有 Key 才能调 /api）
    u = store.create_user("r63@test.local", A.hash_password("r63pass123"), role="user")
    key = A.new_api_key()
    store.create_api_key(u["id"], key, label="r63")
    h2 = {"authorization": f"Bearer {key}"}
    client.cookies.clear()   # 排除前面用例遗留会话 cookie 的干扰（cookie 优先于 Bearer）

    # 非管理员：读 / 写 / 测试都 403
    assert client.get("/api/settings/mail", headers=h2).status_code == 403
    assert client.post("/api/settings/mail", headers=h2, json={"mode": "log"}).status_code == 403
    assert client.post("/api/settings/mail/test", headers=h2, json={"to": "x@y.z"}).status_code == 403

    # 管理员读：密码打码、不含明文
    r = client.get("/api/settings/mail", headers=hdr)
    assert r.status_code == 200
    d = r.json()
    assert d.get("pass") in (None, ""), "响应里不能出现明文密码"
    assert "has_pass" in d and "pass_masked" in d and d["port"]

    # 保存 smtp 配置
    r = client.post("/api/settings/mail", headers=hdr, json={
        "mode": "smtp", "host": "smtp.test.local", "port": 465,
        "user": "relay@test.local", "password": "secret1234",
        "from": "", "starttls": True})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["host"] == "smtp.test.local" and d["port"] == 465
    assert d["has_pass"] is True and d["pass_masked"].endswith("1234")
    assert d.get("pass") in (None, ""), "保存响应也不能带明文密码"

    # 空密码 = 保留旧值
    r = client.post("/api/settings/mail", headers=hdr, json={
        "mode": "smtp", "host": "smtp.test.local", "port": 465,
        "user": "relay@test.local", "password": "", "starttls": True})
    assert r.json()["has_pass"] is True, "留空要沿用已保存的密码"

    # 参数校验
    assert client.post("/api/settings/mail", headers=hdr,
                       json={"mode": "xx"}).status_code == 400
    assert client.post("/api/settings/mail", headers=hdr,
                       json={"mode": "smtp", "host": "", "port": 1}).status_code == 400
    assert client.post("/api/settings/mail", headers=hdr,
                       json={"mode": "log", "port": "abc"}).status_code == 400

    # 优先级：控制台设置（meta）> 环境变量
    monkeypatch.setenv("RELAY_SMTP_HOST", "envhost.local")
    assert M.mail_settings(store)["host"] == "smtp.test.local", "控制台设置优先"
    store.set_meta("mail_smtp_host", "")
    assert M.mail_settings(store)["host"] == "envhost.local", "没存控制台配置时回退环境变量"

    # 测试邮件：smtp 指向死端口 → HTTP 200 但 ok=False（不抛错）
    store.set_meta("mail_smtp_host", "127.0.0.1")
    store.set_meta("mail_smtp_port", "1")
    store.set_meta("mail_smtp_starttls", "0")
    r = client.post("/api/settings/mail/test", headers=hdr, json={"to": "x@y.com"})
    assert r.status_code == 200 and r.json()["ok"] is False
    assert r.json()["mode"] == "smtp" and r.json()["error"]

    # 日志模式：ok=True，message 说明没实际发信
    store.set_meta("mail_mode", "log")
    r = client.post("/api/settings/mail/test", headers=hdr, json={"to": "x@y.com"})
    assert r.json()["ok"] is True and r.json()["mode"] == "log"
    assert "日志模式" in r.json()["message"]

    # 收件人校验
    assert client.post("/api/settings/mail/test", headers=hdr,
                       json={"to": "not-an-email"}).status_code == 400

    # 收尾：清掉本测试写入的 meta，后续用例回默认（log 模式）
    for k in ("mail_smtp_host", "mail_smtp_port", "mail_smtp_user",
              "mail_smtp_pass", "mail_smtp_from", "mail_smtp_starttls", "mail_mode"):
        store.set_meta(k, "")


def test_r64_stale_purge_and_user_delete(client, hdr):
    """pending 账号 30 分钟未确认邮箱 → 自动清理、邮箱可重新注册；
    管理员可彻底删除账号（会话/Key/用户行），自己与最后管理员不可删。"""
    from datetime import datetime, timedelta
    import app.auth as A
    store = client.app.state.store

    # a) 30 分钟内的 pending：同邮箱重新注册被拒
    u = store.create_user("r64a@test.local", A.hash_password("r64pass123"),
                          role="user", status="pending")
    r = client.post("/api/auth/register", json={"email": "r64a@test.local", "password": "r64pass456"})
    assert r.status_code == 400 and "已经注册过" in r.json()["detail"]

    # b) created_at 造到 31 分钟前：purge 清掉（用户行 + 其 Key），邮箱释放
    old_iso = (datetime.now().astimezone() - timedelta(minutes=31)).isoformat(timespec="seconds")
    store.conn_exec("UPDATE users SET created_at = ? WHERE id = ?", (old_iso, u["id"]))
    assert store.purge_stale_pending() >= 1
    assert store.get_user(u["id"]) is None

    # c) 同邮箱可以重新注册（走 register 端点内置的清理逻辑）
    r = client.post("/api/auth/register", json={"email": "r64a@test.local", "password": "r64pass456"})
    assert r.status_code == 200 and r.json()["user"]["status"] == "pending", r.text

    # d) 管理员彻底删除账号：用户行 / 会话 / API Key 都没了，历史保留
    u2 = store.create_user("r64b@test.local", A.hash_password("r64bpass12"), role="user")
    key = A.new_api_key()
    store.create_api_key(u2["id"], key, label="r64")
    tok = A.new_session_token()
    store.create_session(u2["id"], tok, A.session_expiry(), "test")
    r = client.delete(f"/api/users/{u2['id']}", headers=hdr)
    assert r.status_code == 200 and r.json()["ok"]
    assert store.get_user(u2["id"]) is None
    assert store.list_api_keys(u2["id"]) == []
    assert store.get_session(tok) is None

    # e) 不能删自己 / 删不存在的账号 404
    me = store.get_user_by_email("admin@test.local")
    assert client.delete(f"/api/users/{me['id']}", headers=hdr).status_code == 400
    assert client.delete("/api/users/99999", headers=hdr).status_code == 404

    # f) 唯一的最后一个活跃管理员不能被删（造第二个管理员再试删 → 允许，
    #    说明守卫只在「删完就没人管」时拦截）
    u3 = store.create_user("r64c@test.local", A.hash_password("r64cpass12"), role="admin")
    r = client.delete(f"/api/users/{u3['id']}", headers=hdr)
    assert r.status_code == 200   # 还有 admin（自己）在，删普通管理员放行


def test_r65_clickable_verification_mail(client, monkeypatch):
    """确认邮件走 multipart/alternative：HTML 版含可点按钮链接，纯文本版兜底。"""
    import smtplib
    import app.mailer as M
    store = client.app.state.store
    store.set_meta("mail_mode", "smtp")
    store.set_meta("mail_smtp_host", "127.0.0.1")
    store.set_meta("mail_smtp_port", "25")

    captured = {}

    class FakeSMTP:
        def __init__(self, host, port, timeout=15):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def starttls(self):
            pass

        def login(self, u, p):
            pass

        def send_message(self, msg):
            captured["msg"] = msg

    monkeypatch.setattr(smtplib, "SMTP", FakeSMTP)
    link = "http://t.test/api/auth/verify-email?t=abc.def"
    r = M.send_verification_email(store, "x@y.com", link)
    assert r["ok"], r
    parts = {p.get_content_type(): p.get_content() for p in captured["msg"].iter_parts()}
    assert "text/plain" in parts and "text/html" in parts, list(parts)
    html = parts["text/html"]
    assert f'href="{link}"' in html, "HTML 版必须带可点链接"
    assert "点击确认我的邮箱" in html
    assert link in parts["text/plain"], "纯文本兜底保留完整 URL"

    for k in ("mail_mode", "mail_smtp_host", "mail_smtp_port"):
        store.set_meta(k, "")


# ---------------------------------------------------------------- R66 地域访问地图（GeoIP）
def test_r66_region_map_geoip(client, hdr, monkeypatch):
    """概览页 /api/stats/regions：按客户端 IP 聚合地域。
    - GeoIP 库未配置：db=missing、data 空（优雅降级，不影响其他页面）
    - 有解析时：国内到省 / 境外到国家 / 私有网段 local，count 降序、total 不含空 IP
    记录用 client_ip=r66* 前缀标记隔离；境外/省解析 monkeypatch，不依赖真 mmdb。"""
    import app.geoip as G
    from pathlib import Path

    store = client.app.state.store
    admin = next(u for u in store.list_users() if u.get("role") == "admin")
    uid = admin["id"]
    for ip, n in (("r66-gd", 3), ("r66-bj", 2), ("r66-lan", 5), ("r66-us", 1)):
        for _ in range(n):
            store.record(platform="fake", model="fake-gpt", status="ok", user_id=uid,
                         client_ip=ip, prompt_tokens=1, completion_tokens=1)
    store.record(platform="fake", model="fake-gpt", status="ok", user_id=uid,
                 client_ip="", prompt_tokens=1, completion_tokens=1)

    monkeypatch.setenv("RELAY_GEOIP_DB", str(Path("D:/nonexistent/GeoLite2.mmdb")))
    G.reset()
    assert G.available() is False, "库文件不存在时应判定不可用"
    assert G.resolve("8.8.8.8") is None, "无库时 resolve 返回 None（调用方降级）"

    r = client.get("/api/stats/regions?range=today", headers=hdr)
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["db"] == "missing" and j["data"] == [] and j["total"] == 0

    def fake_resolve(ip):
        m = {"r66-gd": ("cn", "广东"), "r66-bj": ("cn", "北京"),
             "r66-us": ("world", "美国"), "r66-lan": ("local", "局域网/代理出口")}
        k, name = m.get(ip, ("unknown", "未知"))
        return {"kind": k, "region": name}

    monkeypatch.setattr(G, "available", lambda: True)
    monkeypatch.setattr(G, "resolve", fake_resolve)
    r2 = client.get("/api/stats/regions?range=today", headers=hdr)
    assert r2.status_code == 200, r2.text
    j2 = r2.json()
    assert j2["db"] == "ok"
    by = {(d["kind"], d["region"]): d["count"] for d in j2["data"]}
    assert by[("cn", "广东")] == 3
    assert by[("cn", "北京")] == 2
    assert by[("world", "美国")] == 1
    assert by[("local", "局域网/代理出口")] == 5
    assert j2["total"] >= 11, "至少包含本测试造的全部 r66* 记录（其他测试的当日记录会计入 total）"
    counts = [d["count"] for d in j2["data"]]
    assert counts == sorted(counts, reverse=True), "按 count 降序"
    ips = {r["ip"] for r in store.ip_counts(uid, 1)}
    assert "" not in ips, "空 client_ip 不进聚合"

    # 未知 range 回退 today；30d 也走日历天窗口
    r3 = client.get("/api/stats/regions?range=30d", headers=hdr)
    assert r3.status_code == 200 and r3.json()["total"] >= 11


def test_r66b_geoip_cn_province_mapping(client, monkeypatch):
    """geoip.resolve 的中国分支（生产实测形态，GeoLite2-City）：
    - subdivision.iso_code 是 'CN-SC' 这类字母码（不是 FIPS 数字码）→ 按英文名 Sichuan 映射到「四川」
    - 中国 IP 查不到 subdivision（移动/运营商/IPv6 段常见）→ 兜底「中国大陆」，不当未知
    - 境外按国家（COUNTRY_ZH 中文 + 英文名回退）；私有段 local。"""
    import app.geoip as G

    class FakeReader:
        def get(self, ip):
            table = {
                "117.139.165.171": {"country": {"iso_code": "CN", "names": {"en": "China"}},
                                     "subdivisions": [{"iso_code": "CN-SC", "names": {"en": "Sichuan"}}]},
                "118.112.73.205": {"country": {"iso_code": "CN", "names": {"en": "China"}},
                                    "subdivisions": []},
                "8.8.8.8": {"country": {"iso_code": "US", "names": {"en": "United States"}},
                             "subdivisions": []},
                "45.6.7.8": {"country": {"iso_code": "FR", "names": {"en": "France"}},
                              "subdivisions": []},
                "9.9.9.9": {"country": {"iso_code": "XX", "names": {"en": "Nowhere"}},
                             "subdivisions": []},
            }
            return table.get(ip)

    monkeypatch.setattr(G, "_load", lambda: FakeReader())
    assert G.resolve("117.139.165.171") == {"kind": "cn", "region": "四川"}, "字母 iso 码靠英文名映射"
    assert G.resolve("118.112.73.205") == {"kind": "cn", "region": "中国大陆"}, "无省数据走兜底桶"
    assert G.resolve("8.8.8.8") == {"kind": "world", "region": "美国"}
    assert G.resolve("45.6.7.8") == {"kind": "world", "region": "法国"}
    assert G.resolve("9.9.9.9") == {"kind": "world", "region": "Nowhere"}, "无中文表国家回退英文名"
    assert G.resolve("192.168.1.9") == {"kind": "local", "region": "局域网/代理出口"}
    assert G.resolve("8.8.8.8") is not None


def test_r67_region_cities_hotspots(client, hdr, monkeypatch):
    """/api/stats/regions 的城市热点（概览页「城市热点」圆点）：
    - 同一城市多个 IP 归并（按 kind+城市+坐标），count 求和；城市名转中文（CITY_ZH）
    - 境外城市也进 cities（kind=world，前端只把 cn 的画到中国地图上，境外列表仍按国家）
    - 无城市数据的 IP（运营商段常见）不产生热点条目
    - data/cities 之外的既有字段（total/db）不变。"""
    import app.geoip as G

    store = client.app.state.store
    admin = next(u for u in store.list_users() if u.get("role") == "admin")
    uid = admin["id"]
    for ip in ("117.139.165.171", "222.209.94.185", "8.8.8.8", "1.2.3.4"):
        store.record(platform="fake", model="fake-gpt", status="ok", user_id=uid,
                     client_ip=ip, prompt_tokens=1, completion_tokens=1)

    class FakeReader:
        def get(self, ip):
            table = {
                # 两个 IP 同一城市（成都）→ 应归并
                "117.139.165.171": {"country": {"iso_code": "CN", "names": {"en": "China"}},
                                     "subdivisions": [{"iso_code": "CN-SC", "names": {"en": "Sichuan"}}],
                                     "city": {"names": {"en": "Chengdu"}},
                                     "location": {"latitude": 30.6667, "longitude": 104.0667}},
                "222.209.94.185": {"country": {"iso_code": "CN", "names": {"en": "China"}},
                                    "subdivisions": [{"iso_code": "CN-SC", "names": {"en": "Sichuan"}}],
                                    "city": {"names": {"en": "Chengdu"}},
                                    "location": {"latitude": 30.6667, "longitude": 104.0667}},
                # 境外城市
                "8.8.8.8": {"country": {"iso_code": "US", "names": {"en": "United States"}},
                             "subdivisions": [],
                             "city": {"names": {"en": "Washington"}},
                             "location": {"latitude": 38.9, "longitude": -77.03}},
                # 无城市数据（运营商段常见形态）
                "1.2.3.4": {"country": {"iso_code": "CN", "names": {"en": "China"}},
                              "subdivisions": [], "city": {"names": {"en": ""}}},
            }
            return table.get(ip)

    monkeypatch.setattr(G, "_load", lambda: FakeReader())

    r = client.get("/api/stats/regions?range=today", headers=hdr)
    assert r.status_code == 200
    body = r.json()
    assert body["db"] == "ok" and body["total"] >= 4, "至少含本测试造的 4 条（其他测试当日记录会计入）"
    # 城市名转中文 + 同城归并（两个成都 IP 各 1 次）
    cd = [c for c in body["cities"] if c["name"] == "成都"]
    assert len(cd) == 1 and cd[0]["kind"] == "cn" and cd[0]["count"] == 2, "同城多 IP 归并且转中文"
    assert abs(cd[0]["lat"] - 30.6667) < 0.01 and abs(cd[0]["lon"] - 104.0667) < 0.01
    # 境外城市带出（前端只画 cn 的；列表仍按国家聚合在 data 里）
    wa = [c for c in body["cities"] if c["name"] == "Washington"]
    assert len(wa) == 1 and wa[0]["kind"] == "world"
    # 无城市数据的 1.2.3.4 不产生热点（cities 里应无第 3 个来源）
    assert len(body["cities"]) >= 2
    # data 聚合不受影响：四川含 2 个成都 IP，中国大陆兜底桶含 1.2.3.4
    by = {(d["kind"], d["region"]): d["count"] for d in body["data"]}
    assert by[("cn", "四川")] >= 2 and by[("cn", "中国大陆")] >= 1 and by[("world", "美国")] >= 1


# ---------------------------------------------------------------- R68 下游 key 备注列 + 过期时间
def test_r68_downstream_key_label_and_expiry(client, hdr):
    """下游客户端 key 的「备注名上统计/日志」+「过期时间」：
    - /api/stats/records 带 dkey_label（key 备注名）
    - summary group_by=dkey / series group_by=dkey / hourly split=dkey（by_dkey）都按备注名归并
    - 新建 key 可带过期时间（默认永不过期）；过期 key 走 relay 被 401「已过期」，
      改回永不过期又能用；坏格式 400。"""
    store = client.app.state.store
    me = [u for u in store.list_users() if u.get("role") == "admin"][0]
    uid = me["id"]

    # 造一把带备注的 key（默认永不过期）
    r = client.post("/api/keys", headers=hdr, json={"label": "r68测试key", "expires_at": ""})
    assert r.status_code == 200, r.text
    k1 = r.json()
    assert k1["expires_at"] in (None, ""), "不填 = 永不过期"
    new_hdr = {"authorization": f"Bearer {k1['key']}"}

    # 用新 key 走 relay：200，且请求记账带 dkey_id
    rr = client.post("/v1/chat/completions", headers=new_hdr,
                     json={"model": "fake/fake-gpt", "messages": [{"role": "user", "content": "hi"}]})
    assert rr.status_code == 200, rr.text
    rows = store.recent(5)
    drow = [x for x in rows if x.get("dkey_id") == k1["id"]]
    assert drow, "relay 请求应记账 dkey_id"
    assert drow[0]["dkey_label"] == "r68测试key", "日志页下游 key 列显示备注名"

    # 按下游 key 统计：summary / series / hourly 三条线都按备注名归并
    s = client.get("/api/stats/summary?days=1&group_by=dkey", headers=hdr).json()
    assert any(d["k"] == "r68测试key" for d in s["data"]), "summary 按备注名聚合"
    se = client.get("/api/stats/series?days=1&group_by=dkey", headers=hdr).json()
    assert "r68测试key" in se["keys"], "series 曲线出现下游 key 线"
    h = client.get("/api/stats/hourly?days=1&group=hour&split=dkey", headers=hdr).json()
    assert any(b.get("by_dkey", {}).get("r68测试key", 0) > 0 for b in h["buckets"]), "概览按下游 key 堆叠"

    # 设过期时间（过去）→ relay 401「已过期」
    p = client.patch(f"/api/keys/{k1['id']}", headers=hdr,
                    json={"expires_at": "2000-01-01 00:00"})
    assert p.status_code == 200 and p.json()["expires_at"], p.text
    rr2 = client.post("/v1/chat/completions", headers=new_hdr,
                      json={"model": "fake/fake-gpt", "messages": []})
    assert rr2.status_code == 401 and "已过期" in rr2.text, "过期 key 不能再访问上游"

    # 控制台会话（Cookie）不受过期影响：过期 key 账号的登录态仍可用
    # （hdr 用的是另一个 key，这里只验证 relay 拦截只针对过期那把）
    rr3 = client.post("/v1/chat/completions", headers=hdr,
                      json={"model": "fake/fake-gpt", "messages": []})
    assert rr3.status_code == 200, "未过期的 key 不受影响"

    # 改回永不过期 → 又能用
    p2 = client.patch(f"/api/keys/{k1['id']}", headers=hdr, json={"expires_at": ""})
    assert p2.status_code == 200 and not p2.json()["expires_at"]
    rr4 = client.post("/v1/chat/completions", headers=new_hdr,
                      json={"model": "fake/fake-gpt", "messages": []})
    assert rr4.status_code == 200, "取消过期后恢复访问"

    # 坏格式 400
    bad = client.post("/api/keys", headers=hdr, json={"label": "x", "expires_at": "不是日期"})
    assert bad.status_code == 400


# ---------------------------------------------------------------- R69 地域地图：点省下钻（cities 带 region）
def test_r69_regions_cities_region_field(client, hdr, monkeypatch):
    """/api/stats/regions 的 cities 每行带 region（所属省）：
    前端「点省份 → 该省城市排名」靠这个字段过滤；未解析到省的城市 region 为空串。"""
    import app.geoip as G

    store = client.app.state.store
    admin = next(u for u in store.list_users() if u.get("role") == "admin")
    uid = admin["id"]
    for ip in ("117.139.165.171", "8.8.8.8"):
        store.record(platform="fake", model="fake-gpt", status="ok", user_id=uid,
                     client_ip=ip, prompt_tokens=1, completion_tokens=1)

    class FakeReader:
        def get(self, ip):
            if ip == "117.139.165.171":
                return {"country": {"iso_code": "CN", "names": {"en": "China"}},
                        "subdivisions": [{"iso_code": "CN-SC", "names": {"en": "Sichuan"}}],
                        "city": {"names": {"en": "Chengdu"}},
                        "location": {"latitude": 30.6667, "longitude": 104.0667}}
            return {"country": {"iso_code": "US", "names": {"en": "United States"}},
                    "subdivisions": [],
                    "city": {"names": {"en": "Washington"}},
                    "location": {"latitude": 38.9, "longitude": -77.03}}

    monkeypatch.setattr(G, "_load", lambda: FakeReader())
    body = client.get("/api/stats/regions?range=today", headers=hdr).json()
    assert body["db"] == "ok"
    for c in body["cities"]:
        assert "region" in c, "cities 每行必须带 region 字段（前端点省下钻靠它）"
    cd = [c for c in body["cities"] if c["name"] == "成都"]
    assert cd and cd[0]["region"] == "四川", "成都城市行带所属省"
    wa = [c for c in body["cities"] if c["name"] == "Washington"]
    assert wa and wa[0]["region"] in ("美国", "United States"), "境外城市行带 region（国家）"





# ---------------------------------------------------------------- R70 IP2Region xdb
def test_r70_ip2region_xdb(monkeypatch):
    """真实 xdb 文件命中中国城市段；私有地址仍归 local；境外落 mmdb。"""
    from pathlib import Path
    xdb = Path(__file__).resolve().parent.parent / "data" / "IP2Region.xdb"
    if not xdb.exists():
        pytest.skip("IP2Region.xdb 未下载（data/IP2Region.xdb）")
    monkeypatch.setenv("RELAY_IP2REGION_XDB", str(xdb))
    monkeypatch.setenv("RELAY_GEOIP_DB", str(Path(__file__).resolve().parent.parent / "data" / "no-mmdb.mmdb"))
    from app import ip2xdb, geoip as G
    ip2xdb.reset()
    G.reset()
    q = ip2xdb.query("118.112.73.205")
    assert q.startswith("中国|"), f"成都电信 IP 应命中中国段: {q!r}"
    assert "成都" in q
    r = G.resolve("118.112.73.205")
    assert r["kind"] == "cn" and r["region"] == "四川" and r["city"] == "成都", r
    assert r.get("lat") is not None and r.get("lon") is not None
    assert G.resolve("10.1.2.3")["kind"] == "local", "私有地址不依赖地理库"
    assert G.resolve("8.8.8.8") is None, "xdb 只采信中国段，境外落 mmdb（此处双库缺 → None）"
    assert G.available() is True, "只配 xdb 也算可用"


def test_r70b_prov_city_norm():
    from app import geoip as G
    assert G._prov_norm("四川省") == "四川"
    assert G._prov_norm("广西壮族自治区") == "广西"
    assert G._prov_norm("新疆维吾尔自治区") == "新疆"
    assert G._prov_norm("北京市") == "北京"
    assert G._prov_norm("上海") == "上海"
    assert G._city_norm("成都市") == "成都"
    assert G._city_norm("0") == ""
    assert G._city_norm("") == ""
    assert G._city_coords("成都", "四川") == (30.6, 104.1)
    assert G._city_coords("没有这个城市", "四川") == (30.6, 104.1), "城市未收录落省会坐标"
    assert G._city_coords("没有这个城市", "宁夏") == (38.5, 106.2)
    assert G._city_coords("没有这个城市", "不知道") == (None, None)


def test_r71_huggingface_preset(client, hdr):
    body = client.get("/api/providers", headers=hdr).json()
    pf = [p for p in body["presets"] if p["platform"] == "huggingface"]
    assert pf, "HuggingFace 模板应在 presets 下拉里"
    assert pf[0]["type"] == "openai"
    assert pf[0]["base_url"].startswith("https://router.huggingface.co")


# ---------------------------------------------------------------- R75 Nous Research 上游模板
def test_r75_nousresearch_preset(client, hdr):
    """Nous Research 推理 API 模板（OpenAI 兼容，inference-api.nousresearch.com/v1）。"""
    body = client.get("/api/providers", headers=hdr).json()
    pf = [p for p in body["presets"] if p["platform"] == "nousresearch"]
    assert pf, "Nous Research 模板应在 presets 下拉里"
    assert pf[0]["type"] == "openai"
    assert pf[0]["base_url"].startswith("https://inference-api.nousresearch.com/v1")
    assert "Hermes-4.3-36B" in pf[0]["models"], "模型列表应预置 Hermes 4 系列"
    assert "nousresearch" in [p["platform"] for p in body["presets"]]


# ---------------------------------------------------------------- R76 NVIDIA NIM 上游模板
def test_r76_nvidianim_preset(client, hdr):
    """NVIDIA NIM（build.nvidia.com）模板：OpenAI 兼容，integrate.api.nvidia.com/v1。"""
    body = client.get("/api/providers", headers=hdr).json()
    pf = [p for p in body["presets"] if p["platform"] == "nvidianim"]
    assert pf, "NVIDIA NIM 模板应在 presets 下拉里"
    assert pf[0]["type"] == "openai"
    assert pf[0]["base_url"].startswith("https://integrate.api.nvidia.com/v1")
    assert "meta/llama-3.3-70b-instruct" in pf[0]["models"], "模型名应是 组织/模型名 形式"


# ---------------------------------------------------------------- R78 Google 平台关思考：extra_body.google 摊平到顶层
def test_r78_google_thinking_off_flattened(client, hdr, upstream_port):
    """model_reasoning=off 时，Google 的关思考写法（google.thinking_config.thinking_budget=0）
    必须摊平到请求体顶层发给上游；extra_body 只是 SDK 载体键，原样发出去上游会忽略（思考关不掉，
    带 tools 的 Gemini 3 会 400 thought_signature）。"""
    r = client.post("/api/providers", headers=hdr, json={
        "platform": "gemini", "name": "Gemini(fake)", "type": "openai",
        "base_url": f"http://127.0.0.1:{upstream_port}/v1", "api_keys": ["***"],
        "models": ["fake-gpt"], "extra": {"model_reasoning": {"fake-gpt": "off"}}})
    assert r.status_code == 200, r.text
    try:
        rr = client.post("/v1/chat/completions", headers=hdr,
                        json={"model": "gemini/fake-gpt", "messages": [{"role": "user", "content": "hi"}]})
        assert rr.status_code == 200, rr.text
        from tests import fake_upstream
        sent = [x for x in fake_upstream.RECEIVED if x["path"] == "/v1/chat/completions"][-1]["body"]
        # Google OpenAI 兼容端点不再收顶层 google 扩展字段（400 Unknown name "google"），
        # 关思考改走 OpenAI 系 reasoning_effort（gemini-3.1 实测 none 真关上）
        assert sent.get("reasoning_effort") == "none", f"Google 关思考应用 reasoning_effort=none: {sent.get('reasoning_effort')}"
        assert "google" not in sent and "extra_body" not in sent, "不能再发顶层 google / extra_body 字段"
    finally:
        assert client.delete("/api/providers/gemini", headers=hdr).status_code == 200


# ---------------------------------------------------------------- R81 聚合模型：一个名字挂多个 平台/模型，自动轮换 + 失败切换
def test_r81_aggregate_model_failover(client, hdr, upstream_port):
    """聚合名 MyAgg -> [坏平台, 好平台]：坏平台连接失败后冷却 60s 自动跳过，
    后续请求直接打到好平台；/v1/models 暴露聚合名；条目解析不到要 400。"""
    r1 = client.post("/api/providers", headers=hdr, json={
        "platform": "aggok", "name": "聚合-好(fake)", "type": "openai",
        "base_url": f"http://127.0.0.1:{upstream_port}/v1",
        "api_keys": ["***"], "models": ["m2"]})
    assert r1.status_code == 200, r1.text
    r2 = client.post("/api/providers", headers=hdr, json={
        "platform": "aggdead", "name": "聚合-坏(连不上)", "type": "openai",
        "base_url": "http://127.0.0.1:9/v1", "api_keys": ["***"], "models": ["m1"]})
    assert r2.status_code == 200, r2.text
    try:
        pa = client.post("/api/aggregates", headers=hdr,
                        json={"name": "MyAgg", "entries": ["aggdead/m1", "aggok/m2"]})
        assert pa.status_code == 200, pa.text

        # /v1/models 暴露聚合名
        ml = client.get("/v1/models", headers=hdr)
        assert ml.status_code == 200, ml.text
        ids = [d["id"] for d in ml.json()["data"]]
        assert "MyAgg" in ids, "聚合名应出现在 /v1/models"

        # 第一次调用：轮询到坏平台 → 失败冷却 → 切到好平台 → 200
        body = {"model": "MyAgg", "messages": [{"role": "user", "content": "hi"}]}
        rr1 = client.post("/v1/chat/completions", headers=hdr, json=body)
        assert rr1.status_code == 200, rr1.text
        # 第二次调用：坏平台在冷却中，直接好平台 → 依然 200
        rr2 = client.post("/v1/chat/completions", headers=hdr, json=body)
        assert rr2.status_code == 200, rr2.text

        # 聚合条目解析不到 → 400
        pb = client.post("/api/aggregates", headers=hdr,
                        json={"name": "Bad", "entries": ["nope/x"]})
        assert pb.status_code == 400, pb.text
    finally:
        client.delete("/api/aggregates/MyAgg", headers=hdr)
        client.delete("/api/providers/aggok", headers=hdr)
        client.delete("/api/providers/aggdead", headers=hdr)


# ---------------------------------------------------------------- R80 「关闭思考」解析预览：预览 = 实发（同一条代码路径）
def test_r80_reasoning_preview_endpoint(client, hdr, upstream_port):
    """GET /api/providers/reasoning-preview 走 gateway 同款解析：
    Gemini 3/4 → low+隐藏思考（摊平到顶层 google）+ 签名自动补齐；Gemini 2.x → budget 0；
    Anthropic 原生 → 不注入字段（省略即关）。"""
    r = client.post("/api/providers", headers=hdr, json={
        "platform": "gemv", "name": "GeminiV(fake)", "type": "openai",
        "base_url": f"http://127.0.0.1:{upstream_port}/v1", "api_keys": ["***"],
        "models": ["gemini-3.1-flash-lite", "gemini-2.5-flash"]})
    assert r.status_code == 200, r.text
    try:
        p1 = client.get("/api/providers/reasoning-preview", headers=hdr,
                       params={"platform": "gemv", "model": "gemini-3.1-flash-lite",
                               "out_fmt": "openai", "field": "auto"})
        assert p1.status_code == 200, p1.text
        j1 = p1.json()
        assert j1["wire"].get("reasoning_effort") == "none", j1
        assert j1["sig_autofill"] is True, "Gemini 3 应提示签名自动补齐"
        assert "google" not in j1["wire"], "Google 端点不收顶层 google 扩展字段"

        p2 = client.get("/api/providers/reasoning-preview", headers=hdr,
                       params={"platform": "gemv", "model": "gemini-2.5-flash",
                               "out_fmt": "openai", "field": "auto"})
        j2 = p2.json()
        assert j2["wire"].get("reasoning_effort") == "none", j2
        assert j2["sig_autofill"] is False

        p3 = client.get("/api/providers/reasoning-preview", headers=hdr,
                       params={"platform": "gemv", "model": "gemini-3.1-flash-lite",
                               "out_fmt": "anthropic", "field": "auto"})
        j3 = p3.json()
        assert j3["wire"] == {}, "Anthropic 原生省略即关"
    finally:
        assert client.delete("/api/providers/gemv", headers=hdr).status_code == 200


# ---------------------------------------------------------------- R79 Google 思考模型：thought_signature 自动补齐 + Gemini 3 最低档
def test_r79_google_thought_signature_bypass(client, hdr, upstream_port):
    """Gemini 3/4 思考模型工具调用：历史 assistant tool_calls 缺 thought_signature 时，
    中转发上游前自动补官方哨兵值（skip_thought_signature_validator），真实签名不覆盖；
    Gemini 3 模型 off 写法是 thinking_level=minimal（不吃 thinking_budget）。"""
    r = client.post("/api/providers", headers=hdr, json={
        "platform": "gemini2", "name": "Gemini2(fake)", "type": "openai",
        "base_url": f"http://127.0.0.1:{upstream_port}/v1", "api_keys": ["***"],
        "models": ["fake-gpt", "gemini-3.1-flash-lite"],
        "extra": {"model_reasoning": {"gemini-3.1-flash-lite": "off"}}})
    assert r.status_code == 200, r.text
    try:
        body = {
            "model": "gemini2/fake-gpt",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "tool_calls": [
                    {"id": "call_1", "type": "function",
                     "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}},
                    {"id": "call_2", "type": "function",
                     "function": {"name": "get_weather", "arguments": "{\"city\":\"London\"}"},
                     "extra_content": {"google": {"thought_signature": "REAL_SIG"}}}]},
                {"role": "tool", "tool_call_id": "call_1", "content": "15C"},
            ],
            "tools": [{"type": "function", "function": {"name": "get_weather",
                     "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}]
        }
        rr = client.post("/v1/chat/completions", headers=hdr, json=body)
        assert rr.status_code == 200, rr.text
        from tests import fake_upstream
        sent = [x for x in fake_upstream.RECEIVED if x["path"] == "/v1/chat/completions"][-1]["body"]
        amsgs = [m for m in sent["messages"] if m.get("role") == "assistant"]
        assert amsgs, "assistant 历史应透传到上游"
        tc1, tc2 = amsgs[0]["tool_calls"][0], amsgs[0]["tool_calls"][1]
        assert tc1.get("extra_content", {}).get("google", {}).get("thought_signature") == \
            "skip_thought_signature_validator", "缺签名的工具调用要补官方哨兵值"
        assert tc2["extra_content"]["google"]["thought_signature"] == "REAL_SIG", "真实签名不能覆盖"

        # Gemini 3 模型 off：thinking_level=minimal（摊平到顶层 google），不是 thinking_budget
        rr2 = client.post("/v1/chat/completions", headers=hdr,
                          json={"model": "gemini2/gemini-3.1-flash-lite",
                                "messages": [{"role": "user", "content": "hi"}]})
        assert rr2.status_code == 200, rr2.text
        sent2 = [x for x in fake_upstream.RECEIVED if x["path"] == "/v1/chat/completions"][-1]["body"]
        assert sent2.get("reasoning_effort") == "none", f"Gemini 3 关思考应用 reasoning_effort=none: {sent2.get('reasoning_effort')}"
        assert "google" not in sent2, "Google 端点不收顶层 google 扩展字段"
    finally:
        assert client.delete("/api/providers/gemini2", headers=hdr).status_code == 200


# ---------------------------------------------------------------- R77 常用上游平台模板补齐
def test_r77_common_provider_presets(client, hdr):
    """国内常用平台（火山/月之暗面/百炼/智谱/硅基/阶跃/星火/MiniMax）+ 海外（OpenAI/OpenRouter/xAI/Mistral）模板齐全。"""
    want = {
        "volcark": "https://ark.cn-beijing.volces.com/api/v3",
        "moonshot": "https://api.moonshot.cn/v1",
        "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "zhipu": "https://open.bigmodel.cn/api/paas/v4",
        "siliconflow": "https://api.siliconflow.cn/v1",
        "stepfun": "https://api.stepfun.com/v1",
        "xfyun": "https://spark-api-open.xf-yun.com/v1",
        "minimax": "https://api.minimaxi.com/v1",
        "openai": "https://api.openai.com/v1",
        "openrouter": "https://openrouter.ai/api/v1",
        "xai": "https://api.x.ai/v1",
        "mistral": "https://api.mistral.ai/v1",
    }
    body = client.get("/api/providers", headers=hdr).json()
    by_slug = {p["platform"]: p for p in body["presets"]}
    for slug, base in want.items():
        assert slug in by_slug, f"缺模板 {slug}"
        assert by_slug[slug]["type"] == "openai"
        assert by_slug[slug]["base_url"].startswith(base), f"{slug} base_url 不对: {by_slug[slug]['base_url']}"
        assert by_slug[slug]["models"], f"{slug} 应有预置模型"
    # 每个上游平台只保留一份模板：slug 与默认 Base URL 都不得重复
    slugs = [p["platform"] for p in body["presets"]]
    bases = [p["base_url"] for p in body["presets"]]
    assert len(slugs) == len(set(slugs)), f"模板 slug 重复: {slugs}"
    assert len(bases) == len(set(bases)), f"模板 Base URL 重复: {bases}"


# ---------------------------------------------------------------- R72 key 过期：纯日期 = 当天仍可用（按 23:59:59 失效）
def test_r72_key_expiry_date_end_of_day(client, hdr):
    """控制台日历控件精确到日：纯 YYYY-MM-DD 的过期时间按「当天仍可用」落库
    （当天 23:59:59 失效）；带时入参保持原义；坏格式仍 400。"""
    from datetime import datetime, timedelta

    store = client.app.state.store
    me = [u for u in store.list_users() if u.get("role") == "admin"][0]

    # 新建：纯日期（明天）→ expires_at = 明天 23:59:59，现在仍能用
    tomorrow = (datetime.now().astimezone() + timedelta(days=1)).strftime("%Y-%m-%d")
    r = client.post("/api/keys", headers=hdr, json={"label": "r72", "expires_at": tomorrow})
    assert r.status_code == 200, r.text
    exp = r.json()["expires_at"]
    assert exp and exp.startswith(tomorrow) and "23:59:59" in exp, f"纯日期应按当天结束落库: {exp}"
    ok = client.post(
        "/v1/chat/completions",
        headers={"authorization": f"Bearer {r.json()['key']}"},
        json={"model": "fake/fake-gpt", "messages": [{"role": "user", "content": "hi"}]})
    assert ok.status_code == 200, f"明天过期的 key 此刻应仍可用（当天仍可用语义）: {ok.text}"

    # 带时入参保持原义（不进 23:59:59 分支）
    r2 = client.post("/api/keys", headers=hdr, json={"label": "r72b", "expires_at": "2030-06-01 12:30"})
    assert r2.status_code == 200 and r2.json()["expires_at"].startswith("2030-06-01T12:30:00"), r2.text

    # PATCH 改纯日期同样按当天结束；清空恢复永不过期
    p = client.patch(f"/api/keys/{r2.json()['id']}", headers=hdr, json={"expires_at": tomorrow})
    assert p.status_code == 200 and "23:59:59" in p.json()["expires_at"], p.text
    p2 = client.patch(f"/api/keys/{r2.json()['id']}", headers=hdr, json={"expires_at": ""})
    assert p2.status_code == 200 and not p2.json()["expires_at"]

    # 坏格式仍 400
    bad = client.post("/api/keys", headers=hdr, json={"label": "x", "expires_at": "2026-99-99"})
    assert bad.status_code == 400


# ---------------------------------------------------------------- R72 key 过期：老库迁移 + 列表回显
def test_r72_key_expires_legacy_db_migration(monkeypatch):
    """模拟升级前的老库（api_keys 无 expires_at 列）：Store 打开自动迁移，
    新建 key 带过期、改过期、列表回显 expires_at，旧 key 保持 None=永不过期。"""
    import sqlite3
    from pathlib import Path
    import tempfile
    from app.storage import Store
    tmp = Path(tempfile.mkdtemp())
    db = tmp / "relay.db"
    c = sqlite3.connect(db)
    c.execute("CREATE TABLE api_keys (id INTEGER PRIMARY KEY, user_id INTEGER, key TEXT, "
              "label TEXT, created_at TEXT, last_used_at TEXT, revoked INTEGER DEFAULT 0)")
    c.execute("INSERT INTO api_keys (user_id, key, label, created_at) VALUES (1, 'sk-old', '旧key', '2026-11-01T00:00:00')")
    c.commit()
    c.close()
    s = Store(db)
    k = s.create_api_key(1, "sk-new", "workbuddy", "2026-11-08T23:59:59+08:00")
    assert k.get("expires_at") == "2026-11-08T23:59:59+08:00", "新建 key 要回传 expires_at"
    assert s.update_key_expires(1, k["id"], "2026-11-07T23:59:59+08:00")
    rows = {r["label"]: r for r in s.list_api_keys(1)}
    assert rows["workbuddy"]["expires_at"] == "2026-11-07T23:59:59+08:00"
    assert rows["旧key"].get("expires_at") in (None, ""), "老 key 无过期 = 永不过期"


# ---------------------------------------------------------------- R73 /api/settings 的 key 列表必须带 expires_at
def test_r73_settings_keys_expires_at(client, hdr):
    """设置页「我的 API Key」表数据来自 /api/settings（不是 /api/keys），
    漏字段会一直显示「永不过期」——锁死回归。"""
    store = client.app.state.store
    admin = next(u for u in store.list_users() if u.get("role") == "admin")
    uid = admin["id"]
    k = store.create_api_key(uid, "sk-r73", "r73", "2026-11-08T23:59:59+08:00")
    r = client.get("/api/settings", headers=hdr)
    assert r.status_code == 200, r.text
    ks = {x["label"]: x for x in r.json()["api_keys"]}
    assert ks["r73"].get("expires_at") == "2026-11-08T23:59:59+08:00",         f"/api/settings 的 api_keys 必须回传 expires_at，实际: {ks.get('r73')}"
    # 没设过期的 key 保持 None（前端显示永不过期）
    assert all("expires_at" in x for x in r.json()["api_keys"]), "每行都要有该字段"


# ---------------------------------------------------------------- R74 过期/禁用 key：401 文案 + 请求日志落账
def test_r74_expired_revoked_key_audit(client, hdr):
    """过期 key → 401 文案带实际过期时间，且 401 事件写进请求日志（带 dkey_id 归属）；
    禁用（吊销）key → 401 文案标明已禁用，同样落日志。missing/invalid 不落账。"""
    store = client.app.state.store
    me = [u for u in store.list_users() if u.get("role") == "admin"][0]
    uid = me["id"]

    # 过期 key
    r = client.post("/api/keys", headers=hdr, json={"label": "r74过期", "expires_at": "2000-01-01 00:00"})
    kid = r.json()["id"]; rkey = r.json()["key"]
    bad = {"authorization": f"Bearer {rkey}"}
    rr = client.post("/v1/chat/completions", headers=bad,
                    json={"model": "fake/fake-gpt", "messages": []})
    assert rr.status_code == 401
    assert "过期" in rr.text and "2000-01-01" in rr.text, f"401 文案要带实际过期时间: {rr.text}"
    rows = store.recent(5)
    hit = [x for x in rows if x.get("dkey_id") == kid and x.get("status") == "error"]
    assert hit, "过期 401 应写进请求日志（带 dkey_id 归属）"
    assert "过期" in (hit[0].get("error") or ""), f"日志 error 字段要含原因: {hit[0].get('error')}"

    # 禁用 key（吊销 = 立即禁用，不删除）
    r2 = client.post("/api/keys", headers=hdr, json={"label": "r74禁用"})
    kid2 = r2.json()["id"]; rkey2 = r2.json()["key"]
    assert store.revoke_api_key(uid, kid2)
    rr2 = client.post("/v1/chat/completions", headers={"authorization": f"Bearer {rkey2}"},
                     json={"model": "fake/fake-gpt", "messages": []})
    assert rr2.status_code == 401
    assert "禁用" in rr2.text, f"401 文案要标明禁用状态: {rr2.text}"
    rows2 = store.recent(5)
    hit2 = [x for x in rows2 if x.get("dkey_id") == kid2 and x.get("status") == "error"]
    assert hit2, "禁用 401 也应写进请求日志"

    # missing / invalid 不写请求日志
    n_before = len(store.recent(5))
    client.post("/v1/chat/completions", json={"model": "fake/fake-gpt", "messages": []})
    client.post("/v1/chat/completions", headers={"authorization": "Bearer sk-r74-no-such-key"},
                json={"model": "fake/fake-gpt", "messages": []})
    assert len(store.recent(5)) == n_before, "missing/invalid 401 不落请求日志（防攻击刷库）"


# ---------------------------------------------------------------- R82 Google 关闭思考值分代 + 400 自适应降级（对齐真实端点实测）
def test_r82_google_effort_generation_and_adapt(client, hdr, upstream_port):
    """真实 /v1beta/openai 实测（2026-11-09）：gemini-3.5+ 的 reasoning_effort 词汇表没有
    "none"（发了 400 "Request contains an invalid argument"），最低档是 minimal；
    2.x/3.0-3.4 收 none。所以：3.5 系自动解析直接给 minimal（一次通过）；表没覆盖的
    新代模型（如 3.9）先发 none 被 400 时，网关自适应降级 minimal 重发，客户端无感。"""
    r = client.post("/api/providers", headers=hdr, json={
        "platform": "goog2", "name": "Gem2(fake)", "type": "openai",
        "base_url": f"http://127.0.0.1:{upstream_port}/v1", "api_keys": ["***"],
        "models": ["gemini-3.5-flash-lite", "gemini-3.9-flash"],
        "extra": {"model_reasoning": {"gemini-3.5-flash-lite": "off",
                                      "gemini-3.9-flash": {"policy": "off", "field": "auto"}}}})
    assert r.status_code == 200, r.text
    try:
        from tests import fake_upstream
        fake_upstream.RECEIVED.clear()

        # 3.5：表里直接 minimal，假上游一次通过（不会有 400 重发）
        p35 = client.post("/v1/chat/completions", headers=hdr, json={
            "model": "goog2/gemini-3.5-flash-lite",
            "messages": [{"role": "user", "content": "hi"}]})
        assert p35.status_code == 200, p35.text
        sent35 = [x for x in fake_upstream.RECEIVED
                  if x["body"].get("model") == "gemini-3.5-flash-lite"]
        assert len(sent35) == 1, "3.5 应一次通过（直接发 minimal），不该有 400 重发"
        assert sent35[0]["body"].get("reasoning_effort") == "minimal", \
            f"3.5 系关思考应是 minimal: {sent35[0]['body'].get('reasoning_effort')}"

        # 3.9（表未覆盖）：auto 解析给 none → 假上游按真实 Google 行为 400
        # → 网关自适应降级 minimal 重发 → 200。两次发送记录都在。
        p39 = client.post("/v1/chat/completions", headers=hdr, json={
            "model": "goog2/gemini-3.9-flash",
            "messages": [{"role": "user", "content": "hi"}]})
        assert p39.status_code == 200, p39.text
        sent39 = [x["body"] for x in fake_upstream.RECEIVED
                  if x["body"].get("model") == "gemini-3.9-flash"]
        assert [b.get("reasoning_effort") for b in sent39] == ["none", "minimal"], \
            f"应先发 none 被 400、再降级 minimal: {[b.get('reasoning_effort') for b in sent39]}"
    finally:
        client.delete("/api/providers/goog2", headers=hdr)


# ---------------------------------------------------------------- R83 瞬时 5xx/429 同 key 快速重试（NVIDIA NIM 503 content-safety）
def test_r83_transient_same_key_retry(client, hdr, upstream_port):
    """单 key 平台遇到瞬时 503（NVIDIA NIM 的 content safety 偶发）时，
    网关同 key 退避重发，第二次 200，客户端无感；不 503 透传。"""
    r = client.post("/api/providers", headers=hdr, json={
        "platform": "nimx", "name": "NIMx(fake)", "type": "openai",
        "base_url": f"http://127.0.0.1:{upstream_port}/v1", "api_keys": ["nk"],
        "models": ["m-503once"]})
    assert r.status_code == 200, r.text
    try:
        from tests import fake_upstream
        fake_upstream._ONCE_503["hit"] = False
        fake_upstream.RECEIVED.clear()
        rr = client.post("/v1/chat/completions", headers=hdr, json={
            "model": "nimx/m-503once", "messages": [{"role": "user", "content": "hi"}]})
        assert rr.status_code == 200, rr.text
        sent = [x for x in fake_upstream.RECEIVED if x["body"].get("model") == "m-503once"]
        assert len(sent) == 2, f"首次 503 应同 key 重试、第二次 200（共 2 次）: {len(sent)}"
    finally:
        client.delete("/api/providers/nimx", headers=hdr)


# ---------------------------------------------------------------- R84 平台「测试」按钮按登记顺序逐个试模型（NVIDIA 按账号授权）
def test_r84_provider_test_tries_models(client, hdr, upstream_port):
    """NVIDIA NIM 这类按账号授权模型清单的平台：登记列表首模型未必对该 key 开放（404
    "Function not found for account"）。测试按钮不指定模型时应按登记顺序逐个 ping
    （最多 5 个），跳过 404 的，第一个通的才算平台可用；全部失败返回 ok=false + 末条原因。"""
    r1 = client.post("/api/providers", headers=hdr, json={
        "platform": "nimt", "name": "NIMt(fake)", "type": "openai",
        "base_url": f"http://127.0.0.1:{upstream_port}/v1", "api_keys": ["tk"],
        "models": ["m-x404", "m-ok-plain"]})
    assert r1.status_code == 200, r1.text
    r2 = client.post("/api/providers", headers=hdr, json={
        "platform": "nimbad", "name": "NIMbad(fake)", "type": "openai",
        "base_url": f"http://127.0.0.1:{upstream_port}/v1", "api_keys": ["tk"],
        "models": ["m-x404"]})
    assert r2.status_code == 200, r2.text
    try:
        rt = client.post("/api/providers/nimt/test", headers=hdr, json={})
        assert rt.status_code == 200, rt.text
        j = rt.json()
        assert j["ok"] is True and j["model"] == "m-ok-plain", \
            f"首模型 404 应跳到第二个: {j}"

        rt2 = client.post("/api/providers/nimbad/test", headers=hdr, json={})
        j2 = rt2.json()
        assert j2["ok"] is False and "m-x404" in j2["error"], \
            f"全模型不可用要 ok=false 并列出试过的: {j2}"
        assert "不可用" in j2["error"]

        # 显式指定 model 时只试它（不扩散）
        rt3 = client.post("/api/providers/nimt/test", headers=hdr,
                          json={"model": "m-x404"})
        assert rt3.json()["ok"] is False, "显式指定 404 模型应直接失败"
    finally:
        client.delete("/api/providers/nimt", headers=hdr)
        client.delete("/api/providers/nimbad", headers=hdr)


# ---------------------------------------------------------------- R85 平台「测试」= 全量模型探测：可用的保留、未授权自动移除写回
def test_r85_probe_provider_models(client, hdr, upstream_port):
    """NVIDIA NIM 式平台：测试按钮探测全部已登记模型——
    模型级 404/400（账号未授权）移除并写回模型清单；瞬时 503 经网关重试后保留；
    401/403 先验证 key 本身：key 能列 /models → 单模型未授权（移除并继续），
    key 连 /models 都 401 → 真 key 级错误（保留清单不写回，见 R89）。"""
    r1 = client.post("/api/providers", headers=hdr, json={
        "platform": "nimp", "name": "NIMp(fake)", "type": "openai",
        "base_url": f"http://127.0.0.1:{upstream_port}/v1", "api_keys": ["pk"],
        "models": ["m-x404", "m-ok-plain", "m-503once"]})
    assert r1.status_code == 200, r1.text
    r2 = client.post("/api/providers", headers=hdr, json={
        "platform": "nimk", "name": "NIMk(fake)", "type": "openai",
        "base_url": f"http://127.0.0.1:{upstream_port}/v1", "api_keys": ["pk"],
        "models": ["m-x401", "m-ok-plain"]})
    assert r2.status_code == 200, r2.text
    try:
        from tests import fake_upstream
        fake_upstream._ONCE_503["hit"] = False

        # 混合平台：404 移除 + 503 瞬时（重试后 200）保留 + 正常保留
        rp = client.post("/api/providers/nimp/probe", headers=hdr, json={})
        assert rp.status_code == 200, rp.text
        j = rp.json()
        assert j["key_error"] is False and j["written"] is True, j
        assert sorted(j["kept"]) == ["m-503once", "m-ok-plain"], j
        assert [x["model"] for x in j["removed"]] == ["m-x404"], j
        assert "m-503once" in j["transient"] or "m-503once" in j["kept"], \
            "瞬时故障（重试后成功）必须保留在清单里"
        gp = [p for p in client.get("/api/providers", headers=hdr).json()["data"]
              if p["platform"] == "nimp"]
        assert gp[0]["models"] == ["m-ok-plain", "m-503once"], \
            f"404 模型应被写回移除: {gp[0]['models']}"

        # 401 但 key 能列 /models（普通 fake 上游）→ 单模型未授权：移除该模型、继续探测、写回
        rp2 = client.post("/api/providers/nimk/probe", headers=hdr, json={})
        j2 = rp2.json()
        assert j2["key_error"] is False and j2["written"] is True, j2
        assert sorted(j2["kept"]) == ["m-ok-plain"] and j2.get("skipped") == 0, j2
        assert [x["model"] for x in j2["removed"]] == ["m-x401"], \
            "401 且 key 可用 → 判单模型未授权并移除"
        gp2 = [p for p in client.get("/api/providers", headers=hdr).json()["data"]
               if p["platform"] == "nimk"]
        assert gp2[0]["models"] == ["m-ok-plain"], \
            f"单模型未授权应写回移除: {gp2[0]['models']}"
    finally:
        client.delete("/api/providers/nimp", headers=hdr)
        client.delete("/api/providers/nimk", headers=hdr)


# ---------------------------------------------------------------- R89 探测 401/连不通 的细分判定：
# key 可用但单模型未授权 → 只删该模型继续探测；key 真坏（/models 也 401）→ 整轮 key_error
def test_r89_probe_401_and_connfail_granularity(client, hdr, upstream_port):
    """① 普通平台（/models 200）上 401 的模型：判「无此模型授权」移除，不影响其余模型；
    ② /x401 平台（chat 与 /models 都 401）：判定 key 级错误，保留清单不写回；
    ③ 连不通的平台：容忍 2 个 connfail 模型不中断，第 3 个连续才整轮放弃。"""
    r1 = client.post("/api/providers", headers=hdr, json={
        "platform": "nimsplit", "name": "NIMsplit(fake)", "type": "openai",
        "base_url": f"http://127.0.0.1:{upstream_port}/v1", "api_keys": ["pk"],
        "models": ["m-x401", "m-ok-plain", "m-x404"]})
    assert r1.status_code == 200, r1.text
    r2 = client.post("/api/providers", headers=hdr, json={
        "platform": "nimdeadkey", "name": "NIMdeadkey(fake)", "type": "openai",
        "base_url": f"http://127.0.0.1:{upstream_port}/x401/v1", "api_keys": ["pk"],
        "models": ["m-a", "m-b", "m-c"]})
    assert r2.status_code == 200, r2.text
    r3 = client.post("/api/providers", headers=hdr, json={
        "platform": "dead2", "name": "死主机2(fake)", "type": "openai",
        "base_url": "http://127.0.0.1:9/v1", "api_keys": ["pk"],
        "models": ["m-a", "m-b"]})
    assert r3.status_code == 200, r3.text
    try:
        # ① 401 单模型未授权（key 能列 /models）→ 移除该模型、继续测、写回
        rp = client.post("/api/providers/nimsplit/probe", headers=hdr, json={})
        assert rp.status_code == 200, rp.text
        j = rp.json()
        assert j["key_error"] is False and j["written"] is True, j
        assert sorted(x["model"] for x in j["removed"]) == ["m-x401", "m-x404"], j
        assert any("无此模型授权" in x["reason"] for x in j["removed"]
                   if x["model"] == "m-x401"), "401 且 key 可用应标注为单模型未授权"
        assert j["kept"] == ["m-ok-plain"], j
        assert j.get("skipped") == 0, "两个被移除模型之后还要继续测"
        gs = [p for p in client.get("/api/providers", headers=hdr).json()["data"]
              if p["platform"] == "nimsplit"]
        assert gs[0]["models"] == ["m-ok-plain"], f"写回后只剩可用模型: {gs[0]['models']}"

        # ② /x401：chat 401 且 /models 也 401 → 真 key 级错误 → 保留清单、提前收工
        rp2 = client.post("/api/providers/nimdeadkey/probe", headers=hdr, json={})
        j2 = rp2.json()
        assert j2["key_error"] is True and j2["written"] is False, j2
        assert j2.get("skipped") == 2, j2
        gd = [p for p in client.get("/api/providers", headers=hdr).json()["data"]
              if p["platform"] == "nimdeadkey"]
        assert gd[0]["models"] == ["m-a", "m-b", "m-c"], "key 真坏不得改动模型清单"

        # ③ 连不通但只有 2 个模型：不触发「连续 3 个」放弃，正常跑完、不写回
        rp3 = client.post("/api/providers/dead2/probe", headers=hdr, json={})
        j3 = rp3.json()
        assert j3["connect_fail"] is False and j3["written"] is False, j3
        assert j3.get("connfails") == ["m-a", "m-b"], j3
        assert j3["kept"] == ["m-a", "m-b"] and j3.get("skipped") == 0, j3
        g2 = [p for p in client.get("/api/providers", headers=hdr).json()["data"]
              if p["platform"] == "dead2"]
        assert g2[0]["models"] == ["m-a", "m-b"], "连不通不动模型清单"
    finally:
        client.delete("/api/providers/nimsplit", headers=hdr)
        client.delete("/api/providers/nimdeadkey", headers=hdr)
        client.delete("/api/providers/dead2", headers=hdr)


# ---------------------------------------------------------------- R86 探测进度流（stream 模式）：逐模型事件 + 末行 done/写回
def test_r86_probe_stream_progress(client, hdr, upstream_port):
    """probe body {"stream": true} → NDJSON 事件流：每个模型一行（i/total/model/result），
    done 行携带写回后的汇总。401 模型（key 能列 /models）按单模型未授权移除，探测继续。"""
    r1 = client.post("/api/providers", headers=hdr, json={
        "platform": "nimstream", "name": "NIMstream(fake)", "type": "openai",
        "base_url": f"http://127.0.0.1:{upstream_port}/v1", "api_keys": ["pk"],
        "models": ["m-x404", "m-x401", "m-ok-plain"]})
    assert r1.status_code == 200, r1.text
    try:
        rp = client.post("/api/providers/nimstream/probe", headers=hdr,
                         json={"stream": True})
        assert rp.status_code == 200, rp.text
        lines = [json.loads(x) for x in rp.text.splitlines() if x.strip()]
        starts = [e for e in lines if e.get("phase") == "start"]
        results = [e for e in lines if "result" in e and not e.get("done")]
        final = lines[-1]
        # 每个被测模型先给 start 事件（UI 立刻显示「正在测哪个/剩几个」，不再卡「开始…」）
        assert [e["model"] for e in starts] == ["m-x404", "m-x401", "m-ok-plain"], starts
        assert len(results) == 3, f"三个模型都应出结果: {results}"
        assert [e["model"] for e in results] == ["m-x404", "m-x401", "m-ok-plain"]
        assert results[0]["result"] == "removed" and results[0]["i"] == 1 \
            and results[0]["total"] == 3
        # 401 且 key 能列 /models → 单模型未授权（removed），探测不停
        assert results[1]["result"] == "removed" and results[1]["code"] == 401, \
            "401 应判单模型未授权并移除（key 可用）"
        assert results[2]["result"] == "ok"
        final = lines[-1]
        assert final["done"] is True and final["key_error"] is False
        assert final["skipped"] == 0, "401 单模型未授权不应中断探测"
        assert final["connect_fail"] is False and final["cancelled"] is False
        assert final["written"] is True, "移除项应写回模型清单"
        assert sorted(x["model"] for x in final["removed"]) == ["m-x401", "m-x404"]
        assert final["kept"] == ["m-ok-plain"]
    finally:
        client.delete("/api/providers/nimstream", headers=hdr)


# ---------------------------------------------------------------- R87 探测连不通上游：容忍抖动，连续 3 个才整轮快速放弃
def test_r87_probe_connect_fail_fast_abort(client, hdr, upstream_port):
    """base_url 指向连不通的地址（127.0.0.1:9 拒绝连接）：
    前 2 个 connfail 容忍（代理可能瞬时抖动，模型保留不删），第 3 个连续才整轮放弃
    （不逐个空等 300s），不改模型清单；done 带 connect_fail=true + connfails 列表。"""
    import time
    r1 = client.post("/api/providers", headers=hdr, json={
        "platform": "deadhost", "name": "死主机(fake)", "type": "openai",
        "base_url": "http://127.0.0.1:9/v1", "api_keys": ["pk"],
        "models": ["m-a", "m-b", "m-c"]})
    assert r1.status_code == 200, r1.text
    try:
        t0 = time.perf_counter()
        rp = client.post("/api/providers/deadhost/probe", headers=hdr, json={})
        elapsed = time.perf_counter() - t0
        assert rp.status_code == 200, rp.text
        j = rp.json()
        assert j["connect_fail"] is True and j["written"] is False, j
        assert j["removed"] == [] and j["cancelled"] is False
        assert j["connfails"] == ["m-a", "m-b", "m-c"], \
            f"连不通的模型记录在 connfails 且保留: {j}"
        assert j["skipped"] == 0, f"3 个模型全测过（第 3 个触发放弃）: {j}"
        assert elapsed < 30, f"连不通应在 30s 内快速放弃（实际 {elapsed:.1f}s）"
        gp = [p for p in client.get("/api/providers", headers=hdr).json()["data"]
              if p["platform"] == "deadhost"]
        assert gp[0]["models"] == ["m-a", "m-b", "m-c"], "连不通不动模型清单"
    finally:
        client.delete("/api/providers/deadhost", headers=hdr)




# ---------------------------------------------------------------- R90 探测结束自动清理 job：stop 端点对已结束的探测返回 stopped:false
def test_r90_probe_job_cleanup_after_done(client, hdr, upstream_port):
    """probe 跑完（done 事件）后 job 句柄应被清掉：再调 /probe/stop 返回 stopped:false。
    否则 stop 永远对着死 job set 事件，前端无法区分「有探测在跑/没有」。"""
    r1 = client.post("/api/providers", headers=hdr, json={
        "platform": "nimgone", "name": "NIMgone(fake)", "type": "openai",
        "base_url": f"http://127.0.0.1:{upstream_port}/v1", "api_keys": ["pk"],
        "models": ["m-ok-plain"]})
    assert r1.status_code == 200, r1.text
    try:
        rp = client.post("/api/providers/nimgone/probe", headers=hdr, json={})
        assert rp.status_code == 200 and rp.json()["done"] is True, rp.text
        sp = client.post("/api/providers/nimgone/probe/stop", headers=hdr, json={})
        assert sp.status_code == 200 and sp.json() == {"stopped": False},             f"已结束探测不应再算「stopped」: {sp.json()}"
    finally:
        client.delete("/api/providers/nimgone", headers=hdr)


# ---------------------------------------------------------------- R88 显式停止探测（「取消」按钮的 stop 端点）
def test_r88_probe_stop_endpoint(client, hdr, upstream_port):
    """点「取消」= POST /probe/stop：置 cancel 事件，探测循环在下个检查点停止，
    剩余模型记入 skipped、不写回清单——uvicorn 客户端断开不会取消服务端循环，
    所以前端取消必须走这个显式端点。"""
    import threading
    import time
    from fastapi.testclient import TestClient
    from app.main import app as relay_app
    from tests import fake_upstream

    r1 = client.post("/api/providers", headers=hdr, json={
        "platform": "nimslow", "name": "NIMslow(fake)", "type": "openai",
        "base_url": f"http://127.0.0.1:{upstream_port}/v1", "api_keys": ["pk"],
        "models": ["m-slow"] * 8})
    assert r1.status_code == 200, r1.text
    try:
        fake_upstream.RECEIVED.clear()
        stop_result = {}

        # 独立 portal 线程发 stop——探测流挂在 client 的 portal 上。
        # 注意：不用 with 块（会重跑 lifespan、把共享 store 关掉），直接实例化
        def stopper():
            time.sleep(2.2)   # 让前 1-2 个 0.8s 慢 ping 完成
            c2 = TestClient(relay_app)
            stop_result["stop"] = c2.post(
                "/api/providers/nimslow/probe/stop", headers=hdr).json()

        t = threading.Thread(target=stopper, daemon=True)
        t.start()
        with client.stream("POST", "/api/providers/nimslow/probe",
                           headers=hdr, json={"stream": True}) as resp:
            lines = [x for x in resp.iter_lines() if x.strip()]
        t.join(timeout=15)

        stop = stop_result.get("stop")
        assert stop == {"stopped": True}, f"stop 端点: {stop_result}"
        parsed = [json.loads(x) for x in lines]
        final = parsed[-1]
        assert final["done"] is True and final["cancelled"] is True, final
        assert final["written"] is False and final["skipped"] > 0, \
            f"取消后不得写回清单且应记录未测数: {final}"
        pings = [x for x in fake_upstream.RECEIVED if x["body"].get("model") == "m-slow"]
        assert len(pings) < 8, f"取消后服务端仍打满 8 个 ping: {len(pings)}"
        gp = [p for p in client.get("/api/providers", headers=hdr).json()["data"]
              if p["platform"] == "nimslow"]
        assert gp[0]["models"] == ["m-slow"] * 8, "取消不动模型清单"
    finally:
        client.delete("/api/providers/nimslow", headers=hdr)


# ---------------------------------------------------------------- R91 读超时 ≠ 连接失败：慢模型不触发整轮放弃
def test_r91_probe_slow_upstream_readtimeout_is_transient(client, hdr, upstream_port, monkeypatch):
    """gateway 把 httpx 异常分两类：连接层（拒绝/代理不可用/ConnectTimeout）→ 502 connfail；
    读/写超时（请求已发出、上游只是慢：NIM 冷启动、代理链路慢）→ 504「上游响应超时」。
    探测把 504 归 transient：模型保留、不计入「连续 3 个 connfail」放弃计数。
    旧逻辑把所有超时算 connfail，3 个慢模型就把整轮探测掐死（实测 8800 代理场景复现过）。"""
    import app.admin as A
    import httpx
    monkeypatch.setattr(A, "PROBE_PING_TIMEOUT", httpx.Timeout(1.0, connect=1.0))
    r1 = client.post("/api/providers", headers=hdr, json={
        "platform": "nimslowup", "name": "NIM慢上游(fake)", "type": "openai",
        "base_url": f"http://127.0.0.1:{upstream_port}/v1", "api_keys": ["pk"],
        "models": ["m-slow3", "m-slow3b", "m-slow3c"]})
    assert r1.status_code == 200, r1.text
    try:
        rp = client.post("/api/providers/nimslowup/probe", headers=hdr, json={})
        assert rp.status_code == 200, rp.text
        j = rp.json()
        assert j["connect_fail"] is False, \
            f"上游慢（读超时）不得判「连不通上游」整轮放弃: {j}"
        assert j["key_error"] is False and j["cancelled"] is False
        assert j["skipped"] == 0, "3 个慢模型都应测完，不提前收工"
        assert j.get("connfails") == [], f"读超时不算连接级 connfail: {j.get('connfails')}"
        assert sorted(j.get("transient") or []) == ["m-slow3", "m-slow3b", "m-slow3c"], \
            f"慢模型应进 transient（保留不误删）: {j}"
        assert j["kept"] == ["m-slow3", "m-slow3b", "m-slow3c"] and j["written"] is False
        gp = [p for p in client.get("/api/providers", headers=hdr).json()["data"]
              if p["platform"] == "nimslowup"]
        assert gp[0]["models"] == ["m-slow3", "m-slow3b", "m-slow3c"], "慢模型保留不删"
    finally:
        client.delete("/api/providers/nimslowup", headers=hdr)
