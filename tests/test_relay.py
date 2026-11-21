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
def test_r40_admin_user_management(client, hdr, monkeypatch):
    r = client.post("/api/users", headers=hdr, json={
        "email": "carol@test.local", "password": "carol-password", "role": "user"})
    assert "api_key" not in r.json(), "管理员建号不应拿到租户的 key"
    from app.main import store as _st
    carol_key = _st.reveal_api_key(_st.list_api_keys(r.json()["user"]["id"])[0]["id"])
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

    # 重置密码（v1.1.0）：管理员不能再直接设密码
    assert client.patch(f"/api/users/{uid}", headers=hdr,
                        json={"password": "carol-new-password"}).status_code == 400
    import app.mailer as _M
    captured = {}
    monkeypatch.setattr(_M, "send_password_reset_email",
                        lambda st_, to, pw, base, **kw: captured.update(pw=pw, to=to) or
                        {"mode": "log", "ok": True, "error": ""})
    rr = client.post(f"/api/users/{uid}/reset-password", headers=hdr)
    assert rr.status_code == 200, rr.text
    assert captured["to"] == "carol@test.local"
    new_pw = captured["pw"]
    assert new_pw and new_pw not in rr.text, "新密码自动生成且不能出现在响应里"
    r = client.post("/api/auth/login", json={
        "email": "carol@test.local", "password": new_pw})
    assert r.status_code == 200, "重置后的新密码可登录"
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

    # v1.1.0 二次验证：密码错 → 401（不留审计）
    assert client.post(f"/api/keys/{kid}/reveal", headers=hdr,
                       json={"password": "wrong"}).status_code == 401
    # 密码对：reveal 取回明文，能直接用来调下游
    rv = client.post(f"/api/keys/{kid}/reveal", headers=hdr,
                    json={"password": "test-password"})
    assert rv.status_code == 200 and rv.json()["key"] == plain
    assert client.get("/v1/models", headers={"authorization": f"Bearer {rv.json()['key']}"}).status_code == 200
    # 审计：成功取回留痕
    assert client.app.state.store.count_key_reveals(kid) == 1

    # 别的账号看不到这把 key（普通账号 → 404）
    r = client.post("/api/users", headers=hdr, json={
        "email": "dave@test.local", "password": "dave-password"})
    dave_key = client.app.state.store.reveal_api_key(client.app.state.store.list_api_keys(r.json()["user"]["id"])[0]["id"])
    # 用 dave 的 key 当 Bearer 打 /api（/api 也认 Bearer）
    assert client.post(f"/api/keys/{kid}/reveal",
                      headers={"authorization": f"Bearer {dave_key}"},
                      json={"password": "dave-password"}).status_code == 404

    # 真删除：记录消失、reveal 404、key 立即失效
    assert client.delete(f"/api/keys/{kid}", headers=hdr).status_code == 200
    assert kid not in [k["id"] for k in client.get("/api/keys", headers=hdr).json()["data"]]
    assert client.post(f"/api/keys/{kid}/reveal", headers=hdr,
                       json={"password": "test-password"}).status_code == 404
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
    frank_key = client.app.state.store.reveal_api_key(client.app.state.store.list_api_keys(r2.json()["user"]["id"])[0]["id"])
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


# ---------------------------------------------------------------- v1.1.0 密钥加密存储（R100-R106）
def test_r100_key_crypto_at_rest(client, hdr):
    """新建 key 立即写加密双列：key_mac（MAC 索引）+ key_enc（AES-GCM 可逆）。"""
    from app.main import store
    r = client.post("/api/keys", headers=hdr, json={"label": "crypto-test"})
    assert r.status_code == 200, r.text
    j = r.json()
    plain = j["key"]
    with store._lock:
        row = store._conn.execute(
            "SELECT key_mac, key_enc FROM api_keys WHERE id = ?", (j["id"],)).fetchone()
    assert row["key_mac"] and len(row["key_mac"]) == 32, "key_mac 应为 32B HMAC 摘要"
    assert row["key_enc"], "key_enc 必须落盘（复制按钮靠它取回明文）"
    assert plain.encode("utf-8") not in row["key_enc"], "密文里不能含明文"
    # 可逆性：服务端解密 == 原明文
    assert store.reveal_api_key(j["id"]) == plain
    client.delete(f"/api/keys/{j['id']}", headers=hdr)


def test_r101_legacy_key_migrated_and_auth_via_mac(client, hdr):
    """历史 key（cipher 注入前创建）由 lifespan 的 migrate_keys 补双列；验签走 MAC 索引。"""
    from app.main import store
    admin_id = store.first_admin()["id"]
    with store._lock:
        n = store._conn.execute(
            "SELECT COUNT(*) FROM api_keys WHERE user_id = ? AND label = '测试' AND key_mac IS NOT NULL",
            (admin_id,)).fetchone()[0]
    assert n >= 1, "conftest 的测试 key（cipher 注入前创建）应已被 lifespan 迁移补 key_mac"
    # 该 key 现在能正常调下游（验签主路径命中 MAC 索引）
    r = client.get("/v1/models", headers=hdr)
    assert r.status_code == 200


def test_r102_unmigrated_row_falls_back_to_plaintext_lookup(client, hdr):
    """模拟一条没迁移的历史行（key_mac NULL）：明文兜底验签仍可用，服务不炸。"""
    from app.main import store
    admin_id = store.first_admin()["id"]
    with store._lock:
        cur = store._conn.execute(
            "INSERT INTO api_keys (user_id, key, label, created_at) VALUES (?,?,?,?)",
            (admin_id, "sk-relay-unmigrated-test-key", "legacy-fallback", store._now()))
        legacy_id = cur.lastrowid
        store._conn.commit()
    try:
        r = client.get("/v1/models", headers={"authorization": "Bearer sk-relay-unmigrated-test-key"})
        assert r.status_code == 200, "未迁移历史 key 应走明文兜底验签"
    finally:
        with store._lock:
            store._conn.execute("DELETE FROM api_keys WHERE id = ?", (legacy_id,))
            store._conn.commit()


def test_r103_reveal_requires_owner_password_and_audits(client, hdr):
    """reveal 三件套：仅所有者 + 当前密码二次验证 + 审计留痕。"""
    from app.main import store
    import app.auth as A
    r = client.post("/api/keys", headers=hdr, json={"label": "reveal-test"})
    kid, plain = r.json()["id"], r.json()["key"]
    # 密码错 → 401，不留审计
    assert client.post(f"/api/keys/{kid}/reveal", headers=hdr,
                       json={"password": "definitely-wrong"}).status_code == 401
    assert store.count_key_reveals(kid) == 0
    # 密码对 → 200 + 明文 + 审计 +1
    ok = client.post(f"/api/keys/{kid}/reveal", headers=hdr, json={"password": "test-password"})
    assert ok.status_code == 200, ok.text
    assert ok.json()["key"] == plain
    assert store.count_key_reveals(kid) == 1
    # 别人家的 key → 404（即便密码填对）
    other = store.create_user("r103-other@test.local", A.hash_password("other-pass-9"), role="user")
    okey = store.create_api_key(other["id"], "sk-relay-r103-other", "x")
    assert client.post(f"/api/keys/{okey['id']}/reveal", headers=hdr,
                       json={"password": "test-password"}).status_code == 404
    client.delete(f"/api/keys/{kid}", headers=hdr)


def test_r104_register_no_plaintext_key_in_response(client, hdr):
    """注册/登录响应不再回明文 key（偷会话 ≠ 偷 key）：pending 账号 api_key 恒为空。"""
    r = client.post("/api/auth/register", json={"email": "r104-new@test.local",
                                                "password": "r104-pass-12345"})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["api_key"] == "", f"pending 账号不应带 key: {j}"
    assert "sk-relay" not in str(j["api_key"])


def test_r105_master_secret_resolution(monkeypatch, tmp_path):
    """主密钥 S 的解析规则：env > 文件 > 自动生成；已有加密 key 而 S 缺失 → 拒绝启动。"""
    import app.config as C
    import app.crypto as CR

    class _FakeStore:
        def __init__(self, n):
            self.n = n
        def has_encrypted_keys(self):
            return self.n > 0

    monkeypatch.delenv("RELAY_KEY_SECRET", raising=False)
    f = tmp_path / "master.key"
    monkeypatch.setattr(C, "MASTER_KEY_FILE", f)
    # ① 无加密 key 且无 S → 自动生成并落盘 0600
    sec = CR.resolve_master_secret(_FakeStore(0))
    assert f.exists() and f.read_bytes() == sec and len(sec) == 32
    # ② 已有加密 key 且 S 两处都缺 → 拒绝（宁可停服，不可换钥匙）
    f.unlink()
    import pytest as _pt
    with _pt.raises(RuntimeError):
        CR.resolve_master_secret(_FakeStore(5))
    # ③ 环境变量优先
    monkeypatch.setenv("RELAY_KEY_SECRET", "00112233445566778899aabbccddeeff")
    assert CR.resolve_master_secret(_FakeStore(5)) == bytes.fromhex("00112233445566778899aabbccddeeff")
    # ④ 密文 round-trip 且 MAC 确定性
    c = CR.KeyCipher(bytes.fromhex("00112233445566778899aabbccddeeff"))
    blob = c.seal("sk-relay-roundtrip-xyz")
    assert c.open(blob) == "sk-relay-roundtrip-xyz"
    assert c.mac("k") == c.mac("k") and c.mac("k") != c.mac("K")


def test_r106_migrate_keys_idempotent(client, hdr):
    """迁移幂等：lifespan 已跑过一轮，再调用应 0 行待迁移。"""
    from app.main import store
    assert store.migrate_keys() == 0

# ---------------------------------------------------------------- R107 key 归账号本人（管理员也拿不到）
def test_r107_admin_cannot_copy_tenant_key(client, hdr):
    """v1.1.0：建号不回明文 key；管理员 reveal 别人的 key 一律 404 且不写审计；本人可正常取回。"""
    from app.main import store

    # ① 管理员建号：响应里没有 api_key
    r = client.post("/api/users", headers=hdr, json={
        "email": "grace@test.local", "password": "grace-password"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert "api_key" not in body, "管理员建号不能拿到租户的 key"
    assert body.get("key_issued") is True
    uid = body["user"]["id"]
    kid = store.list_api_keys(uid)[0]["id"]

    # ② 管理员拿自己的密码去 reveal 租户的 key → 404（不泄露、不写审计）
    r = client.post(f"/api/keys/{kid}/reveal", headers=hdr, json={"password": "test-password"})
    assert r.status_code == 404, r.text
    assert "key" not in r.json()
    assert store.count_key_reveals(kid) == 0

    # ③ 本人可以取回自己的 key（用自己那把 key 当 Bearer + 自己的密码）
    tenant_key = store.reveal_api_key(kid)
    rk = client.post(f"/api/keys/{kid}/reveal",
                     headers={"authorization": f"Bearer {tenant_key}"},
                     json={"password": "grace-password"})
    assert rk.status_code == 200, rk.text
    assert rk.json()["key"] == tenant_key
    assert store.count_key_reveals(kid) == 1

    # ④ 管理员的设置页只能看到自己的 key（租户的 key 不出现在列表里）
    s = client.get("/api/settings", headers=hdr).json()
    assert kid not in [k["id"] for k in s["api_keys"]]

    # ⑤ 取回自己的 key 也要密码：错密码 401
    mine = s["api_keys"][0]["id"]
    assert client.post(f"/api/keys/{mine}/reveal", headers=hdr,
                       json={"password": "not-my-password"}).status_code == 401
# ---------------------------------------------------------------- R108 管理员视角看租户 key 清单（只读、掩码）
def test_r108_settings_tenant_view_readonly(client, hdr):
    from app.main import store as _st
    r = client.post("/api/users", headers=hdr, json={
        "email": "henry@test.local", "password": "henry-password"})
    assert r.status_code == 200
    hid = r.json()["user"]["id"]
    hkey = _st.reveal_api_key(_st.list_api_keys(hid)[0]["id"])
    assert hkey.startswith("sk-relay-")

    # 管理员 ?tenant=hid → 该账号的 key 清单（掩码），viewing_user 指明在看谁
    s = client.get(f"/api/settings?tenant={hid}", headers=hdr).json()
    assert s["viewing_user"] and s["viewing_user"]["email"] == "henry@test.local"
    assert len(s["api_keys"]) == 1
    assert "…" in s["api_keys"][0]["key"] and s["api_keys"][0]["key"] != hkey, "只给掩码"

    # 不带 tenant / all / 不存在的 id → 看自己的
    assert client.get("/api/settings", headers=hdr).json()["viewing_user"] is None
    for q in ("all", "999999"):
        assert client.get(f"/api/settings?tenant={q}", headers=hdr).json()["viewing_user"] is None, q

    # 普通账号带 tenant 参数 → 后端强制只看自己
    login = client.post("/api/auth/login", json={"email": "henry@test.local", "password": "henry-password"})
    assert login.status_code == 200
    cc = {"relay_session": login.cookies.get("relay_session")}
    s4 = client.get("/api/settings?tenant=1", cookies=cc).json()
    assert s4["viewing_user"] is None and len(s4["api_keys"]) == 1

    # 视角不改变权限：管理员视角下 reveal 他人 key 仍 404（cookie 优先于 Bearer，先清掉 henry 的会话）
    client.cookies.clear()
    kid = _st.list_api_keys(hid)[0]["id"]
    assert client.post(f"/api/keys/{kid}/reveal", headers=hdr,
                       json={"password": "test-password"}).status_code == 404
# ---------------------------------------------------------------- R109 全新部署无 S 也能启动；租户上游 key 写入即加密落盘
def test_r109_fresh_deploy_and_tenant_key_encryption(client, hdr):
    """全新部署（无 master.key）直接启动；管理员写入的上游 key 磁盘上只有密文。"""
    import json as _json
    import secrets as _secrets
    from app.main import store
    from app import config as C

    u1 = int(store.first_admin()["id"])
    f = C.CONFIG_DIR / "tenants" / f"u{u1}" / "upstreams.json"
    assert f.exists(), "首管理员租户文件应已存在（legacy 认领或空配置）"

    # 拿一个假上游的 base_url（conftest 种子平台），造一个新平台
    plats = client.get("/api/providers", headers=hdr).json()["data"]
    base = next(p["base_url"] for p in plats if p["platform"] == "fake")
    k = "up-" + _secrets.token_urlsafe(24)
    r = client.post("/api/providers", headers=hdr, json={
        "platform": "vt109", "name": "v109", "type": "openai",
        "base_url": base, "api_keys": [k], "models": ["vt109-m"]})
    assert r.status_code == 200, r.text

    # 磁盘无明文，只有密文
    txt = f.read_text(encoding="utf-8")
    assert k not in txt, "上游 key 明文不能落盘"
    row = next(x for x in _json.loads(txt) if x.get("platform") == "vt109")
    assert row["enc_keys"] and not row.get("api_keys") and row.get("api_key", "") == ""
    # 新建 key 的响应回掩码，不回明文
    assert k not in r.text

    # 内存里解开了：relay 直接 200
    rr = client.post("/v1/chat/completions", headers=hdr, json={
        "model": "vt109/vt109-m", "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 4})
    assert rr.status_code == 200, rr.text

    # 全新部署的主密钥是自动生成的（0600）
    assert C.MASTER_KEY_FILE.exists()

    # 收尾：删掉测试平台，别污染其他用例
    assert client.delete("/api/providers/vt109", headers=hdr).status_code == 200
# ---------------------------------------------------------------- R110 加密文件跨重启/跨版本可直接启动
def test_r110_encrypted_tenant_file_survives_restart(client, hdr):
    """重启（或升到高版本）后，同一把 S 下 K_uid 解得开加密租户文件。"""
    import json as _json
    from app.main import store
    from app.registry import Registry
    from app.crypto import tenant_cipher
    from app import config as C

    u1 = int(store.first_admin()["id"])
    f = C.CONFIG_DIR / "tenants" / f"u{u1}" / "upstreams.json"
    rows = _json.loads(f.read_text(encoding="utf-8"))
    assert any(x.get("enc_keys") for x in rows if isinstance(x, dict)), "应先有加密行（R109/lifespan 迁移）"

    secret = C.MASTER_KEY_FILE.read_bytes()
    clients = C.CONFIG_DIR / "tenants" / "_clients.json"
    reg = Registry(f, clients, cipher=tenant_cipher(secret, u1))
    assert any(u.api_keys for u in reg.upstreams), "重启后密文文件应解得开"
    live_keys = {u.platform: len(u.api_keys) for u in reg.upstreams if u.api_keys}
    assert live_keys, live_keys

    # 换一个 uid 的域（K_9999）：同一份密文解不开 → 平台按无 key 降级、不崩
    other = Registry(f, clients, cipher=tenant_cipher(secret, 9999))
    for p, n in live_keys.items():
        up = other.find(p)
        assert up is not None and not up.api_keys, (p, n)
# ---------------------------------------------------------------- R111 老版明文租户文件自动迁移（幂等）
def test_r111_legacy_plaintext_tenant_file_migrated(client, hdr):
    """v1.1.0 时代的明文租户文件：迁移后磁盘无明文、幂等、重开 Registry 仍可用。"""
    import json as _json
    from app.main import store
    from app.registry import Registry
    from app.tenants import migrate_one_tenant_file
    from app.crypto import tenant_cipher
    from app import config as C

    u1 = int(store.first_admin()["id"])
    f = C.CONFIG_DIR / "tenants" / f"u{u1}" / "upstreams.json"
    rows = _json.loads(f.read_text(encoding="utf-8"))
    rows.append({"platform": "vt111", "name": "legacy plain", "type": "openai",
                 "base_url": "http://127.0.0.1:9", "api_key": "legacy-plain-up-key",
                 "models": ["x"]})
    f.write_text(_json.dumps(rows, ensure_ascii=False, indent=2), "utf-8")

    secret = C.MASTER_KEY_FILE.read_bytes()
    assert migrate_one_tenant_file(f, u1, secret) is True
    txt = f.read_text(encoding="utf-8")
    assert "legacy-plain-up-key" not in txt, "迁移后明文必须消失"
    before = f.read_bytes()
    assert migrate_one_tenant_file(f, u1, secret) is False, "迁移应幂等"
    assert f.read_bytes() == before

    reg = Registry(f, C.CONFIG_DIR / "tenants" / "_clients.json",
                   cipher=tenant_cipher(secret, u1))
    assert reg.find("vt111").api_keys == ["legacy-plain-up-key"], "迁移后 key 应能用"

    # 收尾：删掉 vt111 行（enc 行原样保留，直接写文件，走下一轮热重载）
    left = [x for x in _json.loads(f.read_text(encoding="utf-8"))
            if x.get("platform") != "vt111"]
    f.write_text(_json.dumps(left, ensure_ascii=False, indent=2), "utf-8")
# ---------------------------------------------------------------- R112 K_t 域分离：A 租户密文 B 租户解不开
def test_r112_kt_domain_separation(client):
    import pytest
    from cryptography.exceptions import InvalidTag
    from app.crypto import tenant_cipher
    from app import config as C

    secret = C.MASTER_KEY_FILE.read_bytes()
    c1, c2 = tenant_cipher(secret, 1), tenant_cipher(secret, 2)
    blob = c1.seal_str("some-upstream-key")
    assert c1.open_str(blob) == "some-upstream-key"
    with pytest.raises(InvalidTag):
        c2.open_str(blob)
    # 同 uid 派生确定（无状态）：重新算一遍还能解
    assert tenant_cipher(secret, 1).open_str(blob) == "some-upstream-key"
# ---------------------------------------------------------------- R113 密文被篡改：不崩，平台降级为无 key
def test_r113_tampered_ciphertext_degrades(client, hdr, tmp_path):
    import json as _json
    from app.main import store
    from app.registry import Registry
    from app.crypto import tenant_cipher
    from app import config as C

    u1 = int(store.first_admin()["id"])
    f = C.CONFIG_DIR / "tenants" / f"u{u1}" / "upstreams.json"
    rows = _json.loads(f.read_text(encoding="utf-8"))
    assert any(isinstance(x, dict) and x.get("enc_keys") for x in rows)
    # 复制一份到 tmp 再改坏（不碰活文件，避免 5s 热重载把活实例的平台弄丢 key）
    copy = tmp_path / "tampered.json"
    bad = _json.loads(_json.dumps(rows))
    b = next(r for r in bad if isinstance(r, dict) and r.get("enc_keys"))
    s0 = b["enc_keys"][0]
    b["enc_keys"][0] = ("A" if s0[0] != "A" else "B") + s0[1:]
    copy.write_text(_json.dumps(bad, ensure_ascii=False, indent=2), "utf-8")

    secret = C.MASTER_KEY_FILE.read_bytes()
    reg = Registry(copy, tmp_path / "c.json", cipher=tenant_cipher(secret, u1))
    up = reg.find(b["platform"])
    assert up is not None and not up.api_keys, "篡改的平台应降级为无 key（其他平台不受影响）"
    others = [u for u in reg.upstreams if u.platform != up.platform and u.api_keys]
    assert others, "同文件其他平台的 key 仍应解得开"
# ---------------------------------------------------------------- R114 缺 S + 有加密租户配置 → fail-fast
def test_r114_missing_master_secret_failfast(monkeypatch, tmp_path):
    import pytest
    import app.config as C
    import app.crypto as CR
    from app.tenants import has_encrypted_tenant_files

    assert has_encrypted_tenant_files(C.CONFIG_DIR / "tenants", C.UPSTREAMS_FILE), \
        "前面用例/迁移后应已有加密租户配置"
    monkeypatch.delenv("RELAY_KEY_SECRET", raising=False)
    monkeypatch.setattr(C, "MASTER_KEY_FILE", tmp_path / "nowhere-master.key")

    class _FakeStore:
        def has_encrypted_keys(self):
            return False

    with pytest.raises(RuntimeError):
        CR.resolve_master_secret(_FakeStore(), extra_encrypted=True)
    # 无加密材料时同样缺 S → 不炸（自动生成路径，全新部署保证）
    got = CR.resolve_master_secret(_FakeStore(), extra_encrypted=False)
    assert len(got) == 32 and (tmp_path / "nowhere-master.key").exists()
# ---------------------------------------------------------------- R115 聚合标记行与密文共存 + GET 仍回掩码
def test_r115_aggregates_and_mask_coexist(client, hdr):
    import json as _json
    from app.main import store
    from app import config as C

    r = client.post("/api/aggregates", headers=hdr, json={
        "name": "vt115", "entries": ["fake/fake-gpt"]})
    assert r.status_code == 200, r.text
    f = C.CONFIG_DIR / "tenants" / f"u{int(store.first_admin()['id'])}" / "upstreams.json"
    rows = _json.loads(f.read_text(encoding="utf-8"))
    assert any(x.get("_kind") == "aggregates" and x["aggregates"].get("vt115") for x in rows), \
        "聚合标记行应保留"
    enc = [x for x in rows if isinstance(x, dict) and x.get("enc_keys")]
    assert enc, "加密行应仍在"
    # GET /api/providers 的 key 是掩码（既不是明文也不是原始密文）
    d = client.get("/api/providers", headers=hdr).json()["data"]
    fake = next(p for p in d if p["platform"] == "fake")
    assert all("…" in k for k in fake["api_keys"]) and fake["api_keys"]
    client.delete("/api/aggregates/vt115", headers=hdr)
# ---------------------------------------------------------------- R116 汇总视图：按账号列出全部上游平台 + 聚合模型
def test_r116_all_mode_lists_every_tenant(client, hdr):
    # 建两个普通账号，各自登记一个平台 + 同名聚合（跨账号重名要都能列出来）
    ra = client.post("/api/users", headers=hdr, json={
        "email": "aggx-a@test.local", "password": "aggx-password", "role": "user"})
    assert ra.status_code == 200, ra.text
    uid_a = ra.json()["user"]["id"]
    rb = client.post("/api/users", headers=hdr, json={
        "email": "aggx-b@test.local", "password": "aggx-password", "role": "user"})
    assert rb.status_code == 200, rb.text
    uid_b = rb.json()["user"]["id"]

    ca = _session_cookie(client, "aggx-a@test.local", "aggx-password")
    cb = _session_cookie(client, "aggx-b@test.local", "aggx-password")
    for ck, uid, plat, model in ((ca, uid_a, "aggplat", "aggm1"),
                                 (cb, uid_b, "aggplat2", "aggm2")):
        r = client.post("/api/providers", cookies=ck, json={
            "platform": plat, "type": "openai", "base_url": "http://127.0.0.1:9/v1",
            "api_keys": ["sk-agg-test"], "models": [model]})
        assert r.status_code == 200, r.text
        r = client.post("/api/aggregates", cookies=ck, json={
            "name": "aggpool", "entries": [f"{plat}/{model}"]})
        assert r.status_code == 200, r.text

    # 汇总视图：两个账号的平台都在，且带账号归属
    d = client.get("/api/providers?tenant=all", headers=hdr).json()
    assert d["all"] is True
    tenants = {row["tenant"] for row in d["data"]}
    assert uid_a in tenants and uid_b in tenants, (tenants, "两个账号都应出现在汇总里")
    by = {(row["tenant"], row["platform"]) for row in d["data"]}
    assert (uid_a, "aggplat") in by and (uid_b, "aggplat2") in by
    assert not any(row["platform"] in ("aggplat", "aggplat2") and row["tenant"] not in (uid_a, uid_b)
                   for row in d["data"])
    labels = {row["tenant"]: row["tenant_label"] for row in d["data"]}
    assert labels[uid_a] == "aggx-a@test.local" and labels[uid_b] == "aggx-b@test.local"

    # 聚合模型：同名 aggpool 两个账号各一条，都带归属
    ag = client.get("/api/aggregates?tenant=all", headers=hdr).json()
    assert ag["all"] is True
    rows = {(row["tenant"], row["name"]) for row in ag["data"]}
    assert (uid_a, "aggpool") in rows and (uid_b, "aggpool") in rows

    # 概览/模型清单也按全量算
    ov = client.get("/api/overview?tenant=all", headers=hdr).json()
    assert ov["providers"] >= 2 and ov["models"] >= 2
    ids = client.get("/api/models?tenant=all", headers=hdr).json()
    assert "aggplat/aggm1" in ids["ids"] and "aggplat2/aggm2" in ids["ids"]
    # v1.2.2：多租户合并的三组分组（panel 分组渲染数据源）：去重 + 组内排序
    g = ids["groups"]
    assert g["full"] == sorted(g["full"]) and g["bare"] == sorted(g["bare"]) and g["aggs"] == sorted(g["aggs"])
    assert {"aggplat/aggm1", "aggplat2/aggm2"} <= set(g["full"]), g["full"]
    assert {"aggm1", "aggm2"} <= set(g["bare"]), g["bare"]
    assert g["aggs"].count("aggpool") == 1 and "aggpool" in g["aggs"], f"同名聚合应去重: {g['aggs']}"
    # 汇总视图按用户维度再分组：逐账号的三组（label=邮箱/显示名）
    tmap = {t["id"]: t for t in ids["tenants"]}
    assert uid_a in tmap and uid_b in tmap, f"两个账号都要有逐组数据: {list(tmap)}"
    assert "aggplat/aggm1" in tmap[uid_a]["groups"]["full"], tmap[uid_a]
    assert "aggplat2/aggm2" in tmap[uid_b]["groups"]["full"], tmap[uid_b]
    for t in ids["tenants"]:
        assert t["groups"]["full"] == sorted(t["groups"]["full"]), f"用户 {t['id']} 组内要排序"


# ---------------------------------------------------------------- R117 汇总视图是只读的：写/探测端点 400
def test_r117_all_mode_is_readonly(client, hdr):
    cases = [
        ("POST", "/api/providers?tenant=all",
         {"platform": "aggx3", "type": "openai", "base_url": "http://127.0.0.1:9/v1",
          "api_keys": ["sk-x"], "models": ["m"]}),
        ("POST", "/api/aggregates?tenant=all", {"name": "aggx3", "entries": ["aggx3/m"]}),
        ("DELETE", "/api/providers/aggplat?tenant=all", None),
        ("POST", "/api/providers/aggplat/test?tenant=all", None),
        ("POST", "/api/providers/aggplat/probe?tenant=all", None),
    ]
    for method, path, body in cases:
        if method == "DELETE":
            r = client.delete(path, headers=hdr)
        else:
            r = client.post(path, headers=hdr, json=body if body is not None else None)
        assert r.status_code == 400, (method, path, r.status_code, r.text)
        assert "只读" in r.json()["detail"], (method, path, r.text)
    # 同一批操作不带 tenant=all 时不受影响（管理员自己的配置仍可写）
    ok = client.post("/api/aggregates", headers=hdr,
                     json={"name": "aggxguard", "entries": ["fake/fake-gpt"]})
    assert ok.status_code == 200, ok.text
    client.delete("/api/aggregates/aggxguard", headers=hdr)


# ---------------------------------------------------------------- R118 老配置认领后不重复计数
def test_r118_legacy_no_double_count(client, hdr):
    d = client.get("/api/providers?tenant=all", headers=hdr).json()
    legacy_rows = [row for row in d["data"] if row["platform"] == "fake"]
    assert len(legacy_rows) == 1, (
        f"老配置的平台在汇总里应只出现一次，实际 {len(legacy_rows)} 次"
        + ("（被认领后不应再单列「遗留配置」）" if client.app.state.tenants.claimed else ""))
    t = client.app.state.tenants
    if not t.claimed:
        assert legacy_rows[0]["tenant"] == 0 and "遗留" in legacy_rows[0]["tenant_label"]
    else:
        assert legacy_rows[0]["tenant"] != 0, "认领后应挂在认领者账号名下"


# ---------------------------------------------------------------- R119 指定账号视图不变（含聚合模型跟随账号）
def test_r119_single_tenant_scoping(client, hdr):
    users = {u["email"]: u["id"] for u in client.app.state.store.list_users()}
    uid_a = users["aggx-a@test.local"]
    d = client.get(f"/api/providers?tenant={uid_a}", headers=hdr).json()
    plats = {row["platform"] for row in d["data"]}
    assert "aggplat" in plats and "aggplat2" not in plats and "fake" not in plats
    ag = client.get(f"/api/aggregates?tenant={uid_a}", headers=hdr).json()
    assert [row["name"] for row in ag["data"]] == ["aggpool"], "指定账号只回该账号的聚合"
# ---------------------------------------------------------------- R120 多选汇总：?tenant=a,b 只列勾选的账号
def test_r120_subset_merges_only_selected(client, hdr):
    def _mk(email):
        r = client.post("/api/users", headers=hdr, json={
            "email": email, "password": "agm-password", "role": "user"})
        assert r.status_code == 200, r.text
        return r.json()["user"]["id"]
    ua = _mk("agm-a@test.local")
    ub = _mk("agm-b@test.local")
    for email, plat, model in (("agm-a@test.local", "agmplat", "agmm1"),
                               ("agm-b@test.local", "agmplat2", "agmm2")):
        ck = _session_cookie(client, email, "agm-password")
        r = client.post("/api/providers", cookies=ck, json={
            "platform": plat, "type": "openai", "base_url": "http://127.0.0.1:9/v1",
            "api_keys": ["sk-agg-test"], "models": [model]})
        assert r.status_code == 200, r.text
        r = client.post("/api/aggregates", cookies=ck, json={
            "name": "agmpool", "entries": [f"{plat}/{model}"]})
        assert r.status_code == 200, r.text

    d = client.get(f"/api/providers?tenant={ua},{ub}", headers=hdr).json()
    assert d["all"] is True
    tenants = {row["tenant"] for row in d["data"]}
    assert tenants == {ua, ub}, f"只能出现勾选的账号，实际 {tenants}"
    assert not any(row["platform"] == "fake" for row in d["data"]), \
        "管理员自己的平台不应混进多选汇总"
    labels = {row["tenant"]: row["tenant_label"] for row in d["data"]}
    assert labels[ua] == "agm-a@test.local" and labels[ub] == "agm-b@test.local"

    ag = client.get(f"/api/aggregates?tenant={ua},{ub}", headers=hdr).json()
    rows = {(row["tenant"], row["name"]) for row in ag["data"]}
    assert (ua, "agmpool") in rows and (ub, "agmpool") in rows

    ov = client.get(f"/api/overview?tenant={ua},{ub}", headers=hdr).json()
    assert ov["providers"] == 2, f"两个账号各一个平台，应共 2 个，实际 {ov['providers']}"
    ids = client.get(f"/api/models?tenant={ua},{ub}", headers=hdr).json()["ids"]
    assert "agmplat/agmm1" in ids and "agmplat2/agmm2" in ids


# ---------------------------------------------------------------- R121 多选时报表也按勾选项过滤（user_id IN）
def test_r121_subset_filters_reports(client, hdr):
    from app.main import store

    def _mk(email):
        r = client.post("/api/users", headers=hdr, json={
            "email": email, "password": "agm-password", "role": "user"})
        assert r.status_code == 200, r.text
        return r.json()["user"]["id"]
    ux = _mk("agm-x@test.local")
    uy = _mk("agm-y@test.local")
    me = store.first_admin()["id"]
    me_email = next(u["email"] for u in store.list_users() if u["id"] == me)
    for uid, model in ((me, "rm-me"), (ux, "rm-x"), (uy, "rm-y")):
        store.record(platform="rplat", model=model, total_tokens=100, user_id=uid)

    rows = client.get(f"/api/stats/summary?group_by=account&tenant={ux},{uy}", headers=hdr).json()["data"]
    got = {row["k"] for row in rows}
    assert "agm-x@test.local" in got and "agm-y@test.local" in got, got
    assert me_email not in got, f"多选报表不应包含未勾选的管理员数据：{got}"

    rows1 = client.get(f"/api/stats/summary?group_by=account&tenant={uy}", headers=hdr).json()["data"]
    assert {row["k"] for row in rows1} == {"agm-y@test.local"}, "单选报表只看该账号"

    ov = client.get(f"/api/overview?tenant={ux},{uy}", headers=hdr).json()
    assert ov["records"] == 2, f"记录数应只算勾选的两个账号，实际 {ov['records']}"
    fac = client.get(f"/api/stats/logs/facets?tenant={ux},{uy}", headers=hdr).json()
    assert fac["total"] == 2
    assert {a["id"] for a in fac["accounts"]} == {ux, uy}


# ---------------------------------------------------------------- R122 多选只读 / 单选可写
def test_r122_subset_readonly_single_writable(client, hdr):
    def _mk(email):
        r = client.post("/api/users", headers=hdr, json={
            "email": email, "password": "agm-password", "role": "user"})
        assert r.status_code == 200, r.text
        return r.json()["user"]["id"]
    uw = _mk("agm-w@test.local")
    uz = _mk("agm-z@test.local")
    # 多选 → 写端点 400 只读
    r = client.post(f"/api/providers?tenant={uw},{uz}", headers=hdr, json={
        "platform": "agm-multi", "type": "openai", "base_url": "http://127.0.0.1:9/v1",
        "api_keys": ["sk-x"], "models": ["m"]})
    assert r.status_code == 400 and "只读" in r.json()["detail"], r.text
    r = client.post(f"/api/aggregates?tenant={uw},{uz}", headers=hdr,
                    json={"name": "agm-multi", "entries": ["agm-multi/m"]})
    assert r.status_code == 400, r.text
    # 单选仍是全功能管理视图
    ok = client.post(f"/api/providers?tenant={uw}", headers=hdr, json={
        "platform": "agm-single", "type": "openai", "base_url": "http://127.0.0.1:9/v1",
        "api_keys": ["sk-x"], "models": ["m"]})
    assert ok.status_code == 200, ok.text
    assert client.delete(f"/api/providers/agm-single?tenant={uw}", headers=hdr).status_code == 200
    d = client.get(f"/api/providers?tenant={uw}", headers=hdr).json()
    assert "agm-single" not in {row["platform"] for row in d["data"]}


# ---------------------------------------------------------------- R123 anthropic 协议 /v1 去重（双 v1 → 404 修复）
def test_r123_urlfor_v1_dedup():
    from app.registry import Upstream
    # base 带 /v1（Agnes 风格 hub，同时提供 Messages）：anthropic 路径去掉前导 /v1，不再双 v1
    a1 = Upstream(platform="a1", base_url="https://x.example/v1", type="anthropic")
    assert a1.url_for("messages") == "https://x.example/v1/messages"
    assert a1.url_for("models") == "https://x.example/v1/models"
    assert a1.url_for("count_tokens") == "https://x.example/v1/messages/count_tokens"
    # openai 协议不受影响（默认路径本就不带 /v1）
    o1 = Upstream(platform="o1", base_url="https://x.example/v1", type="openai")
    assert o1.url_for("chat") == "https://x.example/v1/chat/completions"
    assert o1.url_for("models") == "https://x.example/v1/models"
    # 官方 Anthropic（base 不带 /v1）：保留带 /v1 的默认路径
    a2 = Upstream(platform="a2", base_url="https://api.anthropic.com", type="anthropic")
    assert a2.url_for("messages") == "https://api.anthropic.com/v1/messages"
    assert a2.url_for("models") == "https://api.anthropic.com/v1/models"
    # extra 路径覆盖仍然优先，且参与去重
    a3 = Upstream(platform="a3", base_url="https://x.example/v1", type="anthropic",
                  extra={"messages_path": "/v1/messages"})
    assert a3.url_for("messages") == "https://x.example/v1/messages"


# ---------------------------------------------------------------- R124 key↔模型/聚合绑定（v1.2.0 门禁 + 审计 + 列表过滤）
def test_r124_key_model_binding(client, hdr):
    """带绑定的 key：绑定的模型/聚合名 200；未绑定的模型 403 且写请求日志（dkey_id 归属）；
    /v1/models 只列绑定名；不绑定的 key（存量）不限制；count_tokens 同样走门禁。"""
    store = client.app.state.store
    r = client.post("/api/aggregates", headers=hdr,
                    json={"name": "r124agg", "entries": ["fake/fake-gpt", "fake/fake-gpt-2"]})
    assert r.status_code == 200, r.text
    kid = None
    try:
        cr = client.post("/api/keys", headers=hdr,
                         json={"label": "r124bound", "models": ["fake/fake-gpt", "r124agg"]})
        assert cr.status_code == 200, cr.text
        kid = cr.json()["id"]; rk = cr.json()["key"]
        bh = {"authorization": f"Bearer {rk}"}

        ok1 = client.post("/v1/chat/completions", headers=bh, json={"model": "fake/fake-gpt", "messages": []})
        assert ok1.status_code == 200, ok1.text
        ok2 = client.post("/v1/chat/completions", headers=bh, json={"model": "r124agg", "messages": []})
        assert ok2.status_code == 200, ok2.text

        rr = client.post("/v1/chat/completions", headers=bh, json={"model": "fake/fake-gpt-2", "messages": []})
        assert rr.status_code == 403 and "未绑定" in rr.text, rr.text
        rows = store.recent(5)
        hit = [x for x in rows if x.get("dkey_id") == kid and x.get("status") == "error"]
        assert hit and "未绑定" in (hit[0].get("error") or ""), \
            f"403 拦截要写进请求日志（dkey_id 归属）: {rows}"

        ml = client.get("/v1/models", headers=bh).json()
        ids = {i["id"] for i in ml["data"]}
        assert "fake/fake-gpt" in ids and "r124agg" in ids, f"绑定名必须在列: {ids}"
        assert "fake/fake-gpt-2" not in ids and "fakeclaude/fake-claude" not in ids, \
            f"未绑定名不该列出: {ids}"

        u2 = client.post("/v1/chat/completions", headers=hdr, json={"model": "fake/fake-gpt-2", "messages": []})
        assert u2.status_code == 200, "不绑定的 key 不受限制（向后兼容）"

        ct = client.post("/v1/messages/count_tokens", headers=bh, json={"model": "fake/fake-gpt-2", "messages": []})
        assert ct.status_code == 403, f"count_tokens 同样走门禁: {ct.text}"
    finally:
        client.delete("/api/aggregates/r124agg", headers=hdr)
        if kid is not None:
            store.revoke_api_key(me_id(client), kid)


def me_id(client):
    return [u for u in client.app.state.store.list_users() if u.get("role") == "admin"][0]["id"]


# ---------------------------------------------------------------- R125 绑定形式匹配 + PATCH 三态
def test_r125_key_model_binding_forms_and_patch(client, hdr):
    """形式匹配：绑裸名 ≠ 允许全名；别名可绑可直调；PATCH 三态（省略=不动 / []=清空 / list=覆盖）。
    别名自包含（自建 r125-alias，不依赖 fake 平台既有 model_map——其他测试会覆盖它）。"""
    p = client.get("/api/providers", headers=hdr).json()["data"]
    fake = next(x for x in p if x["platform"] == "fake")
    orig_map = dict(fake.get("model_map") or {})
    base = {"platform": "fake", "name": fake["name"], "type": fake["type"],
            "base_url": fake["base_url"], "api_keys": fake["api_keys"],
            "models": fake["models"], "enabled": True, "timeout_s": fake.get("timeout_s", 300)}
    assert client.post("/api/providers", headers=hdr,
                       json={**base, "model_map": {**orig_map, "r125-alias": "fake-gpt"}}).status_code == 200
    r = client.post("/api/keys", headers=hdr,
                    json={"label": "r125", "models": ["fake-gpt", "r125-alias"]})
    assert r.status_code == 200, r.text
    kid = r.json()["id"]
    bh = {"authorization": f"Bearer {r.json()['key']}"}
    try:
        ok1 = client.post("/v1/chat/completions", headers=bh, json={"model": "fake-gpt", "messages": []})
        assert ok1.status_code == 200, ok1.text
        ok2 = client.post("/v1/chat/completions", headers=bh, json={"model": "r125-alias", "messages": []})
        assert ok2.status_code == 200, f"别名直呼应可用: {ok2.text}"

        rr = client.post("/v1/chat/completions", headers=bh, json={"model": "fake/fake-gpt", "messages": []})
        assert rr.status_code == 403, "绑裸名 ≠ 允许全名（形式也是绑定的一部分）"

        # 省略 models = 不动
        p1 = client.patch(f"/api/keys/{kid}", headers=hdr, json={"label": "r125-2"})
        assert p1.status_code == 200 and p1.json()["models"] == ["fake-gpt", "r125-alias"], p1.text
        assert client.post("/v1/chat/completions", headers=bh,
                           json={"model": "fake-gpt", "messages": []}).status_code == 200

        # [] = 清空 → 不限制
        p2 = client.patch(f"/api/keys/{kid}", headers=hdr, json={"models": []})
        assert p2.status_code == 200 and p2.json()["models"] is None, p2.text
        free = client.post("/v1/chat/completions", headers=bh, json={"model": "fake/fake-gpt-2", "messages": []})
        assert free.status_code == 200, "清空绑定后不限制"

        # list = 覆盖
        p3 = client.patch(f"/api/keys/{kid}", headers=hdr, json={"models": ["fake/fake-gpt-2"]})
        assert p3.status_code == 200 and p3.json()["models"] == ["fake/fake-gpt-2"], p3.text
        locked = client.post("/v1/chat/completions", headers=bh, json={"model": "fake/fake-gpt", "messages": []})
        assert locked.status_code == 403, "覆盖后旧绑定失效"
    finally:
        client.patch(f"/api/keys/{kid}", headers=hdr, json={"models": []})
        client.post("/api/providers", headers=hdr, json={**base, "model_map": orig_map})


# ---------------------------------------------------------------- R126 管理 API 校验 + 设置页字段
def test_r126_key_model_binding_admin_api(client, hdr):
    """POST 校验（无效名 400 并列出）；GET /api/settings 与 /api/keys 的 key 行带 models/models_missing，
    /api/settings 带 model_name_options/aggregate_names（R73 式多端点字段回归锁）。"""
    r = client.post("/api/keys", headers=hdr, json={"label": "r126bad", "models": ["no-such-model"]})
    assert r.status_code == 400 and "no-such-model" in r.text, r.text
    r2 = client.post("/api/keys", headers=hdr, json={"label": "r126ok", "models": ["fake/fake-gpt"]})
    assert r2.status_code == 200 and r2.json()["models"] == ["fake/fake-gpt"], r2.text
    kid = r2.json()["id"]

    st = client.get("/api/settings", headers=hdr).json()
    assert "model_name_options" in st and "bare_model_names" in st and "aggregate_names" in st, \
        "/api/settings 要带三组模型名选项（v1.2.2）"
    assert "fake/fake-gpt" in st["model_name_options"], st["model_name_options"]
    assert "fake-gpt" not in st["model_name_options"], "全名组只含 platform/ 前缀行"
    assert "fake-gpt" in st["bare_model_names"], "裸名组恢复裸模型名（v1.2.2 下半场）"
    row = next(k for k in st["api_keys"] if k["id"] == kid)
    assert row["models"] == ["fake/fake-gpt"] and row["models_missing"] == [], row
    assert all("models" in k and "models_missing" in k for k in st["api_keys"]), "每行都要有绑定字段"

    ks = client.get("/api/keys", headers=hdr).json()["data"]
    krow = next(k for k in ks if k["id"] == kid)
    assert krow["models"] == ["fake/fake-gpt"] and krow["models_missing"] == []
    client.delete(f"/api/keys/{kid}", headers=hdr)


# ---------------------------------------------------------------- R127 悬空绑定 + 老库迁移
def test_r127_key_model_binding_dangling_and_migration(client, hdr, upstream_port):
    """绑定的模型被删后：调该名走自然 404（门禁放行、resolve 兜底），key 行 models_missing 警告；
    老库（api_keys 无 model_bindings 列）Store 打开自动迁移，旧 key 不限制。"""
    p = client.post("/api/providers", headers=hdr, json={
        "platform": "r127plat", "name": "R127(fake)", "type": "openai",
        "base_url": f"http://127.0.0.1:{upstream_port}/v1", "api_keys": ["sk-x"], "models": ["r127m"]})
    assert p.status_code == 200, p.text
    cr = client.post("/api/keys", headers=hdr, json={"label": "r127", "models": ["r127plat/r127m"]})
    assert cr.status_code == 200, cr.text
    bh = {"authorization": f"Bearer {cr.json()['key']}"}
    try:
        ok = client.post("/v1/chat/completions", headers=bh, json={"model": "r127plat/r127m", "messages": []})
        assert ok.status_code == 200, ok.text

        assert client.delete("/api/providers/r127plat", headers=hdr).status_code == 200
        rr = client.post("/v1/chat/completions", headers=bh, json={"model": "r127plat/r127m", "messages": []})
        assert rr.status_code == 404, f"悬空绑定走模型不存在的 404（不是 403）: {rr.text}"

        st = client.get("/api/settings", headers=hdr).json()
        row = next(k for k in st["api_keys"] if k["id"] == cr.json()["id"])
        assert "r127plat/r127m" in row["models_missing"], f"悬空绑定要显示警告: {row}"
    finally:
        client.delete(f"/api/keys/{cr.json()['id']}", headers=hdr)

    # 老库迁移：api_keys 无 model_bindings 列 → Store 打开自动补列
    import sqlite3
    import tempfile
    from pathlib import Path
    from app.storage import Store
    tmp = Path(tempfile.mkdtemp())
    db = tmp / "old-r127.db"
    c = sqlite3.connect(db)
    c.execute("CREATE TABLE api_keys (id INTEGER PRIMARY KEY, user_id INTEGER, key TEXT, "
              "label TEXT, created_at TEXT, last_used TEXT, revoked INTEGER DEFAULT 0)")
    c.execute("INSERT INTO api_keys (user_id, key, label, created_at) VALUES (1, 'sk-old-r127', '旧key', '2026-11-01T00:00:00')")
    c.commit()
    c.close()
    s2 = Store(db)
    s2.create_user("r127old@x.local", "ph", role="user")
    old = s2.get_api_key(1)
    assert old["models"] is None, "老库迁移后旧 key 默认不限制"
    got_old = s2.user_for_api_key("sk-old-r127")
    assert got_old and got_old["_key_models"] is None, "旧 key 鉴权附带 _key_models=None"
    assert s2.update_key_models(1, 1, ["fake-gpt"])
    assert s2.get_api_key(1)["models"] == ["fake-gpt"]
    new = s2.create_api_key(1, "sk-new-r127", "new", None, ["fake-gpt", "fake/fake-gpt-2"])
    assert new["models"] == ["fake-gpt", "fake/fake-gpt-2"]
    got_new = s2.user_for_api_key("sk-new-r127")
    assert got_new and got_new["_key_models"] == ["fake-gpt", "fake/fake-gpt-2"]
    s2.close()


# ---------------------------------------------------------------- R128 绑定弹窗选项：三组分组 + 组内排序（v1.2.2）
def test_r128_binding_options_groups_sorted(client, hdr, upstream_port):
    """模型绑定选项（/api/settings）三组数据源（v1.2.2 下半场改型）：
    model_name_options=平台全名（platform/model，每模型/别名一行）、bare_model_names=裸模型名、
    aggregate_names=聚合模型，组内按首字母排序，组间不串味（全名组不含裸行）。
    背景：用户先报「配 48 个模型清单显示 93 项」（裸名/全名混排被误读），后要求恢复裸名组
    并加排序方便挑选。不回归 /v1/models（两种形式仍都列，客户端可发现裸名路由）；
    门禁语义不变：全名与裸名都是合法绑定入参。"""
    p = client.post("/api/providers", headers=hdr, json={
        "platform": "r128plat", "name": "R128(fake)", "type": "openai",
        "base_url": f"http://127.0.0.1:{upstream_port}/v1", "api_keys": ["sk-x"],
        "models": ["r128a", "r128b"], "model_map": {"r128alias": "r128a"}})
    assert p.status_code == 200, p.text
    keys = []
    try:
        # 全名与裸名都仍是合法的绑定入参（门禁精确匹配语义不变）
        r_full = client.post("/api/keys", headers=hdr, json={"label": "r128full", "models": ["r128plat/r128a"]})
        assert r_full.status_code == 200, r_full.text
        r_bare = client.post("/api/keys", headers=hdr, json={"label": "r128bare", "models": ["r128a"]})
        assert r_bare.status_code == 200, f"裸名入参必须仍合法: {r_bare.text}"
        r_open = client.post("/api/keys", headers=hdr, json={"label": "r128open"})   # 无绑定 key：/v1/models 全量目录
        assert r_open.status_code == 200, r_open.text
        keys = [r_full.json()["id"], r_bare.json()["id"], r_open.json()["id"]]

        st = client.get("/api/settings", headers=hdr).json()
        full, bare, aggs = st["model_name_options"], st["bare_model_names"], st["aggregate_names"]
        # 全名组：每个模型/别名一行 platform/model
        for expect in ("r128plat/r128a", "r128plat/r128b", "r128plat/r128alias"):
            assert expect in full, f"{expect} 要在全名组: {full}"
        assert not any(o in ("r128a", "r128b") for o in full), \
            f"全名组不得含裸行（两组串味）: {[o for o in full if o in ('r128a', 'r128b')]}"
        # 裸名组：恢复裸模型名（用户下半场要求）
        assert "r128a" in bare and "r128b" in bare, f"裸名组要含裸模型名: {bare}"
        # 组内按首字母（字典序）排序
        assert full == sorted(full), f"全名组未排序: {full}"
        assert bare == sorted(bare), f"裸名组未排序: {bare}"
        assert aggs == sorted(aggs), f"聚合组未排序: {aggs}"

        # /v1/models 保持旧行为（全量目录视角：无绑定 key）：全名与裸名都列
        m = client.get("/v1/models", headers={"authorization": f"Bearer {r_open.json()['key']}"})
        ids = {x["id"] for x in m.json()["data"]}
        assert {"r128plat/r128a", "r128a"} <= ids, f"/v1/models 两种形式都要有: {ids}"

        # /api/models 三组分组字段（v1.2.2 panel 分组渲染数据源）：组内排序，三组并集 = 全量 ids
        md = client.get("/api/models", headers=hdr).json()
        g = md["groups"]
        assert g["full"] == sorted(g["full"]) and g["bare"] == sorted(g["bare"]) and g["aggs"] == sorted(g["aggs"])
        assert "r128plat/r128a" in g["full"] and "r128a" in g["bare"]
        assert set(g["full"]) | set(g["bare"]) | set(g["aggs"]) == set(md["ids"]), "三组并集应等于全量 ids"
    finally:
        for kid in keys:
            client.delete(f"/api/keys/{kid}", headers=hdr)
        client.delete("/api/providers/r128plat", headers=hdr)

# ---------------------------------------------------------------- R129 请求日志「下游 Key」列可筛选（v1.2.3）
def test_r129_dkey_log_filter(client, hdr, upstream_port):
    """请求日志页「下游 Key」列像其他列一样带筛选箭头（v1.2.3）：
    后端 /api/stats/records 接受 dkey=<key id 逗号列表>（0 = 未记录/NULL），
    /api/stats/logs/facets 返回 dkeys 选项（{id, label}，标签解析与日志行一致：
    名称 /（无名称）/（key 已删除）/（未记录））。列表与 total 走同一筛选条件。"""
    r1 = client.post("/api/keys", headers=hdr, json={"label": "r129-key-A"}).json()
    r2 = client.post("/api/keys", headers=hdr, json={"label": "r129-key-B"}).json()
    assert r1.get("id") and r2.get("id"), (r1, r2)
    try:
        for k in (r1["key"], r2["key"]):
            rr = client.post("/v1/chat/completions",
                             headers={"authorization": f"Bearer {k}"},
                             json={"model": "fake/fake-gpt",
                                   "messages": [{"role": "user", "content": "hi"}]})
            assert rr.status_code == 200, rr.text
        # facets 带 dkeys 选项（含本次两把 key 的标签）
        fc = client.get("/api/stats/logs/facets", headers=hdr).json()
        assert "dkeys" in fc, f"facets 缺 dkeys: {list(fc.keys())}"
        by_label = {x["label"]: x["id"] for x in fc["dkeys"]}
        assert "r129-key-A" in by_label and "r129-key-B" in by_label, by_label
        # 按单把 key 筛：所有行都是它的标签，total 一致
        recs = client.get(f"/api/stats/records?limit=50&dkey={by_label['r129-key-A']}",
                          headers=hdr).json()
        assert recs["total"] == 1, f"只应有 1 条 A 的请求: {recs['total']}"
        assert all(x["dkey_label"] == "r129-key-A" for x in recs["data"])
        # 多选两把：并集
        recs2 = client.get(
            f"/api/stats/records?limit=50&dkey={by_label['r129-key-A']},{by_label['r129-key-B']}",
            headers=hdr).json()
        assert recs2["total"] == 2, recs2["total"]
        assert {x["dkey_label"] for x in recs2["data"]} == {"r129-key-A", "r129-key-B"}
        # 0 = 未记录（NULL dkey_id）：行标签恒为空串
        recs0 = client.get("/api/stats/records?limit=10&dkey=0", headers=hdr).json()
        assert all(x["dkey_label"] == "" for x in recs0["data"]), \
            "dkey=0 只能命中未记录（dkey_id 为 NULL）的行"
        # 与现有筛选可叠加（平台 + 下游 key）
        recs3 = client.get(
            f"/api/stats/records?limit=50&platform=fake&dkey={by_label['r129-key-B']}",
            headers=hdr).json()
        assert recs3["total"] == 1
        assert all(x["dkey_label"] == "r129-key-B" for x in recs3["data"])
    finally:
        client.delete(f"/api/keys/{r1['id']}", headers=hdr)
        client.delete(f"/api/keys/{r2['id']}", headers=hdr)


# ---------------------------------------------------------------- R130 设置页「全部账号（汇总）」列出全部用户的下游 key（v1.2.3）
def test_r130_settings_aggregate_keys(client, hdr):
    """管理员选「全部账号（汇总）」(?tenant=all) 或多选时，/api/settings 的「我的 API Key」
    panel 列出全部用户的下游 key（bug 修复：此前只显示管理员自己的 key）：
    行带 owner_email/self，key 打掩码（明文不泄露），别人的行前端只读；
    单账号视角 (?tenant=<id>) 与缺省（只看我的数据）行为不变。"""
    from app import auth as auth_mod
    from app.main import store
    dave = store.create_user("dave130@test.local", auth_mod.hash_password("dave-pw"))
    dk = store.create_api_key(dave["id"], auth_mod.new_api_key(), label="dave-key")
    erin = store.create_user("erin130@test.local", auth_mod.hash_password("erin-pw"))
    try:
        # all：跨账号罗列
        st = client.get("/api/settings?tenant=all", headers=hdr).json()
        assert st["keys_scope"] == "all"
        labels = {k["label"] for k in st["api_keys"]}
        assert "dave-key" in labels and "测试" in labels, \
            f"汇总视角要同时列出管理员与他人的 key: {labels}"
        mine = [k for k in st["api_keys"] if k["self"]]
        others = [k for k in st["api_keys"] if not k["self"]]
        assert mine and others, "应同时包含自己与他人的行"
        drow = [k for k in st["api_keys"] if k["label"] == "dave-key"][0]
        assert drow["owner_email"] == "dave130@test.local" and not drow["self"]
        # 掩码：明文 key 不得出现在响应里
        blob = str(st["api_keys"])
        assert dk["key"] not in blob, "汇总视角泄露了明文 key"
        assert all("…" in k["key"] for k in st["api_keys"]), "key 必须是掩码形式"
        # 多选：只列勾选账号的 key
        st2 = client.get(f"/api/settings?tenant={dave['id']},{erin['id']}", headers=hdr).json()
        assert st2["keys_scope"] == "all"
        assert {k.get("owner_email") for k in st2["api_keys"]} == {"dave130@test.local"}, \
            f"只应列 dave（erin 还没建 key）: {st2['api_keys']}"
        # 单账号视角：不变（ro，只看该账号）
        st3 = client.get(f"/api/settings?tenant={dave['id']}", headers=hdr).json()
        assert st3["keys_scope"] == "self" and st3["viewing_user"]["email"] == "dave130@test.local"
        assert all("owner_email" not in k for k in st3["api_keys"])
        # 缺省（只看我的数据）：不变
        st4 = client.get("/api/settings", headers=hdr).json()
        assert st4["keys_scope"] == "self" and st4["viewing_user"] is None
        assert all(k["label"] != "dave-key" for k in st4["api_keys"]), "自己视角不得看到他人 key"
    finally:
        store.delete_user_full(dave["id"])
        store.delete_user_full(erin["id"])


# ---------------------------------------------------------------- R131 探测生成类模型判「gen」保留（v2.0.0）
def test_r131_probe_gen_models_kept(client, hdr, upstream_port):
    """图像/视频/音频类模型在 chat 端点 4xx + 专用端点提示：探测判 gen 保留（不写回删掉），
    真 404 仍移除；无 removed 的平台不写回、模型清单原样。"""
    r1 = client.post("/api/providers", headers=hdr, json={
        "platform": "r131mix", "name": "R131 mix(fake)", "type": "openai",
        "base_url": f"http://127.0.0.1:{upstream_port}/v1", "api_keys": ["pk"],
        "models": ["m-ok-plain", "m-genimg", "m-genvid", "m-x404"]})
    assert r1.status_code == 200, r1.text
    r2 = client.post("/api/providers", headers=hdr, json={
        "platform": "r131gen", "name": "R131 gen-only(fake)", "type": "openai",
        "base_url": f"http://127.0.0.1:{upstream_port}/v1", "api_keys": ["pk"],
        "models": ["m-ok-plain", "m-genimg", "m-400sse"]})
    assert r2.status_code == 200, r2.text
    try:
        # 混合平台：2 个 gen 保留 + 1 个 x404 移除 → 写回时去掉 x404、保留 gen
        rp = client.post("/api/providers/r131mix/probe", headers=hdr, json={})
        assert rp.status_code == 200, rp.text
        j = rp.json()
        assert sorted(x["model"] for x in j["gen"]) == ["m-genimg", "m-genvid"], j
        assert [x["model"] for x in j["removed"]] == ["m-x404"], j
        assert sorted(x for x in j["kept"]) == ["m-genimg", "m-genvid", "m-ok-plain"], j
        assert j["written"] is True, "有 removed 应写回模型清单"
        assert any("/v1/images" in x["reason"] for x in j["gen"]), j
        models_now = [p for p in client.get("/api/providers", headers=hdr).json()["data"]
                      if p["platform"] == "r131mix"][0]["models"]
        assert models_now == ["m-ok-plain", "m-genimg", "m-genvid"], \
            f"生成类模型必须保留在清单里: {models_now}"
        # 仅 gen 平台：没有 removed → 不写回，清单原样
        rp2 = client.post("/api/providers/r131gen/probe", headers=hdr, json={})
        j2 = rp2.json()
        assert sorted(x["model"] for x in j2["gen"]) == ["m-400sse", "m-genimg"], j2
        assert not j2["removed"] and j2["written"] is False, j2
        models2 = [p for p in client.get("/api/providers", headers=hdr).json()["data"]
                   if p["platform"] == "r131gen"][0]["models"]
        assert models2 == ["m-ok-plain", "m-genimg", "m-400sse"], \
            f"无 removed 时清单必须原样: {models2}"
    finally:
        client.delete("/api/providers/r131mix", headers=hdr)
        client.delete("/api/providers/r131gen", headers=hdr)


# ---------------------------------------------------------------- R132 /v1/images/generations 透传（v2.0.0）
def test_r132_images_generations_passthrough(client, hdr, upstream_port):
    """POST /v1/images/generations → 上游 /v1/images/generations 透传：2xx 产物 URL 抓进日志
    output_info（images=N）；上游 400 记 error 行并透传错误；未知模型 404；
    anthropic 上游 400 unsupported；下游 key 绑定门禁对生成类端点同样生效。"""
    import json as _json
    from tests import fake_upstream
    r1 = client.post("/api/providers", headers=hdr, json={
        "platform": "r132img", "name": "R132 images(fake)", "type": "openai",
        "base_url": f"http://127.0.0.1:{upstream_port}/v1", "api_keys": ["pk"],
        "models": ["img-model", "m-genbad"]})
    assert r1.status_code == 200, r1.text
    bh = {"authorization": f"Bearer {conftest._TEST_API_KEY}"}
    kid = None
    try:
        store = client.app.state.store
        # 正常透传：n=2 → 2 个 URL，日志 output_info 记产物
        ri = client.post("/v1/images/generations", headers=bh,
                         json={"model": "img-model", "prompt": "a cat", "n": 2})
        assert ri.status_code == 200, ri.text
        data = ri.json()["data"]
        assert [d["url"] for d in data] == ["https://fake.example/gen-img-model-0.png",
                                            "https://fake.example/gen-img-model-1.png"], data
        rows = [x for x in store.recent(5)
                if x.get("model") == "img-model" and x.get("status") == "ok"]
        assert rows, "图像透传应落请求日志"
        row = rows[0]
        oi = _json.loads(row["output_info"])
        assert oi["images"] == 2 and len(oi["urls"]) == 2, row
        assert row["total_tokens"] == 0 and row["estimated"], \
            "生成类请求 0 token 且标 estimated"
        # 上游 400 → 透传错误（400）+ 日志记 error 行（含上游原文）
        fake_upstream.RECEIVED.clear()
        rb = client.post("/v1/images/generations", headers=bh, json={"model": "m-genbad"})
        assert rb.status_code == 400, rb.text
        erow = [x for x in store.recent(5)
                if x.get("model") == "m-genbad" and x.get("status") == "error"]
        assert erow and "not enabled" in erow[0]["error"], erow
        # 未知模型 → 404
        ru = client.post("/v1/images/generations", headers=bh, json={"model": "no-such-model"})
        assert ru.status_code == 404, ru.text
        # anthropic 上游 → 400 unsupported
        ra = client.post("/v1/images/generations", headers=bh,
                         json={"model": "fakeclaude/fake-claude"})
        assert ra.status_code == 400 and "只支持 OpenAI 兼容" in ra.text, ra.text
        # 绑定门禁：只绑 img-model 的 key 调 m-genbad → 403，绑定的 img-model 放行
        cr = client.post("/api/keys", headers=hdr,
                         json={"label": "r132bound", "models": ["r132img/img-model"]})
        assert cr.status_code == 200, cr.text
        kid = cr.json()["id"]
        bkey = {"authorization": f"Bearer {cr.json()['key']}"}
        rg = client.post("/v1/images/generations", headers=bkey, json={"model": "m-genbad"})
        assert rg.status_code == 403, rg.text
        ro = client.post("/v1/images/generations", headers=bkey,
                         json={"model": "r132img/img-model"})
        assert ro.status_code == 200, ro.text
        # 转发到上游的 model 字段 = 客户端模型名（无别名时同名透传）；
        # 取最后一条（绑定 key 成功调用的 r132img/img-model → 上游名 img-model）
        sent = [x for x in fake_upstream.RECEIVED if x["path"] == "/v1/images/generations"]
        assert sent and sent[-1]["body"]["model"] == "img-model", sent[-1]
    finally:
        if kid is not None:
            client.app.state.store.revoke_api_key(me_id(client), kid)
        client.delete("/api/providers/r132img", headers=hdr)


# ---------------------------------------------------------------- R133 /v1/videos + /v1/audio/speech 透传（v2.0.0）
def test_r133_videos_and_speech_passthrough(client, hdr, upstream_port):
    """videos：任务型 JSON 原样透传，任务 id 抓进 output_info；
    audio/speech：二进制原样透传（content-type + 字节），字节数抓进 output_info。"""
    import json as _json
    r1 = client.post("/api/providers", headers=hdr, json={
        "platform": "r133av", "name": "R133 av(fake)", "type": "openai",
        "base_url": f"http://127.0.0.1:{upstream_port}/v1", "api_keys": ["pk"],
        "models": ["vid-model", "tts-model"]})
    assert r1.status_code == 200, r1.text
    bh = {"authorization": f"Bearer {conftest._TEST_API_KEY}"}
    try:
        store = client.app.state.store
        # videos：任务型 JSON
        rv = client.post("/v1/videos", headers=bh, json={"model": "vid-model", "prompt": "a dog"})
        assert rv.status_code == 200, rv.text
        jv = rv.json()
        assert jv["id"] == "vid-vid-model" and jv["status"] == "queued", jv
        vrow = [x for x in store.recent(3)
                if x.get("model") == "vid-model" and x.get("status") == "ok"][0]
        assert "vid-vid-model" in vrow["output_info"], vrow
        # audio speech：二进制透传
        speech = b"ID3\x04fake-mp3-audio"
        rs = client.post("/v1/audio/speech", headers=bh,
                         json={"model": "tts-model", "input": "hi"})
        assert rs.status_code == 200, rs.text
        assert rs.content == speech, "二进制必须原样透传"
        assert "audio/mpeg" in (rs.headers.get("content-type") or "")
        srow = [x for x in store.recent(3)
                if x.get("model") == "tts-model" and x.get("status") == "ok"][0]
        oi = _json.loads(srow["output_info"])
        assert oi["audio_bytes"] == len(speech) and \
            oi["content_type"].startswith("audio/"), srow
    finally:
        client.delete("/api/providers/r133av", headers=hdr)


# ---------------------------------------------------------------- R134 /v1/audio/transcriptions multipart 透传（v2.0.0）
def test_r134_audio_transcriptions_multipart(client, hdr, upstream_port):
    """multipart 音频转写：file 二进制原样透传（不重编码），form 的 model 字段按别名做
    字节级替换（客户端别名 → 上游模型名）；转写文本抓进 output_info；缺 model → 400。"""
    import json as _json
    from tests import fake_upstream
    r1 = client.post("/api/providers", headers=hdr, json={
        "platform": "r134tr", "name": "R134 trans(fake)", "type": "openai",
        "base_url": f"http://127.0.0.1:{upstream_port}/v1", "api_keys": ["pk"],
        "models": ["trans-model"], "model_map": {"trans-alias": "trans-model"}})
    assert r1.status_code == 200, r1.text
    bh = {"authorization": f"Bearer {conftest._TEST_API_KEY}"}
    try:
        store = client.app.state.store
        audio = b"RIFF0000fakeaudiobytes"
        # 直呼上游模型名
        rt = client.post("/v1/audio/transcriptions", headers=bh,
                         data={"model": "trans-model", "language": "zh"},
                         files={"file": ("a.wav", audio, "audio/wav")})
        assert rt.status_code == 200, rt.text
        assert rt.json()["text"] == "hello from fake transcription"
        trow = [x for x in store.recent(3)
                if x.get("model") == "trans-model" and x.get("status") == "ok"][0]
        assert "hello from fake transcription" in trow["output_info"], trow
        # 别名：客户端发 trans-alias → 转发到上游的 form model 字段须是 trans-model
        fake_upstream.RECEIVED.clear()
        ra = client.post("/v1/audio/transcriptions", headers=bh,
                         data={"model": "trans-alias", "language": "zh"},
                         files={"file": ("a.wav", audio, "audio/wav")})
        assert ra.status_code == 200, ra.text
        sent = [x for x in fake_upstream.RECEIVED if x["path"] == "/v1/audio/transcriptions"]
        assert sent and sent[0]["body"]["model"] == "trans-model", sent[0]
        assert sent[0]["body"]["file_bytes"] == len(audio), "file 字段字节数必须完整"
        # 缺 model 字段 → 400
        fake_upstream.RECEIVED.clear()
        rm = client.post("/v1/audio/transcriptions", headers=bh,
                         data={"language": "zh"},
                         files={"file": ("a.wav", audio, "audio/wav")})
        assert rm.status_code == 400, rm.text
        assert not [x for x in fake_upstream.RECEIVED
                    if x["path"] == "/v1/audio/transcriptions"], "缺 model 不应打到上游"
        # 未知模型 → 404
        rn = client.post("/v1/audio/transcriptions", headers=bh,
                         data={"model": "no-such"},
                         files={"file": ("a.wav", audio, "audio/wav")})
        assert rn.status_code == 404, rn.text
    finally:
        client.delete("/api/providers/r134tr", headers=hdr)


# ---------------------------------------------------------------- R135 model_kind 路由 + 扁平路径（v2.0.0）
def test_r135_model_kind_routing(client, hdr, upstream_port):
    """extra.model_kind 路由：类型与端点不匹配 → 400 指路（严格模式）；未配置 → 放行。
    扁平 /v1/images、/v1/audio（JSON=TTS，multipart=转写）；深路径别名照旧；
    生成类模型误入 chat 端点 → 400 指路，不影响普通聊天模型。"""
    r1 = client.post("/api/providers", headers=hdr, json={
        "platform": "r135gen", "name": "R135 kinds(fake)", "type": "openai",
        "base_url": f"http://127.0.0.1:{upstream_port}/v1", "api_keys": ["pk"],
        "models": ["img-model", "vid-model", "tts-model", "trans-model", "plain-model"],
        "extra": {"model_kind": {"img-model": "image", "vid-model": "video",
                                  "tts-model": "audio_tts", "trans-model": "audio_trans"}}})
    assert r1.status_code == 200, r1.text
    bh = {"authorization": f"Bearer {conftest._TEST_API_KEY}"}
    try:
        # 扁平 /v1/images（深路径 /v1/images/generations 同 handler）
        ri = client.post("/v1/images", headers=bh,
                         json={"model": "r135gen/img-model", "n": 1})
        assert ri.status_code == 200, ri.text
        assert ri.json()["data"][0]["url"].startswith("https://fake.example/gen-img-model"), ri.text
        rd = client.post("/v1/images/generations", headers=bh,
                         json={"model": "r135gen/img-model"})
        assert rd.status_code == 200, rd.text
        # 类型不匹配：图像模型调 /v1/videos → 400 指路
        rv = client.post("/v1/videos", headers=bh, json={"model": "r135gen/img-model"})
        assert rv.status_code == 400 and "/v1/images" in rv.text, rv.text
        # 未配 kind → 放行（照转上游）
        rp = client.post("/v1/videos", headers=bh, json={"model": "plain-model"})
        assert rp.status_code == 200, rp.text
        # /v1/audio：JSON → TTS
        rs = client.post("/v1/audio", headers=bh, json={"model": "tts-model", "input": "hi"})
        assert rs.status_code == 200, rs.text
        assert "audio/mpeg" in (rs.headers.get("content-type") or ""), rs.headers
        # /v1/audio：multipart → 转写
        rt = client.post("/v1/audio", headers=bh,
                         data={"model": "trans-model"},
                         files={"file": ("a.wav", b"RIFFfake", "audio/wav")})
        assert rt.status_code == 200, rt.text
        assert rt.json()["text"] == "hello from fake transcription"
        # TTS 模型带文件调 → 400 指路（multipart 判为转写端点）
        ra = client.post("/v1/audio", headers=bh,
                         data={"model": "tts-model", "language": "zh"},
                         files={"file": ("a.wav", b"RIFFfake", "audio/wav")})
        assert ra.status_code == 400, ra.text
        # 生成类模型误入 chat → 400 指路
        rc = client.post("/v1/chat/completions", headers=bh, json={
            "model": "r135gen/img-model", "messages": [{"role": "user", "content": "hi"}]})
        assert rc.status_code == 400 and "/v1/images" in rc.text, rc.text
        # 普通聊天模型不受影响
        rk = client.post("/v1/chat/completions", headers=bh, json={
            "model": "plain-model", "messages": [{"role": "user", "content": "hi"}]})
        assert rk.status_code == 200, rk.text
    finally:
        client.delete("/api/providers/r135gen", headers=hdr)


# ---------------------------------------------------------------- R136 探测自动回填 model_kind（v2.0.0）
def test_r136_probe_autofill_model_kind(client, hdr, upstream_port):
    """探测生成类签名明确指向端点时自动回填 extra.model_kind（不依赖有没有 removed），
    回填后模型立即可用扁平端点调用；模糊签名（仅「不支持SSE」）不自动回填。"""
    r1 = client.post("/api/providers", headers=hdr, json={
        "platform": "r136probe", "name": "R136 probe(fake)", "type": "openai",
        "base_url": f"http://127.0.0.1:{upstream_port}/v1", "api_keys": ["pk"],
        "models": ["m-ok-plain", "m-genimg", "m-genvid", "m-400sse"]})
    assert r1.status_code == 200, r1.text
    bh = {"authorization": f"Bearer {conftest._TEST_API_KEY}"}
    try:
        rp = client.post("/api/providers/r136probe/probe", headers=hdr, json={})
        assert rp.status_code == 200, rp.text
        j = rp.json()
        kinds = {x["model"]: x.get("kind") for x in j["gen"]}
        assert kinds.get("m-genimg") == "image", j["gen"]
        assert kinds.get("m-genvid") == "video", j["gen"]
        assert not kinds.get("m-400sse"), "模糊签名（不支持SSE）不应自动回填"
        assert sorted(j["gen_kind_set"]) == ["m-genimg", "m-genvid"], j
        # 回填已持久化（零 removed 也写）
        p = [x for x in client.get("/api/providers", headers=hdr).json()["data"]
             if x["platform"] == "r136probe"][0]
        assert (p.get("extra") or {}).get("model_kind") == \
            {"m-genimg": "image", "m-genvid": "video"}, p.get("extra")
        # 回填后直接调扁平端点
        ri = client.post("/v1/images", headers=bh, json={"model": "m-genimg", "n": 1})
        assert ri.status_code == 200, ri.text
        assert ri.json()["data"][0]["url"].startswith(
            "https://fake.example/gen-m-genimg"), ri.text
        # 未回填的模糊签名模型：chat 端点照旧透传上游 400（不是 kind 拦截）
        rc = client.post("/v1/chat/completions", headers=bh, json={
            "model": "m-400sse", "messages": [{"role": "user", "content": "hi"}]})
        assert rc.status_code == 400, rc.text
        assert "model_kind" not in rc.text and "请改用" not in rc.text, \
            "模糊签名模型未配 kind，chat 端点不应本地拦截"
    finally:
        client.delete("/api/providers/r136probe", headers=hdr)

# ---------------------------------------------------------------- R137 GBK 中文 body 抢救（v2.0.0）
def test_r137_gbk_body_rescued(client, hdr, upstream_port):
    """中文 Windows 命令行 curl 会把中文 body 按 GBK 发出（非法 JSON UTF-8）：
    relay 抢救为 UTF-8 后照常透传，上游收到的是正确中文与替换后的模型名。"""
    import json as _json
    from tests import fake_upstream
    r1 = client.post("/api/providers", headers=hdr, json={
        "platform": "r137gbk", "name": "R137 gbk(fake)", "type": "openai",
        "base_url": f"http://127.0.0.1:{upstream_port}/v1", "api_keys": ["pk"],
        "models": ["img-model"]})
    assert r1.status_code == 200, r1.text
    bh = {"authorization": f"Bearer {conftest._TEST_API_KEY}"}
    try:
        body = _json.dumps(
            {"model": "img-model", "prompt": "一只可爱的小猫咪", "size": "512x512"},
            ensure_ascii=False).encode("gbk")  # 模拟中文 Windows curl 的出站字节
        fake_upstream.RECEIVED.clear()
        ri = client.post("/v1/images/generations",
                         headers={**bh, "content-type": "application/json"},
                         content=body)
        assert ri.status_code == 200, ri.text
        assert ri.json()["data"][0]["url"].startswith(
            "https://fake.example/gen-img-model"), ri.text
        got = [x for x in fake_upstream.RECEIVED if x["path"] == "/v1/images/generations"]
        assert got, "上游应收到抢救后的请求"
        assert got[0]["body"].get("prompt") == "一只可爱的小猫咪", got[0]["body"]
    finally:
        client.delete("/api/providers/r137gbk", headers=hdr)


# ---------------------------------------------------------------- R138 生成类本地 4xx 也落日志（v2.0.0）
def test_r138_local_4xx_logged(client, hdr, upstream_port):
    """生成类端点本地校验失败（非法 JSON / 找不到模型）也记请求日志 error 行，
    不再出现「调用没记录」的黑洞。"""
    r1 = client.post("/api/providers", headers=hdr, json={
        "platform": "r138loc", "name": "R138 local(fake)", "type": "openai",
        "base_url": f"http://127.0.0.1:{upstream_port}/v1", "api_keys": ["pk"],
        "models": ["img-model"]})
    assert r1.status_code == 200, r1.text
    bh = {"authorization": f"Bearer {conftest._TEST_API_KEY}"}
    store = client.app.state.store
    try:
        # 彻底非法 JSON（GBK 也救不回来）→ 400 + error 行
        rb = client.post("/v1/images/generations",
                         headers={**bh, "content-type": "application/json"},
                         content=b"\x00\x01\x02not json")
        assert rb.status_code == 400 and "invalid_json" in rb.text, rb.text
        erow = [x for x in store.recent(5)
                if x.get("status") == "error" and "不是合法 JSON" in (x.get("error") or "")]
        assert erow and erow[0].get("http_status") == 400, erow
        # 找不到模型 → 404 + error 行
        ru = client.post("/v1/images/generations", headers=bh,
                         json={"model": "no-such-model-x"})
        assert ru.status_code == 404, ru.text
        erow = [x for x in store.recent(5)
                if x.get("model") == "no-such-model-x" and x.get("status") == "error"]
        assert erow, "model_not_found 应落 error 日志行"
        assert "没有找到模型" in erow[0].get("error"), erow[0]
    finally:
        client.delete("/api/providers/r138loc", headers=hdr)


# ---------------------------------------------------------------- R139 GET /v1/videos 任务状态查询（v2.0.0）
def test_r139_video_status_get_passthrough(client, hdr, upstream_port):
    """GET /v1/videos?model_name=..&video_id=.. 透传上游查询：
    别名替换为上游模型名；query 原样转发；extra.videos_status_path 完整 URL 覆盖；
    缺 model 400 / 未知模型 404 / kind 不匹配 400（均落 error 行）；成功 GET 不落日志（防轮询刷屏）。"""
    from tests import fake_upstream
    r1 = client.post("/api/providers", headers=hdr, json={
        "platform": "r139vid", "name": "R139 video(fake)", "type": "openai",
        "base_url": f"http://127.0.0.1:{upstream_port}/v1", "api_keys": ["pk"],
        "models": ["vid-model", "img-model"],
        "extra": {"model_kind": {"img-model": "image"}}})
    assert r1.status_code == 200, r1.text
    r2 = client.post("/api/providers", headers=hdr, json={
        "platform": "r139vid2", "name": "R139 video url-override(fake)", "type": "openai",
        "base_url": f"http://127.0.0.1:{upstream_port}/v1", "api_keys": ["pk"],
        "models": ["vid2-model"],
        "extra": {"videos_status_path":
                  f"http://127.0.0.1:{upstream_port}/custom_video_status"}})
    assert r2.status_code == 200, r2.text
    bh = {"authorization": f"Bearer {conftest._TEST_API_KEY}"}
    store = client.app.state.store
    try:
        # 默认路径（未覆盖）：GET → 假上游 GET /v1/videos，别名替换 + video_id 透传
        fake_upstream.RECEIVED.clear()
        rs = client.get("/v1/videos", params={"model_name": "vid-model", "video_id": "t-1"},
                        headers=bh)
        assert rs.status_code == 200, rs.text
        assert rs.json()["video_id"] == "t-1" and rs.json()["status"] == "completed", rs.text
        got = [x for x in fake_upstream.RECEIVED if x["path"] == "/v1/videos[GET]"]
        assert got and got[0]["body"].get("video_id") == "t-1", got
        assert got[0]["body"].get("model_name") == "vid-model", got
        # 完整 URL 覆盖（模拟 Agnes：查询地址与创建不同前缀）
        fake_upstream.RECEIVED.clear()
        rc = client.get("/v1/videos", params={"model": "r139vid2/vid2-model", "video_id": "t-2"},
                        headers=bh)
        assert rc.status_code == 200, rc.text
        got = [x for x in fake_upstream.RECEIVED if x["path"] == "/custom_video_status[GET]"]
        assert got and got[0]["body"].get("video_id") == "t-2", got
        assert got[0]["body"].get("model") == "vid2-model", got  # model 参数也被替换
        # 成功 GET 不落请求日志（轮询防刷屏）
        rows = [x for x in store.recent(5)
                if x.get("model") in ("vid-model", "vid2-model")
                and x.get("in_format") == "openai"]
        assert not rows, "GET 状态查询成功不应写请求日志"
        # 缺 model → 400 + error 行
        rm = client.get("/v1/videos", params={"video_id": "t-3"}, headers=bh)
        assert rm.status_code == 400 and "missing_model" in rm.text, rm.text
        erow = [x for x in store.recent(5)
                if "缺少 model 参数" in (x.get("error") or "")]
        assert erow, "缺 model 400 应落 error 行"
        # 未知模型 → 404
        ru = client.get("/v1/videos", params={"model_name": "no-such-v", "video_id": "t-4"},
                        headers=bh)
        assert ru.status_code == 404 and "model_not_found" in ru.text, ru.text
        # kind 不匹配：image 模型来查视频 → 400 指路
        rk = client.get("/v1/videos", params={"model_name": "img-model", "video_id": "t-5"},
                        headers=bh)
        assert rk.status_code == 400 and "model_kind_mismatch" in rk.text, rk.text
    finally:
        client.delete("/api/providers/r139vid", headers=hdr)
        client.delete("/api/providers/r139vid2", headers=hdr)


# ---------------------------------------------------------------- R140 日志筛选 facets：类型/账号选项不为空（v2.0.1）
def test_r140_log_facets_type_and_account(client, hdr):
    """生产 bug 回归：日志页「类型」「账号」筛选框一个选项都没有。
    根因①：stream 列是 INTEGER，facet 只认字符串 "0"/"1" → 类型选项恒空；
    根因②：单账号视图（管理员默认 self，int user_id）账号映射没有分支 →
    accounts 全空 email，前端 filter(Boolean) 全滤掉（R140）。"""
    from app.main import store
    uid = client.get("/api/settings", headers=hdr).json()["user"]["id"]
    me_email = next(u["email"] for u in store.list_users() if u["id"] == uid)
    store.record(platform="r140p", model="r140-a", stream=1, total_tokens=10, user_id=uid)
    store.record(platform="r140p", model="r140-b", stream=0, total_tokens=10, user_id=uid)

    # 默认不带 tenant = 单账号视图（int user_id）：类型/账号选项不得为空
    f = client.get("/api/stats/logs/facets", headers=hdr).json()
    assert 0 in f["stream"] and 1 in f["stream"], f"类型 facet 为空（int/str 错位？）: {f['stream']}"
    assert [a["email"] for a in f["accounts"]] == [me_email], \
        f"单账号视图账号 facet 无映射: {f['accounts']}"

    # 类型筛选按 int 生效，总数与列表一致
    r1 = client.get("/api/stats/records", params={"platform": "r140p", "stream": "1"}, headers=hdr).json()
    assert r1["total"] == 1 and r1["data"][0]["model"] == "r140-a", r1
    r0 = client.get("/api/stats/records", params={"platform": "r140p", "stream": "0"}, headers=hdr).json()
    assert r0["total"] == 1 and r0["data"][0]["model"] == "r140-b", r0


# ---------------------------------------------------------------- R141 日志 facets 空平台/模型 → 「（未解析）」选项（v2.0.1）
def test_r141_log_facets_unresolved_placeholder(client, hdr):
    """生产 bug 回归：「平台」筛选框出现一个什么都没有的空白筛选项。
    根因：本地审计行（401/403/生成类本地 4xx）落日志时还没解析到平台，DB 存空串，
    facet 默认值也是空串 → 渲染出空白 checkbox（R141）。现在统一映射「（未解析）」
    占位，筛选侧转回空串去匹配，勾选该选项能真正筛出这些审计行。"""
    from app.main import store
    uid = client.get("/api/settings", headers=hdr).json()["user"]["id"]
    store.record(platform="", model="", status="error", http_status=401,
                 error="API Key 无效（R141 占位）", user_id=uid)
    store.record(platform="", model="r141-ghost", status="error", http_status=404,
                 error="没有找到模型（R141 占位）", user_id=uid)

    f = client.get("/api/stats/logs/facets", headers=hdr).json()
    assert "" not in f["platform"] and "" not in f["model"], \
        f"platform/model facet 仍有空白选项: {f['platform']!r} / {f['model']!r}"
    assert "（未解析）" in f["platform"] and "（未解析）" in f["model"], f

    # 勾选「（未解析）」→ 占位转回空串，匹配 platform 为空的审计行
    r = client.get("/api/stats/records", params={"platform": "（未解析）"}, headers=hdr).json()
    assert r["total"] >= 2 and all(x["platform"] == "" for x in r["data"]), r
    # 组合筛选：占位 + 具体 model 精确定位
    rg = client.get("/api/stats/records",
                    params={"platform": "（未解析）", "model": "r141-ghost"}, headers=hdr).json()
    assert rg["total"] == 1 and rg["data"][0]["model"] == "r141-ghost", rg


# ---------------------------------------------------------------- R142 日志 facets：「-」选项可匹配 + 已删除账号标签（v2.0.1）
def test_r142_log_facets_dash_and_deleted_user(client, hdr):
    """同类两个问题（R142）：
    ① key/IP facet 把空值映射成「-」占位，但此前勾选后按字面「-」发后端，
      DB 存的是空串 → 恒零行；现在「-」转回空串匹配；
    ② 账号被删（历史行保留作审计）后，行里 user_email 渲染成空白格、
      账号 facet 也没有对应选项；现在行与 facet 选项统一「（用户已删除 #id）」标签。"""
    from app.main import store
    uid = client.get("/api/settings", headers=hdr).json()["user"]["id"]
    store.record(platform="r142p", model="r142-m", total_tokens=5,
                 user_id=uid, key_hint="", client_ip="")
    f = client.get("/api/stats/logs/facets", headers=hdr).json()
    assert "-" in f["key_hint"] and "-" in f["client_ip"], f

    r = client.get("/api/stats/records", params={"key": "-", "ip": "-"}, headers=hdr).json()
    assert r["total"] >= 1 and all(x["key_hint"] == "" and x["client_ip"] == ""
                                   for x in r["data"]), r

    # 删除账号：facet 选项与行展示同标签
    ur = client.post("/api/users", headers=hdr, json={
        "email": "r142-del@test.local", "password": "agm-password", "role": "user"})
    assert ur.status_code == 200, ur.text
    dud = ur.json()["user"]["id"]
    store.record(platform="r142p", model="r142-d", total_tokens=5, user_id=dud)
    dr = client.delete(f"/api/users/{dud}", headers=hdr)
    assert dr.status_code == 200, dr.text

    lab = f"（用户已删除 #{dud}）"
    f2 = client.get("/api/stats/logs/facets?tenant=all", headers=hdr).json()
    assert lab in [a["email"] for a in f2["accounts"]], f2["accounts"]
    r3 = client.get("/api/stats/records?tenant=all&users=%d" % dud, headers=hdr).json()
    assert r3["total"] == 1 and r3["data"][0]["user_email"] == lab, r3


# ---------------------------------------------------------------- R143 日志 facets 跟随时间范围与筛选，不再全量（v2.0.1）
def test_r143_log_facets_follow_range_and_filters(client, hdr):
    """筛选项不再全量（R143）：按「当前时间范围 + 其他已生效筛选」罗列，每个维度排除
    自身筛选（勾了值仍可切换其他值）。修复生产体验「选项在、勾选后 0 条」
    （如「（未解析）」行只在 11-15，看近 15 分钟时全量 facet 仍列出该死选项）。"""
    from datetime import datetime, timedelta
    from app.main import store
    uid = client.get("/api/settings", headers=hdr).json()["user"]["id"]
    now = datetime.now().astimezone()
    old = (now - timedelta(days=9)).isoformat(timespec="seconds")
    store.record(ts=old, platform="r143old", model="r143-m1", stream=0,
                 total_tokens=10, user_id=uid)
    store.record(platform="r143new", model="r143-m2", stream=1, total_tokens=10, user_id=uid)

    # 不带范围（该账号全量）：新旧平台都在选项里
    f0 = client.get("/api/stats/logs/facets", headers=hdr).json()
    assert "r143old" in f0["platform"] and "r143new" in f0["platform"], f0["platform"]
    # 当日范围：旧数据选项消失（死选项问题修复），total 跟着结果集收窄
    today = now.strftime("%Y-%m-%d")
    f1 = client.get("/api/stats/logs/facets", params={"from": today, "to": today}, headers=hdr).json()
    assert "r143new" in f1["platform"] and "r143old" not in f1["platform"], f1["platform"]
    # 跨维度筛选：platform=r143new 之下，模型/类型选项只剩该行的取值
    f2 = client.get("/api/stats/logs/facets",
                    params={"from": today, "to": today, "platform": "r143new"}, headers=hdr).json()
    assert f2["model"] == ["r143-m2"], f2["model"]
    assert f2["stream"] == [1], f2["stream"]
    # 自身排除：勾了 platform=r143new，平台选项里它仍然在（可切换到别的值）
    assert "r143new" in f2["platform"], f2["platform"]
    assert f2["total"] == 1, f2["total"]


# ---------------------------------------------------------------- R144 平台设置删掉模型后，聚合模型不再保留其配置（v2.0.1）
def test_r144_aggregate_prune_stale_entries(client, hdr, upstream_port):
    """聚合陈旧条目清理（R144）：Bug 为上游平台设置里去掉模型后，聚合模型仍保留
    已删模型的配置（resolve 透传不校验模型是否登记，死条目写进配置层）。现语义：
    1) 平台模型清单变更删掉模型 → 指向它的聚合条目自动清理，聚合丢空则整个删除；
    2) 平台删除 → 其全部「平台/*」聚合条目清理；
    3) 聚合保存严格校验（未登记模型 → 400）；model_map 别名不受清理影响；
    4) 绕过 API 校验的陈旧条目（直写 registry / 手改 JSON）API 标记不可用，
       并在下一次模型清单变更时清理。"""
    from app.main import store, tenants
    reg = tenants.get(store.first_admin()["id"])

    P = "aggr144"
    base = {"platform": P, "type": "openai",
            "base_url": f"http://127.0.0.1:{upstream_port}/v1",
            "api_keys": ["pk144"]}

    def names():
        return [a["name"] for a in client.get("/api/aggregates", headers=hdr).json()["data"]]

    def entries(name):
        for a in client.get("/api/aggregates", headers=hdr).json()["data"]:
            if a["name"] == name:
                return [e["entry"] for e in a["entries"]]
        return None

    def agg(name):
        for a in client.get("/api/aggregates", headers=hdr).json()["data"]:
            if a["name"] == name:
                return a
        return None

    try:
        # 两个登记模型的 fake 平台（模型清单在配置层，无需真实可达）
        assert client.post("/api/providers", headers=hdr,
                           json={**base, "models": ["alpha", "beta"]}).status_code == 200
        # 1) 指向登记模型的聚合条目照常可存
        assert client.post("/api/aggregates", headers=hdr,
                          json={"name": "A144", "entries": [f"{P}/alpha", f"{P}/beta"]}).status_code == 200
        assert entries("A144") == [f"{P}/alpha", f"{P}/beta"]

        # 2) 改模型清单去掉 beta → 指向 beta 的聚合条目自动清理
        r = client.post("/api/providers", headers=hdr,
                       json={**base, "models": ["alpha"]})
        assert r.status_code == 200, r.text
        assert entries("A144") == [f"{P}/alpha"], entries("A144")

        # 3) 严格校验：指向已不登记模型的聚合保存 → 400
        r = client.post("/api/aggregates", headers=hdr,
                       json={"name": "B144", "entries": [f"{P}/beta"]})
        assert r.status_code == 400, r.text

        # 4) model_map 别名是登记目标，不受清理影响
        r = client.post("/api/providers", headers=hdr,
                       json={**base, "models": ["alpha"], "model_map": {"fast": "alpha"}})
        assert r.status_code == 200, r.text
        assert client.post("/api/aggregates", headers=hdr,
                           json={"name": "C144", "entries": [f"{P}/fast"]}).status_code == 200
        assert entries("C144") == [f"{P}/fast"]

        # 5) 陈旧条目（绕过 API 校验直写 registry，模拟手改 JSON）：
        #    API 标记不可用；下一次模型清单变更时被清理，聚合丢空随之删除
        reg.upsert_aggregate("D144", [f"{P}/ghost"])
        d = agg("D144")
        assert d and d["entries"][0]["ok"] is False and d["usable"] == 0, d
        r = client.post("/api/providers", headers=hdr,
                       json={**base, "models": ["alpha"], "model_map": {"fast": "alpha"}})
        assert r.status_code == 200, r.text
        assert entries("D144") is None, names()

        # 6) 平台删除 → 指向它的聚合条目全部清理，聚合丢空删除
        assert client.delete(f"/api/providers/{P}", headers=hdr).status_code == 200
        assert "A144" not in names() and "C144" not in names(), names()
    finally:
        client.delete(f"/api/providers/{P}", headers=hdr)
        for n in ("A144", "B144", "C144", "D144"):
            client.delete(f"/api/aggregates/{n}", headers=hdr)


# ---------------------------------------------------------------- R145 平台新增 key：掩码+新明文混合提交不得丢旧 key（v2.0.2）
def test_r145_provider_keys_additive_save(client, hdr):
    """平台 API Key 增量保存（R145）：编辑平台时已存 key 以掩码显示，用户在掩码行后
    换行加新 key 保存，旧逻辑把「含明文的提交」当整体替换 → 全部旧 key 丢失。现语义：
    1) 加新 key（掩码行原样 + 新明文行）→ 旧 key 按掩码指纹保留，新 key 追加；
    2) 删某条掩码行 → 只移除那一个 key，其余不受影响；
    3) 不动 key 直接保存 → 不丢 key；
    4) key 框清空保存 → 保留现有 key（掩码框里删不掉自己看不到的明文，防误清空）；
    5) 主动删掉全部掩码行、纯明文新 key → 用户意图整体替换，重复行去重；
    6) 匹配不到已存 key 的伪掩码行 → 丢弃，绝不把掩码串存成 key。"""
    P = "keyadd145"
    K1 = "sk-legacykey-one-abcdefgh"
    K2 = "sk-legacykey-two-xyz99999"
    K3 = "sk-newkey-third-1234567890"
    K4 = "sk-extra-dup-check-55555555"
    from app.main import store, tenants
    reg = tenants.get(store.first_admin()["id"])

    def save(keys, **kw):
        body = {"platform": P, "name": "KeyAdd", "type": "openai",
                "base_url": "https://fake145.example.com/v1",
                "api_keys": keys, "models": ["m145"]}
        body.update(kw)
        r = client.post("/api/providers", headers=hdr, json=body)
        assert r.status_code == 200, r.text
        return list(reg.find(P).api_keys)

    def masks():
        d = client.get("/api/providers", headers=hdr).json()["data"]
        return [x["api_keys"] for x in d if x["platform"] == P][0]

    try:
        assert save([K1, K2]) == [K1, K2]

        # 1) 新增：掩码行保持原样 + 新明文追加 → 旧 key 全在
        m1, m2 = masks()
        assert m1 != K1 and K1 not in masks(), "列表接口必须只给掩码"
        assert save([m1, m2, K3]) == [K1, K2, K3]

        # 2) 删掉中间一条掩码行 → 只少那一个
        m1, _, m3 = masks()
        assert save([m1, m3]) == [K1, K3]

        # 3) 不动直接保存 → 不丢
        m1, m3 = masks()
        assert save([m1, m3]) == [K1, K3]

        # 4) 清空 key 框 → 现有 key 原样保留
        assert save([]) == [K1, K3]

        # 5) 主动整体替换（掩码行全删、纯明文）→ 只留新 key，重复行去重
        assert save([K4, K4]) == [K4]

        # 6) 伪掩码行（匹配不到已存 key）→ 丢弃，不存成 key
        m4 = masks()[0]
        assert save([m4, "sk-…ffff"]) == [K4]
    finally:
        client.delete(f"/api/providers/{P}", headers=hdr)

# ---------------------------------------------------------------- R146 embedding 模型：探测识别 + /v1/embeddings 透传（v2.0.2）
def test_r146_embedding_probe_and_endpoint(client, hdr, upstream_port):
    """embedding 模型与 image/video/audio 同等支持（R146）：
    1) 未配类型的模型 chat 探测遇 embedding 特征 4xx（提示 /v1/embeddings）→ 判 embed 保留，
       自动回填 extra.model_kind=embedding（不依赖有没有 removed）；
    2) 已标 embedding 类型的模型探测改走 /v1/embeddings 端点，该端点 404（模型下线）→ 正常判 removed；
    3) 回填后 /v1/embeddings 直接透传可用；embedding 模型误入 chat 端点 → 400 指路。"""
    r1 = client.post("/api/providers", headers=hdr, json={
        "platform": "r146emb", "name": "R146 emb(fake)", "type": "openai",
        "base_url": f"http://127.0.0.1:{upstream_port}/v1", "api_keys": ["pk"],
        "models": ["m-genemb", "m-embgone", "fake-gpt"],
        "extra": {"model_kind": {"m-embgone": "embedding"}}})
    assert r1.status_code == 200, r1.text
    bh = {"authorization": f"Bearer {conftest._TEST_API_KEY}"}
    try:
        rp = client.post("/api/providers/r146emb/probe", headers=hdr, json={})
        assert rp.status_code == 200, rp.text
        j = rp.json()
        gen = {x["model"]: x for x in j["gen"]}
        # 1) 未配类型 → chat 探测 → embedding 签名 → embed 保留 + 自动回填
        assert "m-genemb" in j["kept"], j
        assert gen.get("m-genemb", {}).get("kind") == "embedding", j["gen"]
        assert "m-genemb" in j["gen_kind_set"], j
        # 2) 已标 embedding → /embeddings 探测 → 404 → removed（不误留）
        assert any(x["model"] == "m-embgone" for x in j["removed"]), j
        assert "m-embgone" not in j["kept"], j
        # 普通模型照旧 ok
        assert "fake-gpt" in j["kept"], j
        # 回填已持久化
        p = [x for x in client.get("/api/providers", headers=hdr).json()["data"]
             if x["platform"] == "r146emb"][0]
        assert (p.get("extra") or {}).get("model_kind", {}).get("m-genemb") == "embedding", \
            p.get("extra")
        # 3) 回填后 /v1/embeddings 直接透传可用
        re_ = client.post("/v1/embeddings", headers=bh,
                          json={"model": "m-genemb", "input": "ping"})
        assert re_.status_code == 200, re_.text
        d = re_.json()
        assert d["object"] == "list" and d["data"][0]["embedding"] == [0.1] * 8, d
        # embedding 模型误入 chat 端点 → 400 指路（model_kind 校验）
        rc = client.post("/v1/chat/completions", headers=bh, json={
            "model": "m-genemb", "messages": [{"role": "user", "content": "hi"}]})
        assert rc.status_code == 400, rc.text
        assert "/v1/embeddings" in rc.text, rc.text
    finally:
        client.delete("/api/providers/r146emb", headers=hdr)


def test_r147_google_style_embedding_name_probe(client, hdr, upstream_port):
    """embedding-only 模型 google 同款签名（R147）：chat 端点 404 "models/… is not found"、
    报错里没有任何 /embeddings 提示 → 签名漏判 → 模型名含 embed 自动去 /embeddings 端点重验，
    通过则判 embedding 保留 + 类型回填（此前会误判 removed 把模型清出清单——生产反馈：
    google 平台 gemini-embedding-1/2 探测 404 被移除，但 /v1/embeddings 与 curl 直连均可用）。"""
    r1 = client.post("/api/providers", headers=hdr, json={
        "platform": "r147emb", "name": "R147 embname(fake)", "type": "openai",
        "base_url": f"http://127.0.0.1:{upstream_port}/v1", "api_keys": ["pk"],
        "models": ["fake-gemini-embedding-2", "fake-gpt"]})
    assert r1.status_code == 200, r1.text
    bh = {"authorization": f"Bearer {conftest._TEST_API_KEY}"}
    try:
        rp = client.post("/api/providers/r147emb/probe", headers=hdr, json={})
        assert rp.status_code == 200, rp.text
        j = rp.json()
        # 按模型名重验 /embeddings 通过 → embed 保留 + 自动回填类型
        assert "fake-gemini-embedding-2" in j["kept"], j
        assert not any(x["model"] == "fake-gemini-embedding-2" for x in j["removed"]), j
        gen = {x["model"]: x for x in j["gen"]}
        assert gen.get("fake-gemini-embedding-2", {}).get("kind") == "embedding", j["gen"]
        assert "fake-gemini-embedding-2" in j["gen_kind_set"], j
        # 普通模型照旧 ok
        assert "fake-gpt" in j["kept"], j
        # 回填已持久化
        p = [x for x in client.get("/api/providers", headers=hdr).json()["data"]
             if x["platform"] == "r147emb"][0]
        assert (p.get("extra") or {}).get("model_kind", {}).get("fake-gemini-embedding-2") == "embedding", \
            p.get("extra")
        # 回填后 /v1/embeddings 透传端点可用
        re_ = client.post("/v1/embeddings", headers=bh,
                          json={"model": "fake-gemini-embedding-2", "input": "ping"})
        assert re_.status_code == 200, re_.text
        assert re_.json()["data"][0]["embedding"] == [0.1] * 8, re_.text
    finally:
        client.delete("/api/providers/r147emb", headers=hdr)


def test_r148_models_expose_context_length(client, hdr, upstream_port):
    """v2.0.2: /v1/models 按模型暴露 context_length（extra.model_context_length）。

    relay 此前 /v1/models 只回 id/object/created/owned_by，不带任何上下文窗口元数据
    （上游 Agnes 等也不返回）→ Hermes 等客户端无从得知真实窗口，只能落本地缓存 /
    探测兜底值。现操作员把窗口写进平台 extra.model_context_length（模型名 -> 正整数，
    支持 '*' 通配），/v1/models 与 /v1/models/{id} 透传给客户端（Hermes 自动识别
    context_length 字段并缓存，无需再探测）。未配置的模型不带该键（向后兼容）。"""
    r1 = client.post("/api/providers", headers=hdr, json={
        "platform": "r148ctx", "name": "R148 ctx(fake)", "type": "openai",
        "base_url": f"http://127.0.0.1:{upstream_port}/v1", "api_keys": ["pk"],
        "models": ["ctx-big", "ctx-fallback", "ctx-wild"],
        "model_map": {"ctx-alias": "ctx-big"},
        "extra": {"model_context_length": {
            "ctx-big": 524288, "ctx-fallback": "131072", "*": 128000}}})
    assert r1.status_code == 200, r1.text
    bh = {"authorization": f"Bearer {conftest._TEST_API_KEY}"}
    try:
        data = {m["id"]: m for m in client.get("/v1/models", headers=bh).json()["data"]}
        # 精确模型名
        assert data["r148ctx/ctx-big"]["context_length"] == 524288, data["r148ctx/ctx-big"]
        # 数字字符串值同样认
        assert data["r148ctx/ctx-fallback"]["context_length"] == 131072, data["r148ctx/ctx-fallback"]
        # 无精确值的模型落 '*' 通配
        assert data["r148ctx/ctx-wild"]["context_length"] == 128000, data["r148ctx/ctx-wild"]
        # 别名继承其指向的真实模型窗口
        assert data["r148ctx/ctx-alias"]["context_length"] == 524288, data.get("r148ctx/ctx-alias")
        # 单模型端点同样带出
        single = client.get("/v1/models/r148ctx/ctx-big", headers=bh).json()
        assert single.get("context_length") == 524288, single
        # 值已持久化进平台配置
        p = [x for x in client.get("/api/providers", headers=hdr).json()["data"]
             if x["platform"] == "r148ctx"][0]
        assert (p.get("extra") or {}).get("model_context_length", {}).get("ctx-big") in (524288, "524288"), \
            p.get("extra")
    finally:
        client.delete("/api/providers/r148ctx", headers=hdr)

    # 未配置 model_context_length 的平台 → 不带 context_length 键（向后兼容）
    r2 = client.post("/api/providers", headers=hdr, json={
        "platform": "r148plain", "type": "openai",
        "base_url": f"http://127.0.0.1:{upstream_port}/v1", "api_keys": ["pk"],
        "models": ["no-ctx-model"]})
    assert r2.status_code == 200, r2.text
    try:
        data = {m["id"]: m for m in client.get("/v1/models", headers=bh).json()["data"]}
        assert "context_length" not in data["r148plain/no-ctx-model"], data["r148plain/no-ctx-model"]
    finally:
        client.delete("/api/providers/r148plain", headers=hdr)


# ---------------------------------------------------------------- R149 失效模型绑定：按名列出 + 一键解绑（v2.0.3）
def test_r149_key_binding_dead_models_unbind(client, hdr, upstream_port):
    """上游平台删掉某模型后，key 行 models_missing 按名列出失效模型（设置页表格内联列名 +
    「模型」弹窗警示块 + 「移除 N 个不存在模型」一键解绑，前端 R149）：
    解绑语义 = PATCH models 提交「当前绑定 − models_missing」（全为现存有效名，入参校验必过），
    保留其余有效绑定；全移除则 models:[] 恢复不限制（存 None）；直接提交已删模型名仍 400
    （校验不放松）；汇总视角（?tenant=all）按 key 归属账号自己的上游配置判 missing（R144 口径）。"""
    p = client.post("/api/providers", headers=hdr, json={
        "platform": "r149plat", "name": "R149(fake)", "type": "openai",
        "base_url": f"http://127.0.0.1:{upstream_port}/v1", "api_keys": ["pk"],
        "models": ["r149m1", "r149m2"]})
    assert p.status_code == 200, p.text
    cr = client.post("/api/keys", headers=hdr, json={
        "label": "r149", "models": ["r149plat/r149m1", "r149plat/r149m2"]})
    assert cr.status_code == 200, cr.text
    kid = cr.json()["id"]
    try:
        # 上游删掉 r149m1（平台设置保存：models 列表收窄 = 删模型）
        p2 = client.post("/api/providers", headers=hdr, json={
            "platform": "r149plat", "name": "R149(fake)", "type": "openai",
            "base_url": f"http://127.0.0.1:{upstream_port}/v1", "api_keys": ["pk"],
            "models": ["r149m2"]})
        assert p2.status_code == 200, p2.text

        st = client.get("/api/settings", headers=hdr).json()
        row = next(k for k in st["api_keys"] if k["id"] == kid)
        assert row["models_missing"] == ["r149plat/r149m1"], f"失效模型要按名列出: {row}"

        # 汇总视角同口径（按 key 归属账号自己的上游配置判）
        st_all = client.get("/api/settings?tenant=all", headers=hdr).json()
        assert st_all["keys_scope"] == "all"
        row_all = next(k for k in st_all["api_keys"] if k["id"] == kid)
        assert row_all["models_missing"] == ["r149plat/r149m1"], row_all
        assert row_all["self"] is True and row_all["owner_email"], row_all

        # 一键解绑（前端「移除 N 个不存在模型」按钮的 PATCH 语义）：保留有效绑定
        keep = [m for m in row["models"] if m not in row["models_missing"]]
        assert keep == ["r149plat/r149m2"], keep
        r = client.patch(f"/api/keys/{kid}", headers=hdr, json={"models": keep})
        assert r.status_code == 200, r.text
        assert r.json()["models"] == ["r149plat/r149m2"], r.json()
        st2 = client.get("/api/settings", headers=hdr).json()
        row2 = next(k for k in st2["api_keys"] if k["id"] == kid)
        assert row2["models_missing"] == [] and row2["models"] == ["r149plat/r149m2"], row2

        # 全移除（keep 为空 = 恢复不限制，后端存 None）
        r3 = client.patch(f"/api/keys/{kid}", headers=hdr, json={"models": []})
        assert r3.status_code == 200, r3.text
        assert r3.json()["models"] is None, r3.json()
        st3 = client.get("/api/settings", headers=hdr).json()
        row3 = next(k for k in st3["api_keys"] if k["id"] == kid)
        assert row3["models_missing"] == [] and not row3["models"], row3

        # 直接提交已删模型名 → 仍 400 并点名（入参校验不放松）
        r4 = client.patch(f"/api/keys/{kid}", headers=hdr, json={"models": ["r149plat/r149m1"]})
        assert r4.status_code == 400 and "r149plat/r149m1" in r4.text, r4.text
    finally:
        client.delete(f"/api/keys/{kid}", headers=hdr)
        client.delete("/api/providers/r149plat", headers=hdr)


# ---------------------------------------------------------------- R150 删模型后同步清理 extra.model_kind 陈旧条目（v2.0.3）
def test_r150_set_models_prunes_stale_model_kind(client, hdr, upstream_port):
    """生产反馈：非聊天模型（google gemini-embedding-*）被探测自动删除后，其「类型」回填条目
    仍留在平台高级 JSON（extra.model_kind）里——探测删除 / refresh 拉取只改写模型清单，
    model_kind 不同步。现：Registry.set_models（探测删模型与 refresh 拉取的共同收口）
    同步清掉清单里已不存在模型的 model_kind 条目：
    1) 清单收窄 → 被删模型条目移除、剩余模型条目保留；
    2) 加模型 / 无陈旧条目 → 现条目原样不动；
    3) 探测 e2e（用户现场）：标了 embedding 的非聊天模型，上游 /embeddings 已下线（404）
       → 探测按 model_kind 改 ping /embeddings（R146 语义：chat ping 必 4xx 会误判）→ 判 removed
       删除后，其 model_kind 回填条目同步清掉，保留模型不受影响。"""
    from app.main import store, tenants
    reg = tenants.get(store.first_admin()["id"])
    P = "mk150"
    base = {"platform": P, "name": "R150 mk(fake)", "type": "openai",
            "base_url": f"http://127.0.0.1:{upstream_port}/v1", "api_keys": ["pk"]}

    def extra():
        p = [x for x in client.get("/api/providers", headers=hdr).json()["data"]
             if x["platform"] == P]
        return (p[0].get("extra") or {}) if p else {}

    try:
        assert client.post("/api/providers", headers=hdr,
                           json={**base, "models": ["r150a", "r150b", "r150gone"],
                                 "extra": {"model_kind": {"r150gone": "embedding",
                                                          "r150a": "image"}}}).status_code == 200
        assert extra().get("model_kind") == {"r150gone": "embedding", "r150a": "image"}, extra()

        # 1) 清单收窄（模拟探测删模型）：被删模型条目移除，剩余模型条目保留
        reg.set_models(P, ["r150a", "r150b"])
        assert extra().get("model_kind") == {"r150a": "image"}, extra()

        # 2) 加模型 / 无陈旧条目 → 现条目原样不动
        reg.set_models(P, ["r150a", "r150b", "r150c"])
        assert extra().get("model_kind") == {"r150a": "image"}, extra()

        # 3) 探测 e2e（用户现场）：m-embgone* 在 fake 上游 /embeddings 端点回 404（模型已下线），
        #    标了 embedding → 探测 ping /embeddings → 判 removed（命名不含 embed → 不走按名重验）
        P2 = "mk150probe"
        r1 = client.post("/api/providers", headers=hdr, json={
            **base, "platform": P2, "name": "R150 probe(fake)",
            "models": ["r150keep", "m-embgone-r150"],
            "extra": {"model_kind": {"m-embgone-r150": "embedding"}}})
        assert r1.status_code == 200, r1.text
        try:
            rp = client.post(f"/api/providers/{P2}/probe", headers=hdr, json={})
            assert rp.status_code == 200, rp.text
            j = rp.json()
            assert "m-embgone-r150" in [x["model"] for x in j["removed"]], j
            assert "r150keep" in j["kept"], j
            assert j["written"] is True, j
            p2 = [x for x in client.get("/api/providers", headers=hdr).json()["data"]
                  if x["platform"] == P2][0]
            assert p2.get("models") == ["r150keep"], p2.get("models")
            assert (p2.get("extra") or {}).get("model_kind") in (None, {}), \
                f"删除后 model_kind 陈旧条目要同步清掉: {p2.get('extra')}"
        finally:
            client.delete(f"/api/providers/{P2}", headers=hdr)
    finally:
        client.delete(f"/api/providers/{P}", headers=hdr)


# ---------------------------------------------------------------- R151 请求体 GBK 编码中文抢救（v2.0.3）
def test_r151_gbk_chinese_body_rescue(client, hdr, upstream_port):
    """中文 Windows 命令行（cmd/PowerShell/git-bash 转发原生 curl）把非 ASCII 参数按
    GBK/GB18030 发出，而 JSON 只收 UTF-8 → 含中文的请求体 400 invalid_json（用户生产实测：
    curl 调 /v1/embeddings 带中文报「请求体不是合法 JSON」；v2.0.0 R137 起仅生成类端点有
    抢救，/v1/chat/completions、/v1/embeddings、/v1/messages 均缺）。现由共用 helper
    _json_object_lenient 统一：UTF-8 首解失败按 gb18030 救一次（ASCII 结构字符不变、
    中文修回正确字节）→ 正常转发且上游（fake）收到原样中文；救不回仍 400 invalid_json
    且文案带 GBK 排查提示；embeddings 本地 4xx 按 R146 体例落请求日志 error 行。"""
    import json as _json
    from tests import fake_upstream
    def gbk(payload: dict) -> bytes:
        return _json.dumps(payload, ensure_ascii=False).encode("gb18030")

    bh = {"authorization": f"Bearer {conftest._TEST_API_KEY}",
          "content-type": "application/json"}
    # 1) /v1/chat/completions（_handle 补洞）：GBK 中文 body → 200，上游中文原样
    fake_upstream.RECEIVED.clear()
    r1 = client.post("/v1/chat/completions", headers=bh,
                     content=gbk({"model": "fake/fake-gpt", "messages": [
                         {"role": "user", "content": "你好啊！"}], "max_tokens": 1}))
    assert r1.status_code == 200, r1.text
    sent = [x for x in fake_upstream.RECEIVED if x["path"] == "/v1/chat/completions"][-1]["body"]
    assert sent["messages"][0]["content"] == "你好啊！", sent

    # 2) /v1/embeddings（用户场景）：GBK 中文 input → 200，上游 input 原样
    rp = client.post("/api/providers", headers=hdr, json={
        "platform": "r151emb", "name": "R151 emb(fake)", "type": "openai",
        "base_url": f"http://127.0.0.1:{upstream_port}/v1", "api_keys": ["pk"],
        "models": ["fake-gemini-embedding-2"],
        "extra": {"model_kind": {"fake-gemini-embedding-2": "embedding"}}})
    assert rp.status_code == 200, rp.text
    try:
        fake_upstream.RECEIVED.clear()
        r2 = client.post("/v1/embeddings", headers=bh,
                         content=gbk({"model": "r151emb/fake-gemini-embedding-2",
                                      "input": "你好啊！"}))
        assert r2.status_code == 200, r2.text
        assert r2.json()["data"][0]["embedding"] == [0.1] * 8, r2.text
        sent2 = [x for x in fake_upstream.RECEIVED if x["path"] == "/v1/embeddings"][-1]["body"]
        assert sent2["input"] == "你好啊！", sent2

        # 3) 救不回（UTF-8/GB18030 皆非合法 JSON）→ 三端点均 400 + 文案带 GBK 提示
        bad = b"\x00\x01\x02not json"
        for path in ("/v1/chat/completions", "/v1/embeddings"):
            rn = client.post(path, headers=bh, content=bad)
            assert rn.status_code == 400, (path, rn.text)
            assert "invalid_json" in rn.text and "GBK" in rn.text, (path, rn.text)
        # anthropic 形 error body 无 code 字段（体例同 R141），只断言文案
        rn3 = client.post("/v1/messages", headers=bh, content=bad)
        assert rn3.status_code == 400, rn3.text
        assert "不是合法 JSON" in rn3.text and "GBK" in rn3.text, rn3.text
        # embeddings 本地 4xx 落 error 日志行（R146 体例）
        erow = [x for x in client.app.state.store.recent(5)
                if x.get("status") == "error" and x.get("http_status") == 400
                and "不是合法 JSON" in (x.get("error") or "")]
        assert erow, "embeddings 非法 JSON 应落 error 日志行"
    finally:
        client.delete("/api/providers/r151emb", headers=hdr)

    # 4) helper 单元面：空 body → {}；非对象 JSON → None；GBK 字节 → 修回 dict
    from app.main import _json_object_lenient
    assert _json_object_lenient(b"") == {}
    assert _json_object_lenient(b"[1, 2]") is None
    assert _json_object_lenient(gbk({"a": "中"})) == {"a": "中"}
