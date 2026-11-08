"""把「关闭思考」翻译成各上游平台真正认的线上字段。

这张表是从 Hermes Agent 的 provider profile（``build_api_kwargs_extras``）逐条抄过来的，
也就是 Hermes 自己在各平台关闭思考时实际发的字节。要点：

* 每个平台认的字段不一样，写错等于没关（甚至 400）。
* ``extra_body.X`` 在 Hermes 里由 SDK 摊平到请求体，这里直接写顶层 ``X``。
* Kimi/Moonshot 的 ``thinking`` 与 ``reasoning_effort`` 互斥（同时发会 400）；
  DeepSeek 每次请求都必须显式给 ``thinking``，否则服务端默认开思考。
* Anthropic 原生协议没有「disabled」类型，**省略** ``thinking`` 就是关。

用法：``off_payload(upstream, out_fmt, model)`` 返回要合并进请求体的 dict。
"""

from __future__ import annotations

from urllib.parse import urlparse

# 明确的写法 → 要发的字段。UI 里的下拉就是这几个 key。
FIELD_PAYLOADS: dict[str, dict] = {
    "reasoning_effort": {"reasoning_effort": "none"},
    "think": {"reasoning_effort": "none", "think": False},
    "enable_thinking": {"enable_thinking": False},
    "thinking_type": {"thinking": {"type": "disabled"}},
    "thinking_object": {"reasoning": {"enabled": False, "effort": "none"}},
    "chat_template": {"chat_template_kwargs": {"enable_thinking": False}},
    "none": {},
}

FIELD_LABELS: dict[str, str] = {
    "auto": "自动（按平台，同 Hermes）",
    "reasoning_effort": 'reasoning_effort:"none"',
    "think": 'reasoning_effort:"none" + think:false（Ollama）',
    "enable_thinking": "enable_thinking:false（通义/DashScope）",
    "thinking_type": 'thinking:{"type":"disabled"}（智谱/DeepSeek/Kimi）',
    "thinking_object": 'reasoning:{"enabled":false}（OpenRouter/Nous）',
    "chat_template": "chat_template_kwargs（vLLM/SGLang 系）",
    "none": "不注入（只清掉客户端的）",
}

# 命中这些词就认为是该平台（平台名 / 显示名 / base_url 一起匹配）
_OLLAMA_HOSTS = ("ollama.com", "ollama.ai")
_ZAI = ("zhipu", "bigmodel", "z.ai", "glm")
_DEEPSEEK = ("deepseek",)
_KIMI = ("moonshot", "kimi")
_DASHSCOPE = ("dashscope", "alibaba", "aliyuncs", "bailian", "qwen", "siliconflow", "modelscope")
_GEMINI = ("generativelanguage.googleapis.com", "gemini", "vertex")
_MINIMAX = ("minimax",)
_REASONING_OBJECT = ("openrouter", "nousresearch", "ai-gateway", "portal.nousresearch")


def _probe(upstream) -> tuple[str, str, str]:
    blob = f"{getattr(upstream, 'platform', '')} {getattr(upstream, 'name', '')} " \
           f"{getattr(upstream, 'base_url', '')}".lower()
    raw = (getattr(upstream, "base_url", "") or "").strip()
    try:
        parsed = urlparse(raw if "://" in raw else f"//{raw}")
        host = (parsed.hostname or "").lower()
        port = parsed.port
    except ValueError:
        host, port = "", None
    return blob, host, f"{host}:{port}"


def is_gemini_family(upstream) -> bool:
    """Google 系上游（Gemini 思考模型相关处理：关思考写法、thought_signature 补齐）。"""
    blob, _, _ = _probe(upstream)
    return any(w in blob for w in _GEMINI)


def is_ollama_endpoint(upstream) -> bool:
    """Ollama 端点：端口 11434、ollama.com 域名，或平台名/显示名里写着 ollama。"""
    blob, host, hostport = _probe(upstream)
    if hostport.endswith(":11434"):
        return True
    if any(h == host or host.endswith("." + h) for h in _OLLAMA_HOSTS):
        return True
    return "ollama" in blob


def auto_off_payload(upstream, out_fmt: str, model: str = "") -> dict:
    """按平台自动挑「关闭思考」的字段（与 Hermes 各 profile 一致）。"""
    if out_fmt == "anthropic":
        # Anthropic 原生协议没有 disabled 类型，省略 thinking 就是关。
        return {}

    blob, host, _ = _probe(upstream)
    m = (model or "").lower()

    if is_ollama_endpoint(upstream):
        # Ollama：顶层 reasoning_effort="none" 是唯一认的开关，think=false 是老版本的写法，
        # 两个一起发（Hermes 对 custom 端点就是这么干的）。
        return dict(FIELD_PAYLOADS["think"])

    if any(w in blob for w in _ZAI):
        # 智谱：GLM-5.3 拒绝 disabled，只能降到最低档 low。
        if "glm-5.3" in m or "glm-5.3" in blob:
            return {"reasoning_effort": "low"}
        return dict(FIELD_PAYLOADS["thinking_type"])

    if any(w in blob for w in _KIMI):
        return dict(FIELD_PAYLOADS["thinking_type"])

    if any(w in blob for w in _DEEPSEEK) and "chat.deepseek" not in blob:
        return dict(FIELD_PAYLOADS["thinking_type"])

    if any(w in blob for w in _DASHSCOPE):
        return dict(FIELD_PAYLOADS["enable_thinking"])

    if any(w in blob for w in _GEMINI) or "gemini" in m:
        # 实测（2026-11-08，/v1beta/openai 直连验证）：Google 的 OpenAI 兼容端点**不再接受**
        # 顶层 google.thinking_config 扩展字段（发了直接 400 "Unknown name \"google\""），
        # 认的是 OpenAI 系 reasoning_effort：gemini-3.1 实测 none/minimal/high 都收，
        # 且 none 时响应无 reasoning_content / reasoning_tokens——真关上了。
        return dict(FIELD_PAYLOADS["reasoning_effort"])

    if any(w in blob for w in _MINIMAX):
        return dict(FIELD_PAYLOADS["thinking_type"])

    if any(w in blob for w in _REASONING_OBJECT):
        return dict(FIELD_PAYLOADS["thinking_object"])

    # 默认：OpenAI 兼容平台的通用写法
    return dict(FIELD_PAYLOADS["reasoning_effort"])


def off_payload(upstream, out_fmt: str, model: str = "", field: str = "",
                custom: dict | None = None) -> dict:
    """最终要写进上游请求体的「关闭思考」字段。

    优先级：模型级指定的 field > extra.thinking_off_payload（自定义 dict）> 自动。
    """
    if isinstance(custom, dict) and custom:
        return dict(custom)
    key = (field or "").strip().lower()
    if key and key not in ("auto", "off"):
        return dict(FIELD_PAYLOADS.get(key, {}))
    return auto_off_payload(upstream, out_fmt, model)


def describe(upstream, model: str = "") -> str:
    """给人看的一句话：这个平台关思考时会发什么。"""
    auto = auto_off_payload(upstream, "openai", model)
    return ", ".join(f"{k}={v!r}" for k, v in auto.items()) or "（省略 thinking 字段）"
