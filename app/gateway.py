"""转发核心：鉴权、路由、上游调用（含多 key 重试）、4 种协议组合的流式/非流式转发与用量记账。"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, AsyncIterator, Optional

import httpx
from fastapi.responses import JSONResponse, StreamingResponse

from . import config
from . import translate as T
from .config import ANTHROPIC_VERSION, CONNECT_TIMEOUT_S, VERSION
from . import reasoning as R
from .registry import Registry, Upstream
from .storage import Store

# Google 官方提供的 thought_signature 哨兵值：签名确实拿不到（客户端把 extra_content 丢了、
# 跨模型迁移历史等）时填这个，Google 跳过校验（代价是模型上下文里少了那段思考）。
THOUGHT_SIGNATURE_BYPASS = "skip_thought_signature_validator"


def inject_thought_signatures(payload: dict) -> int:
    """Google 思考模型（Gemini 3/4）工具调用要求：当前轮每个 step 第一个 functionCall 必须
    带 thought_signature，缺了直接 400。客户端（OpenAI SDK/httpx）经常把响应里的
    extra_content.google.thought_signature 丢掉，这里给缺签名的 assistant tool_calls 补哨兵值。
    真实签名一律保留。返回补齐的数量。"""
    n = 0
    for m in payload.get("messages") or []:
        if not isinstance(m, dict) or m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            sig = ((tc.get("extra_content") or {}).get("google") or {}).get("thought_signature") \
                or tc.get("thought_signature")
            if sig:
                continue
            extra = tc.get("extra_content") if isinstance(tc.get("extra_content"), dict) else {}
            gg = extra.get("google") if isinstance(extra.get("google"), dict) else {}
            gg["thought_signature"] = THOUGHT_SIGNATURE_BYPASS
            extra["google"] = gg
            tc["extra_content"] = extra
            n += 1
    return n

SSE_HEADERS = {
    "cache-control": "no-cache",
    "x-accel-buffering": "no",
    "connection": "keep-alive",
}

log = logging.getLogger("relay.upstream")


class UpstreamError(Exception):
    def __init__(self, status: int, body: bytes | str, platform: str = "",
                 key_hint: str = "") -> None:
        self.status = status
        self.key_hint = key_hint
        self.body = body.decode("utf-8", "replace") if isinstance(body, bytes) else (body or "")
        self.platform = platform
        super().__init__(f"{status} {self.body[:200]}")

    def message(self) -> str:
        try:
            data = json.loads(self.body)
        except ValueError:
            return self.body[:500] or f"上游返回 HTTP {self.status}"
        if isinstance(data, dict):
            err = data.get("error")
            if isinstance(err, dict):
                return str(err.get("message") or err.get("type") or data)[:500]
            if isinstance(err, str):
                return err[:500]
            if data.get("message"):
                return str(data["message"])[:500]
        return str(data)[:500]


def new_request_id() -> str:
    return "req-" + uuid.uuid4().hex[:16]


def build_headers(up: Upstream, key: str, stream: bool) -> dict[str, str]:
    headers = {
        "content-type": "application/json",
        "accept": "text/event-stream" if stream else "application/json",
        "user-agent": f"llm-relay/{VERSION}",
    }
    if up.type == "anthropic":
        headers["x-api-key"] = key
        headers["authorization"] = f"Bearer {key}"
        headers["anthropic-version"] = str(up.extra.get("anthropic_version") or ANTHROPIC_VERSION)
        betas = up.extra.get("anthropic_beta")
        if betas:
            headers["anthropic-beta"] = ",".join(betas) if isinstance(betas, list) else str(betas)
    else:
        headers["authorization"] = f"Bearer {key}"
    for k, v in (up.extra.get("headers") or {}).items():
        headers[str(k)] = str(v)
    return headers


def error_response(fmt: str, status: int, message: str, code: str | None = None) -> JSONResponse:
    if fmt == "anthropic":
        etype = {
            400: "invalid_request_error", 401: "authentication_error",
            403: "permission_error", 404: "not_found_error",
            429: "rate_limit_error", 500: "api_error", 502: "api_error",
        }.get(status, "api_error")
        return JSONResponse(status_code=status,
                            content={"type": "error", "error": {"type": etype, "message": message}})
    return JSONResponse(status_code=status, content={
        "error": {"message": message, "type": "invalid_request_error" if status < 500 else "api_error",
                  "code": code or f"http_{status}"},
    })


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    estimated: bool = False


def mask_key(key: str) -> str:
    """把 api key 打码成可辨认的短标识，用来做用量统计的维度。"""
    key = key or ""
    if len(key) > 12:
        return f"{key[:6]}…{key[-4:]}"
    if key:
        return "…" + key[-4:]
    return ""


def normalize_reasoning_effort(up: Upstream, payload: dict) -> None:
    """线边界兼容：reasoning_effort="max" 只有 OpenAI 新模型接受，第三方 OpenAI 兼容端点
    （Google 等）会 400 INVALID_ARGUMENT（Google 只收 high/low/medium/minimal/none）——
    统一降级为 high；extra.reasoning_effort_map 可覆盖，如 {"max": "medium"}。
    OpenAI 官方 base 保持 max 不动。"""
    if up.type != "openai" or not isinstance(payload.get("reasoning_effort"), str):
        return
    key = payload["reasoning_effort"].strip().lower()
    remap = up.extra.get("reasoning_effort_map") or {}
    if key in remap:
        payload["reasoning_effort"] = remap[key]
    elif key == "max" and "api.openai.com" not in up.base_url:
        payload["reasoning_effort"] = "high"


class Gateway:
    def __init__(self, tenants, store: Store):
        self.tenants = tenants
        self.store = store
        self.client: httpx.AsyncClient | None = None
        self._proxy_clients: dict[str, httpx.AsyncClient] = {}   # 平台专用代理 -> 独立 client

    # ---------------- 生命周期 ----------------
    async def startup(self) -> None:
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(600.0, connect=CONNECT_TIMEOUT_S),
            limits=httpx.Limits(max_connections=200, max_keepalive_connections=50),
            follow_redirects=True,
            trust_env=True,
        )

    async def shutdown(self) -> None:
        for cli in self._proxy_clients.values():
            with contextlib.suppress(Exception):
                await cli.aclose()
        self._proxy_clients = {}
        if self.client is not None:
            await self.client.aclose()
            self.client = None

    @property
    def http(self) -> httpx.AsyncClient:
        if self.client is None:
            raise RuntimeError("Gateway 未启动")
        return self.client

    def client_for(self, up: Upstream) -> httpx.AsyncClient:
        """平台配了 proxy 就用该代理的独立 client（socks5/socks5h 走 httpx[socks]），否则用共享 client。"""
        proxy = (up.proxy or "").strip()
        if not proxy:
            return self.http
        cli = self._proxy_clients.get(proxy)
        if cli is None:
            # 并发首次创建可能重复建一两个，多出来的实例在 shutdown 一并关闭，无害
            cli = httpx.AsyncClient(
                timeout=httpx.Timeout(600.0, connect=CONNECT_TIMEOUT_S),
                limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
                follow_redirects=True,
                trust_env=True,
                proxy=proxy,
            )
            self._proxy_clients[proxy] = cli
        return cli

    # ---------------- 上游调用 ----------------
    async def send(self, reg, up: Upstream, url: str, payload: dict, *,
                   stream: bool) -> httpx.Response:
        """多 key 轮换重试。只在上游还没开始回内容前重试。"""
        attempts = max(1, min(len(up.api_keys), 3))
        last_status, last_body = 502, b"no available key"
        hint, key_idx = "", 0
        client = self.client_for(up)
        adapted = False
        for i in range(attempts):
            key, idx = reg.next_key(up)
            hint, key_idx = mask_key(key), int(idx)
            headers = build_headers(up, key, stream)
            req = client.build_request(
                "POST", url, json=payload, headers=headers,
                timeout=httpx.Timeout(up.timeout_s, connect=CONNECT_TIMEOUT_S),
            )
            try:
                resp = await client.send(req, stream=stream)
            except httpx.HTTPError as exc:
                reg.report(up, idx, False)
                last_status, last_body = 502, f"连接上游失败: {exc}".encode()
                if i == attempts - 1:
                    break
                continue
            if resp.status_code < 400:
                reg.report(up, idx, True)
                # 带上用了哪个 key，记账时写进 key_hint 维度
                with contextlib.suppress(Exception):
                    resp.extensions["relay_key_hint"] = hint
                    resp.extensions["relay_key_index"] = key_idx
                return resp
            # 400 自愈（只做一次）：Gemini 各代模型的“关闭思考”值词汇表不一样——
            # 实测 3.5+ 不认 reasoning_effort=none（400 "Request contains an invalid argument"），
            # 最低档是 minimal；2.x/3.0-3.4 认 none。表（reasoning.py）没覆盖到的新代模型
            # 也会在这里自动降级：none→minimal→去掉字段，同 key 原样重发一次；
            # 400 是参数问题不是 key 问题，key 记健康、不换 key。
            if (resp.status_code == 400 and not adapted and
                    isinstance(payload.get("reasoning_effort"), str) and
                    (R.is_gemini_family(up) or "gemini" in str(payload.get("model") or "").lower())):
                adapted = True
                current_effort = payload["reasoning_effort"]
                with contextlib.suppress(Exception):
                    last_body = await resp.aread()
                with contextlib.suppress(Exception):
                    await resp.aclose()
                last_status = resp.status_code
                if current_effort != "minimal":
                    payload["reasoning_effort"] = "minimal"
                else:
                    payload.pop("reasoning_effort", None)
                reg.report(up, idx, True)
                req = client.build_request(
                    "POST", url, json=payload, headers=headers,
                    timeout=httpx.Timeout(up.timeout_s, connect=CONNECT_TIMEOUT_S),
                )
                try:
                    resp = await client.send(req, stream=stream)
                except httpx.HTTPError as exc:
                    last_status, last_body = 502, f"连接上游失败: {exc}".encode()
                    if i == attempts - 1:
                        break
                    continue
                if resp.status_code < 400:
                    reg.report(up, idx, True)
                    with contextlib.suppress(Exception):
                        resp.extensions["relay_key_hint"] = hint
                        resp.extensions["relay_key_index"] = key_idx
                    return resp
                # 还是 400 → 走下面常规错误路径
            with contextlib.suppress(Exception):
                last_body = await resp.aread()
            with contextlib.suppress(Exception):
                await resp.aclose()
            last_status = resp.status_code
            retryable = resp.status_code in (401, 403, 408, 409, 425, 429) or resp.status_code >= 500
            reg.report(up, idx, not retryable)
            if not retryable:
                break
        raise UpstreamError(last_status, last_body, up.platform, key_hint=hint)

    async def relay_aggregate(self, *, in_fmt: str, kind: str, body: dict,
                              agg_name: str, client_model: str,
                              request_id: str, started_at: float, stream: bool,
                              est_prompt: int, client_ip: str = "",
                              user_id: int | None = None, client: str = "",
                              dkey_id: int | None = None):
        """聚合模型转发：在配置的多个「平台/模型」条目之间轮询 + 失败冷却自动切换
        （和多 key 轮换同一套语义：可重试错误冷却该条目 60s 并换下一个，
        不可重试错误直接返回）。"""
        reg = self.tenants.get(user_id) if user_id is not None else None
        chosen = reg.next_aggregate(agg_name) if reg else None
        if chosen is None:
            return error_response(in_fmt, 404,
                                  f"聚合模型 {agg_name!r} 没有可用条目：配置为空、条目指向上游已删的模型，"
                                  f"或全部条目都在冷却中。请检查控制台「聚合模型」卡片", "aggregate_empty")
        up, up_model, entry = chosen
        last_exc: Optional[UpstreamError] = None
        while True:
            try:
                return await self.relay(
                    in_fmt=in_fmt, kind=kind, body=body, upstream=up, up_model=up_model,
                    client_model=client_model or agg_name, request_id=request_id,
                    started_at=started_at, stream=stream, est_prompt=est_prompt,
                    client_ip=client_ip, user_id=user_id, client=client, dkey_id=dkey_id,
                    propagate_upstream_error=True,
                )
            except UpstreamError as exc:
                reg.report_aggregate(agg_name, entry, False)
                retryable = exc.status in (401, 403, 408, 409, 425, 429) or exc.status >= 500
                if not retryable:
                    raise  # 不可重试（如 400 参数错）：换条目也没用，交给 main 的统一错误处理
                last_exc = exc
                nxt = reg.next_aggregate(agg_name)
                if nxt is None or nxt[2] == entry:
                    break  # 没有别的健康条目（或全冷却回到同一个）
                up, up_model, entry = nxt
                continue
        log.warning("聚合模型 [%s] 所有条目均失败，最后错误: %s", agg_name,
                    last_exc.status if last_exc else "?")
        if last_exc is not None:
            return error_response(in_fmt, last_exc.status if last_exc.status >= 400 else 502,
                                  f"[{agg_name}] 所有条目均失败：{last_exc.message()}", "aggregate_exhausted")
        return error_response(in_fmt, 502, f"[{agg_name}] 没有可用条目", "aggregate_exhausted")

    # ---------------- 记账 ----------------
    def record(self, resp=None, **kw) -> None:
        if resp is not None:
            with contextlib.suppress(Exception):
                kw.setdefault("key_hint", resp.extensions.get("relay_key_hint") or "")
                kw.setdefault("key_index", resp.extensions.get("relay_key_index") or 0)
        try:
            self.store.record(**kw)
        except Exception:  # 记账失败不能影响转发
            pass

    # ---------------- 主流程 ----------------
    async def relay(
        self,
        *,
        in_fmt: str,
        kind: str,
        body: dict,
        upstream: Upstream,
        up_model: str,
        client_model: str,
        request_id: str,
        started_at: float,
        stream: bool,
        est_prompt: int,
        client_ip: str = "",
        user_id: int | None = None,
        client: str = "",
        dkey_id: int | None = None,
        propagate_upstream_error: bool = False,
    ):
        reg = self.tenants.get(user_id) if user_id is not None else None
        out_fmt = upstream.type
        # Google AI Studio 的 OpenAI 兼容 /v1beta/openai/models 返回带 "models/" 前缀的模型 id，
        # 但 chat/completions 只收无前缀的名字（带前缀直接 400）——线边界统一剥掉。
        if out_fmt == "openai" and up_model.startswith("models/"):
            up_model = up_model[len("models/"):]
        stream = bool(body.get("stream")) if stream is None else stream
        # 模型级思考设置：off 时在**发给上游的请求字段上**显式关闭思考
        policy, off_field = Registry.reasoning_rule(upstream, up_model)
        payload, url = self._build_payload(in_fmt, out_fmt, kind, body, upstream, up_model,
                                          est_prompt=est_prompt, policy=policy,
                                          off_field=off_field)
        # extra_body 只是 OpenAI SDK 客户端侧的载体键：raw HTTP 打上游前要摊平到请求体顶层。
        # Google 的 google.thinking_config（关思考等）必须挂在顶层，原样嵌套发出去上游会直接忽略。
        if out_fmt == "openai" and isinstance(payload.get("extra_body"), dict):
            extra = payload.pop("extra_body")
            for k, v in extra.items():
                payload.setdefault(k, v)
        # Google 思考模型：历史里缺 thought_signature 的 assistant 工具调用自动补哨兵值，
        # 避免 400 "Function call is missing a thought_signature"。平台 extra
        # thought_signature_bypass: false 可关闭补齐。
        if out_fmt == "openai" and (R.is_gemini_family(upstream)
                                     or "gemini" in (up_model or "").lower()) \
                and upstream.extra.get("thought_signature_bypass", True) is not False:
            if inject_thought_signatures(payload):
                pass  # 静默补齐；真实签名已带时不动
        try:
            resp = await self.send(reg, upstream, url, payload, stream=stream)
        except UpstreamError as exc:
            log.warning("上游错误 [%s] HTTP %s: %s", upstream.platform, exc.status, exc.message()[:300])
            self.record(
                request_id=request_id, platform=upstream.platform, model=up_model,
                model_alias=client_model, in_format=in_fmt, stream=stream,
                status="error", http_status=exc.status, error=exc.message(),
                latency_ms=int((time.perf_counter() - started_at) * 1000),
                client_ip=client_ip, reasoning_policy=policy,
                key_hint=getattr(exc, "key_hint", ""),
                user_id=user_id, client=client or "master", dkey_id=dkey_id,
            )
            # 聚合模型 failover：让调用方（relay_aggregate）决定换不换下一个条目
            if propagate_upstream_error:
                raise
            return error_response(in_fmt, exc.status if exc.status >= 400 else 502,
                                  f"[{upstream.platform}] {exc.message()}", "upstream_error")

        if stream:
            return self._streaming_response(
                resp=resp, in_fmt=in_fmt, out_fmt=out_fmt, upstream=upstream,
                up_model=up_model, client_model=client_model, request_id=request_id,
                started_at=started_at, est_prompt=est_prompt, body=body,
                policy=policy, client_ip=client_ip, user_id=user_id, client=client, dkey_id=dkey_id,
            )
        return await self._json_response(
            resp=resp, in_fmt=in_fmt, out_fmt=out_fmt, upstream=upstream,
            up_model=up_model, client_model=client_model, request_id=request_id,
            started_at=started_at, est_prompt=est_prompt,
            policy=policy, client_ip=client_ip, user_id=user_id, client=client, dkey_id=dkey_id,
        )

    # ---------------- 思考控制 ----------------
    # 客户端可能用到的所有“思考相关”字段。策略不是 auto 时，一律先清掉，避免客户端把思考开回来。
    THINKING_KEYS = (
        "reasoning_effort", "reasoning", "thinking", "enable_thinking",
        "thinking_budget", "thinking_budget_tokens", "thinking_config",
        "thinkingConfig", "include_thoughts", "reasoning_effort_level",
    )
    THINKING_OFF_MODES = ("auto", "reasoning_effort", "think", "enable_thinking",
                          "thinking_type", "thinking_object", "chat_template", "none")

    @classmethod
    def _strip_thinking_fields(cls, payload: dict) -> None:
        for k in cls.THINKING_KEYS:
            payload.pop(k, None)
        ctk = payload.get("chat_template_kwargs")
        if isinstance(ctk, dict):
            ctk.pop("enable_thinking", None)
            if not ctk:
                payload.pop("chat_template_kwargs", None)

    @classmethod
    def _thinking_off_payload(cls, upstream: Upstream, out_fmt: str, field: str = "",
                              model: str = "") -> dict:
        """返回要显式写进上游请求体的"关闭思考"字段。

        优先级：模型级指定的 field > extra.thinking_off_payload（自定义 dict）
                > extra.thinking_off（老的平台级写法）> 按平台自动（表见 app/reasoning.py）。
        """
        from . import reasoning

        custom = upstream.extra.get("thinking_off_payload") or upstream.extra.get("reasoning_off_payload")
        mode = str(upstream.extra.get("thinking_off") or "").lower().strip()
        use_field = (field or "").lower().strip()
        if not use_field and mode in cls.THINKING_OFF_MODES:
            use_field = mode           # 老配置继续生效
        return reasoning.off_payload(upstream, out_fmt, model=model,
                                     field=use_field, custom=custom)

    @classmethod
    def _apply_reasoning(cls, payload: dict, out_fmt: str, upstream: Upstream, policy: str,
                         off_field: str = "", model: str = "") -> dict:
        """按模型设置改写发给上游的请求体。

        off：**先清掉客户端所有思考字段，再按平台写入"关闭"字段**——上游收到的请求就是关闭状态。
        low/medium/high：同样以模型设置为准（覆盖客户端值）。
        auto：不动，听客户端的。
        """
        if policy == "auto":
            return payload

        cls._strip_thinking_fields(payload)

        if policy == "off":
            off = cls._thinking_off_payload(upstream, out_fmt, field=off_field, model=model)
            if off:
                payload.update(off)
            return payload

        if out_fmt == "openai":
            payload["reasoning_effort"] = policy
            return payload

        budget = {"low": 1024, "medium": 4096, "high": 8192}.get(policy, 4096)
        max_tokens = int(payload.get("max_tokens") or 8192)
        if max_tokens <= budget:
            payload["max_tokens"] = budget + 1024
        payload["thinking"] = {"type": "enabled", "budget_tokens": budget}
        # Anthropic 开思考时 temperature/top_p 必须为默认值
        payload.pop("temperature", None)
        payload.pop("top_p", None)
        return payload

    def _build_payload(self, in_fmt: str, out_fmt: str, kind: str, body: dict,
                       upstream: Upstream, up_model: str, est_prompt: int,
                       policy: str = "auto", off_field: str = "") -> tuple[dict, str]:
        if kind == "messages":
            path_kind = "messages" if out_fmt == "anthropic" else "chat"
        else:
            path_kind = "chat" if out_fmt == "openai" else "messages"
        url = upstream.url_for(path_kind)
        default_max = int(upstream.extra.get("default_max_tokens") or 8192)

        if in_fmt == "openai" and out_fmt == "openai":
            payload = dict(body)
            payload["model"] = up_model
            if payload.get("stream") and upstream.extra.get("inject_stream_usage", True):
                so = dict(payload.get("stream_options") or {})
                so.setdefault("include_usage", True)
                payload["stream_options"] = so
        elif in_fmt == "openai" and out_fmt == "anthropic":
            payload = T.openai_to_anthropic(body, up_model, default_max_tokens=default_max)
        elif in_fmt == "anthropic" and out_fmt == "anthropic":
            payload = {k: v for k, v in body.items() if k != "model"}
            payload["model"] = up_model
            if "max_tokens" not in payload:
                payload["max_tokens"] = default_max
        else:
            payload = T.anthropic_to_openai(body, up_model)
            if payload.get("stream") and upstream.extra.get("inject_stream_usage", True):
                payload["stream_options"] = {"include_usage": True}

        payload = self._apply_reasoning(payload, out_fmt, upstream, policy, off_field, model=up_model)
        if out_fmt == "openai":
            normalize_reasoning_effort(upstream, payload)
        return payload, url

    # ---------------- 流式 ----------------
    def _make_bridge(self, in_fmt: str, out_fmt: str, client_model: str,
                     body: dict, est_prompt: int) -> T.Bridge:
        if in_fmt == "openai" and out_fmt == "openai":
            asked = bool((body.get("stream_options") or {}).get("include_usage"))
            bridge: T.Bridge = T.OpenAIPassthrough(include_usage=asked, swallow_usage=not asked)
        elif in_fmt == "anthropic" and out_fmt == "anthropic":
            bridge = T.AnthropicPassthrough()
        elif in_fmt == "openai" and out_fmt == "anthropic":
            asked = bool((body.get("stream_options") or {}).get("include_usage"))
            bridge = T.AnthropicToOpenAIStream(client_model, include_usage=asked)
        else:
            bridge = T.OpenAIToAnthropicStream(client_model, est_input_tokens=est_prompt)
        return bridge

    def _streaming_response(self, *, resp: httpx.Response, in_fmt: str, out_fmt: str,
                            upstream: Upstream, up_model: str, client_model: str,
                            request_id: str, started_at: float, est_prompt: int,
                            body: dict, policy: str = "auto",
                            client_ip: str = "", user_id: int | None = None,
                            client: str = "", dkey_id: int | None = None):
        bridge = self._make_bridge(in_fmt, out_fmt, client_model, body, est_prompt)
        gw = self

        async def gen() -> AsyncIterator[str]:
            parser = T.SSEParser()
            ttft_ms = 0
            status = "ok"
            error = ""
            try:
                async for text in resp.aiter_text():
                    for event, data_lines, comments in parser.feed(text):
                        if comments and not data_lines:
                            yield "".join(c + "\n" for c in comments) + "\n"
                            continue
                        payload = "\n".join(data_lines)
                        for frame in bridge.push(event, payload):
                            if not ttft_ms and frame.strip():
                                ttft_ms = int((time.perf_counter() - started_at) * 1000)
                            yield frame
                for event, data_lines, comments in parser.flush():
                    payload = "\n".join(data_lines)
                    for frame in bridge.push(event, payload):
                        yield frame
                for frame in bridge.close():
                    yield frame
            except (asyncio.CancelledError, GeneratorExit):
                status = "aborted"
                raise
            except httpx.HTTPError as exc:
                status, error = "error", f"上游流中断: {exc}"
                yield T.sse_frame(json.dumps({"error": {"message": error}}, ensure_ascii=False))
            finally:
                usage = bridge.usage.finish()
                estimated = usage.estimated
                if usage.total_tokens == 0:
                    estimated = True
                    usage.prompt_tokens = est_prompt
                    usage.completion_tokens = T.estimate_tokens_from_chars(bridge.text_chars)
                    usage.total_tokens = usage.prompt_tokens + usage.completion_tokens
                if not ttft_ms:
                    ttft_ms = int((time.perf_counter() - started_at) * 1000)
                gw.record(
                    resp=resp,          # 必须带上，否则记账读不到用了哪把 upstream key
                    request_id=request_id, platform=upstream.platform, model=up_model,
                    model_alias=client_model, in_format=in_fmt, stream=True,
                    status=status, http_status=resp.status_code, error=error,
                    latency_ms=int((time.perf_counter() - started_at) * 1000),
                    ttft_ms=ttft_ms,
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                    total_tokens=usage.total_tokens,
                    cached_tokens=usage.cached_tokens,
                    cache_write_tokens=usage.cache_write_tokens,
                    reasoning_tokens=usage.reasoning_tokens,
                    estimated=estimated,
                    client_ip=client_ip,
                    user_id=user_id, client=client, dkey_id=dkey_id,
                    reasoning_policy=policy,
                )
                with contextlib.suppress(BaseException):
                    await resp.aclose()

        headers = dict(SSE_HEADERS)
        headers["x-request-id"] = request_id
        headers["x-relay-platform"] = upstream.platform
        headers["x-relay-model"] = up_model
        return StreamingResponse(gen(), media_type="text/event-stream", headers=headers)

    # ---------------- 非流式 ----------------
    async def _json_response(self, *, resp: httpx.Response, in_fmt: str, out_fmt: str,
                             upstream: Upstream, up_model: str, client_model: str,
                             request_id: str, started_at: float, est_prompt: int,
                             policy: str = "auto", client_ip: str = "",
                             user_id: int | None = None, client: str = "",
                             dkey_id: int | None = None):
        try:
            raw = await resp.aread()
        finally:
            with contextlib.suppress(Exception):
                await resp.aclose()
        if resp.status_code >= 400:
            self.record(
                resp=resp,
                request_id=request_id, platform=upstream.platform, model=up_model,
                model_alias=client_model, in_format=in_fmt, stream=False,
                status="error", http_status=resp.status_code,
                error=raw.decode("utf-8", "replace")[:800],
                latency_ms=int((time.perf_counter() - started_at) * 1000),
                client_ip=client_ip, reasoning_policy=policy,
                user_id=user_id, client=client, dkey_id=dkey_id,
            )
            return error_response(in_fmt, resp.status_code,
                                  f"[{upstream.platform}] {raw.decode('utf-8', 'replace')[:500]}",
                                  "upstream_error")
        try:
            data = json.loads(raw)
        except ValueError:
            self.record(
                resp=resp,
                request_id=request_id, platform=upstream.platform, model=up_model,
                model_alias=client_model, in_format=in_fmt, stream=False,
                status="error", http_status=502, error="上游返回非 JSON",
                latency_ms=int((time.perf_counter() - started_at) * 1000),
                client_ip=client_ip, reasoning_policy=policy,
                user_id=user_id, client=client, dkey_id=dkey_id,
            )
            return error_response(in_fmt, 502, "上游返回非 JSON 内容", "bad_upstream_response")

        if in_fmt == "openai" and out_fmt == "anthropic":
            usage = T.usage_from_anthropic((data or {}).get("usage"))
            out = T.anthropic_response_to_openai(data, client_model)
        elif in_fmt == "anthropic" and out_fmt == "openai":
            usage = T.usage_from_openai((data or {}).get("usage"))
            out = T.openai_response_to_anthropic(data, client_model)
        else:
            usage = (T.usage_from_openai if out_fmt == "openai" else T.usage_from_anthropic)(
                (data or {}).get("usage"))
            out = data
            if isinstance(out, dict):
                out["model"] = client_model

        estimated = usage.estimated
        if usage.total_tokens == 0:
            estimated = True
            usage.prompt_tokens = est_prompt
            usage.completion_tokens = T.estimate_tokens_from_chars(
                len(json.dumps(data, ensure_ascii=False)))
            usage.total_tokens = usage.prompt_tokens + usage.completion_tokens

        self.record(
            resp=resp,
            request_id=request_id, platform=upstream.platform, model=up_model,
            model_alias=client_model, in_format=in_fmt, stream=False,
            status="ok", http_status=200,
            latency_ms=int((time.perf_counter() - started_at) * 1000),
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            cached_tokens=usage.cached_tokens,
            cache_write_tokens=usage.cache_write_tokens,
            reasoning_tokens=usage.reasoning_tokens,
            estimated=estimated,
            client_ip=client_ip,
            user_id=user_id, client=client, dkey_id=dkey_id,
            reasoning_policy=policy,
        )
        return JSONResponse(content=out, headers={
            "x-request-id": request_id,
            "x-relay-platform": upstream.platform,
            "x-relay-model": up_model,
        })

    # ---------------- 其它透传（embeddings 等） ----------------
    async def passthrough_json(self, *, path_suffix: str, body: dict, upstream: Upstream,
                               client_model: str, request_id: str, started_at: float,
                               user_id: int | None = None, client: str = "",
                               dkey_id: int | None = None) -> Any:
        url = upstream.base_url + path_suffix
        payload = dict(body)
        payload["model"] = client_model
        resp = await self.send(self.tenants.get(user_id) if user_id is not None else None,
                               upstream, url, payload, stream=False)
        try:
            data = resp.json()
        finally:
            with contextlib.suppress(Exception):
                await resp.aclose()
        usage = T.usage_from_openai((data or {}).get("usage") if isinstance(data, dict) else None)
        self.record(
            resp=resp,
            request_id=request_id, platform=upstream.platform, model=client_model,
            model_alias=client_model, in_format="openai", stream=False, status="ok",
            http_status=200, latency_ms=int((time.perf_counter() - started_at) * 1000),
            prompt_tokens=usage.prompt_tokens, completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens, estimated=usage.total_tokens == 0,
            user_id=user_id, client=client or "master", dkey_id=dkey_id,
        )
        return JSONResponse(content=data, headers={"x-request-id": request_id})
