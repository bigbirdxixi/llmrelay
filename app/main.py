"""HTTP 入口：下游 OpenAI / Anthropic 双协议 + 管理 API + Web 控制台。"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from . import config
from . import translate as T
from .admin import router as admin_router
from .auth import _bearer as bearer_token
from .auth import authenticate, current_user, tenant_of
from .crypto import KeyCipher, resolve_master_secret
from .tenants import (TenantRegistries, has_encrypted_tenant_files,
                      migrate_one_tenant_file, migrate_tenant_files)
from .gateway import Gateway, error_response, new_request_id
from .registry import Registry
from .storage import Store

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

store = Store(config.DB_FILE)
tenants = TenantRegistries(config.CONFIG_DIR / "tenants", config.UPSTREAMS_FILE, store)
gw = Gateway(tenants, store)
registry = tenants          # 兼容老代码/测试里的 registry.reload()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 密钥加密（v1.1.0 下游 key / v1.1.1 租户上游 key）：先定主密钥再开服务——
    # ① 已有加密材料（DB 加密 key 或租户配置加密 key）而 S 找不到了 → 拒绝启动（fail-fast）
    # ② 全新部署（无任何加密材料）且没配 S → 自动生成 config/master.key（0600）并大声警告
    _extra_enc = has_encrypted_tenant_files(config.CONFIG_DIR / "tenants", config.UPSTREAMS_FILE)
    _secret = resolve_master_secret(store, extra_encrypted=_extra_enc)
    _cipher = KeyCipher(_secret)
    store.set_key_cipher(_cipher)
    _n = store.migrate_keys()
    if _n:
        log.warning("已把 %d 把历史明文 API Key 迁移为加密存储（key_mac/key_enc）", _n)
    # v1.1.1：各租户上游配置里的明文 key 一次性加密（K_t 按 uid 派生，幂等）
    tenants.set_secret(_secret)
    # 老单体配置若还没被认领且已有第一个管理员 → 立即认领（拷贝会在 K_uid 域归一化）
    if tenants.legacy_file is not None and tenants.legacy_file.exists() and not tenants.claimed:
        _fa = store.first_admin()
        if _fa:
            tenants.get(int(_fa["id"]))
    _tn = migrate_tenant_files(config.CONFIG_DIR / "tenants", _secret)
    # 老 legacy 文件按 K_0 域加密（幂等；认领后它是历史遗留，get(0) 兜底实例与它同域自洽）
    if tenants.legacy_file is not None and tenants.legacy_file.exists():
        migrate_one_tenant_file(tenants.legacy_file, 0, _secret)
    if _tn:
        log.warning("已把 %d 份租户上游配置迁移为加密存储（enc_keys）", _tn)

    await gw.startup()
    watcher = asyncio.create_task(tenants.watch(5.0))
    try:
        yield
    finally:
        watcher.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await watcher
        await gw.shutdown()
        store.close()


log = logging.getLogger("relay")

app = FastAPI(title="llm-relay", version=config.VERSION, lifespan=lifespan,
              description="本地 LLM token 中转站：多上游 · 双协议 · 用量统计")
app.state.registry = tenants
app.state.tenants = tenants
app.state.store = store
app.state.gw = gw
app.state.probes = {}   # 平台「测试」探测任务：{platform: SimpleNamespace(cancel=asyncio.Event)}

app.add_middleware(
    CORSMiddleware, allow_origins=config.CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"],
    expose_headers=["x-request-id", "x-relay-platform", "x-relay-model"],
)


# ---------------------------------------------------------------- 反向代理子路径
# 站点挂在子目录时（nginx: location /llmrelay/ { proxy_pass http://127.0.0.1:8800/; }）
# 把 settings.json 里的 base_path 设成 /llmrelay，网关会额外接受 /llmrelay/** 的请求；
# 不带前缀的路径（本地直连）继续可用，两边都能打。
@app.middleware("http")
async def _base_path(request: Request, call_next):
    base = config.get_base_path()
    if not base:
        return await call_next(request)
    path = request.scope.get("path") or ""
    if path == base:
        # /llmrelay → /llmrelay/ ，否则控制台的相对路径会算错
        return RedirectResponse(base + "/", status_code=307)
    if path.startswith(base + "/"):
        request.scope["path"] = path[len(base):] or "/"
        request.scope["root_path"] = (request.scope.get("root_path") or "") + base
    return await call_next(request)


app.include_router(admin_router)


# ---------------------------------------------------------------- 公共
@app.get("/healthz")
async def healthz():
    users = store.count_users()
    return {
        "ok": True,
        "version": config.VERSION,
        "accounts": users,
        "requests": store.record_count(),
    }


def _mask_token(tok: str) -> str:
    tok = (tok or "").strip()
    if len(tok) <= 12:
        return f"长度 {len(tok)} / 内容 {tok!r}"
    return f"{tok[:10]}…{tok[-4:]}（长度 {len(tok)}）"


def _auth_error(fmt: str, reason: str, token: str = "", path: str = "", request=None):
    """401 时说清楚原因，并把收到的 key 打码写进日志，方便比对到底是哪一步错了。

    expired / revoked：把实际过期时间 / 禁用状态回显给下游，并把这条 401
    作为一条 error 请求记录写进请求日志（带 dkey_id 归属，日志页「下游 Key」列可查）。
    """
    audit = getattr(request.state, "auth_audit", None) if request is not None else None
    if reason == "missing":
        msg = ("缺少 API Key：把「设置 → 我的 API Key」里的 sk-relay-... 放进 "
               "Authorization: Bearer *** x-api-key 头）")
        log.warning("401 %s 没有携带 API Key", path or "-")
    elif reason == "revoked":
        msg = ("该 API Key 已被禁用（吊销/删除），立即失效："
               "请让账号所有者在控制台「设置 → 我的 API Key」里重新创建一把 key")
        log.warning("401 %s 收到已禁用（吊销）的 API Key: %s", path or "-", _mask_token(token))
    elif reason == "expired":
        exp = (audit or {}).get("expires_at") or ""
        if exp:
            msg = (f"API Key 已过期（过期时间 {exp.replace('T', ' ')[:16]}，当天仍可用、次日失效）："
                   "请让账号所有者在控制台「设置 → 我的 API Key」里把过期时间改到未来，"
                   "或删除并重新创建一把 key")
        else:
            msg = ("API Key 已过期：请让账号所有者在控制台「设置 → 我的 API Key」"
                   "里把过期时间改到未来，或删除并重新创建一把 key")
        log.warning("401 %s 收到已过期的 API Key: %s（过期时间 %s）",
                    path or "-", _mask_token(token), exp or "未知")
    else:
        msg = ("API Key 无效：该 key 不存在、已被删除，或所属账号被禁用。"
               "列表里显示的是掩码，请点「复制」拿完整 key")
        log.warning("401 %s 收到无效 API Key: %s", path or "-", _mask_token(token))
    # 过期/禁用 401 写进请求日志（下游 key 归属 + 事件原因），供审计与日志页查看
    if request is not None and audit is not None and reason in ("expired", "revoked"):
        try:
            ip = request.client.host if request.client else ""
            request.app.state.store.record(
                platform="", model="", in_format=fmt,
                status="error", http_status=401, error=msg[:800],
                client_ip=ip, user_id=audit.get("user_id"), dkey_id=audit.get("dkey_id"))
        except Exception:
            log.exception("401 审计日志写入失败")
    return error_response(fmt, 401, msg, "invalid_api_key")


async def _handle(request: Request, in_fmt: str, kind: str):
    """kind: chat -> /v1/chat/completions ；messages -> /v1/messages。"""
    user, reason = authenticate(request)
    if reason:
        return _auth_error(in_fmt, reason, bearer_token(request), str(request.url.path), request)
    reg = tenants.get(user["id"])

    started_at = time.perf_counter()
    request_id = new_request_id()
    raw = await request.body()
    try:
        import json as _json
        body = _json.loads(raw) if raw else {}
    except Exception:
        return error_response(in_fmt, 400, "请求体不是合法 JSON", "invalid_json")
    if not isinstance(body, dict):
        return error_response(in_fmt, 400, "请求体必须是 JSON 对象", "invalid_json")

    model = str(body.get("model") or "").strip()
    force = (request.headers.get("x-upstream-platform") or "").strip() or None
    # 聚合模型：裸名命中聚合配置 → 在多个「平台/模型」条目间自动轮换/切换
    if model and model in reg.aggregates and not force:
        est_prompt = T.estimate_tokens(T.flatten_messages_text(body))
        client_ip = request.client.host if request.client else ""
        return await gw.relay_aggregate(
            in_fmt=in_fmt, kind=kind, body=body, agg_name=model, client_model=model,
            request_id=request_id, started_at=started_at, stream=bool(body.get("stream")),
            est_prompt=est_prompt, client_ip=client_ip,
            user_id=user["id"], client=user.get("email") or "", dkey_id=user.get("_key_id"),
        )
    resolved = reg.resolve(model, force)
    if not resolved:
        avail = reg.all_model_ids()[:20]
        return error_response(
            in_fmt, 404,
            f"没有找到模型 {model!r} 对应的上游平台。请用 'platform/model' 形式，"
            f"或先在控制台为该平台登记这个模型。当前可用示例：{', '.join(avail) or '（还没有配置上游）'}",
            "model_not_found",
        )
    upstream, up_model = resolved
    est_prompt = T.estimate_tokens(T.flatten_messages_text(body))
    client_ip = request.client.host if request.client else ""
    return await gw.relay(
        in_fmt=in_fmt, kind=kind, body=body, upstream=upstream, up_model=up_model,
        client_model=model or up_model, request_id=request_id, started_at=started_at,
        stream=bool(body.get("stream")), est_prompt=est_prompt, client_ip=client_ip,
        user_id=user["id"], client=user.get("email") or "", dkey_id=user.get("_key_id"),
    )


# ---------------------------------------------------------------- OpenAI 兼容入口
@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    return await _handle(request, "openai", "chat")


@app.post("/v1/completions")
async def completions(request: Request):
    """旧版 text completion：仅支持 OpenAI 兼容上游，直接透传。"""
    _u, reason = authenticate(request)
    if reason:
        return _auth_error("openai", reason, bearer_token(request), str(request.url.path), request)
    return await _handle(request, "openai", "chat")


@app.post("/v1/embeddings")
async def embeddings(request: Request):
    _u, reason = authenticate(request)
    if reason:
        return _auth_error("openai", reason, bearer_token(request), str(request.url.path), request)
    user, reason = authenticate(request)
    if reason:
        return _auth_error("openai", reason, bearer_token(request), str(request.url.path), request)
    reg = tenants.get(user["id"])
    started_at = time.perf_counter()
    request_id = new_request_id()
    try:
        body = await request.json()
    except Exception:
        return error_response("openai", 400, "请求体不是合法 JSON", "invalid_json")
    model = str(body.get("model") or "")
    resolved = reg.resolve(model, (request.headers.get("x-upstream-platform") or "").strip() or None)
    if not resolved:
        return error_response("openai", 404, f"没有找到模型 {model!r} 对应的上游平台", "model_not_found")
    upstream, up_model = resolved
    if upstream.type != "openai":
        return error_response("openai", 400, "embeddings 只支持 OpenAI 兼容上游", "unsupported")
    try:
        return await gw.passthrough_json(
            path_suffix="/embeddings", body=body, upstream=upstream,
            client_model=up_model, request_id=request_id, started_at=started_at,
            user_id=user["id"], client=user.get("email") or "", dkey_id=user.get("_key_id"),
        )
    except Exception as exc:  # noqa: BLE001
        return error_response("openai", 502, f"上游调用失败: {exc}", "upstream_error")


# ---------------------------------------------------------------- Anthropic 兼容入口
@app.post("/v1/messages")
async def messages(request: Request):
    return await _handle(request, "anthropic", "messages")


@app.post("/v1/messages/count_tokens")
async def count_tokens(request: Request):
    user, reason = authenticate(request)
    if reason:
        return _auth_error("anthropic", reason, bearer_token(request), str(request.url.path), request)
    reg = tenants.get(user["id"])
    try:
        body = await request.json()
    except Exception:
        return error_response("anthropic", 400, "请求体不是合法 JSON", "invalid_json")
    model = str(body.get("model") or "")
    resolved = reg.resolve(model, (request.headers.get("x-upstream-platform") or "").strip() or None)
    if not resolved:
        return error_response("anthropic", 404, f"没有找到模型 {model!r} 对应的上游平台", "model_not_found")
    upstream, up_model = resolved
    if upstream.type == "anthropic":
        payload = {**body, "model": up_model}
        try:
            resp = await gw.send(reg, upstream, upstream.url_for("count_tokens"), payload,
                                 stream=False)
        except Exception as exc:  # noqa: BLE001
            return error_response("anthropic", 502, f"上游调用失败: {exc}", "upstream_error")
        try:
            return JSONResponse(content=resp.json())
        finally:
            with contextlib.suppress(Exception):
                await resp.aclose()
    return JSONResponse(content={
        "input_tokens": T.estimate_tokens(T.flatten_messages_text(body)),
        "estimated": True,
    })


# ---------------------------------------------------------------- 模型列表
def _is_anthropic_client(request: Request) -> bool:
    return bool(request.headers.get("anthropic-version")) or (
        "x-api-key" in request.headers and "authorization" not in request.headers
    )


@app.get("/v1/models")
async def list_models(request: Request):
    user, reason = authenticate(request)
    if reason:
        return _auth_error("anthropic" if _is_anthropic_client(request) else "openai",
                           reason, bearer_token(request), str(request.url.path), request)
    reg = tenants.get(user["id"])
    now = int(time.time())
    # 同时给 "platform/model" 和裸模型名（裸名由 registry.resolve 按登记顺序兜底）
    items: list[dict] = []
    seen: set[str] = set()
    for it in reg.model_index():
        if not it["enabled"]:
            continue
        for mid in (it["id"], it["model"]):
            if mid in seen:
                continue
            seen.add(mid)
            items.append({**it, "id": mid})
    # 聚合模型：以裸名暴露（owned_by=aggregation）
    for ag in reg.aggregate_info():
        if ag["name"] not in seen:
            seen.add(ag["name"])
            items.append({"id": ag["name"], "platform": "aggregation",
                         "provider_name": f"聚合模型 · {ag['usable']}/{len(ag['entries'])} 可用",
                         "model": ag["name"], "type": "openai", "enabled": ag["usable"] > 0})
    if _is_anthropic_client(request):
        return JSONResponse(content={
            "data": [{"id": it["id"], "display_name": it["id"], "type": "model",
                      "created_at": "2025-01-01T00:00:00Z"} for it in items],
            "has_more": False,
            "first_id": items[0]["id"] if items else None,
            "last_id": items[-1]["id"] if items else None,
        })
    return JSONResponse(content={
        "object": "list",
        "data": [{"id": it["id"], "object": "model", "created": now,
                  "owned_by": it["platform"]} for it in items],
    })


@app.get("/v1/models/{model_id:path}")
async def get_model(model_id: str, request: Request):
    user, reason = authenticate(request)
    if reason:
        return _auth_error("openai", reason, bearer_token(request), str(request.url.path), request)
    for it in tenants.get(user["id"]).model_index():
        if it["id"] == model_id:
            return JSONResponse(content={"id": it["id"], "object": "model",
                                         "created": int(time.time()), "owned_by": it["platform"]})
    return error_response("openai", 404, f"没有模型 {model_id}", "model_not_found")


# ---------------------------------------------------------------- 控制台静态页
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
async def index():
    page = STATIC_DIR / "index.html"
    if page.exists():
        # no-store：控制台页永远取最新——服务重启后旧标签页残留的老 JS（探测进度/取消逻辑）
        # 会与服务端新行为对不上（症状：探测卡「开始…」、取消无效），刷新即愈
        return FileResponse(str(page), headers={"Cache-Control": "no-store"})
    return PlainTextResponse("llm-relay 已启动，但 app/static/index.html 缺失。", status_code=500)


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return PlainTextResponse("", status_code=204)
