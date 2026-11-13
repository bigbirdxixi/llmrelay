"""管理 API：上游平台 CRUD、模型探测、用量统计、设置。"""
from __future__ import annotations

import asyncio
import contextlib
import html as html_mod
import json
import time
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, StreamingResponse
import httpx

from . import auth as A
from . import config
from . import geoip as G
from . import mailer as M
from .gateway import UpstreamError
from .registry import Upstream, fetch_models

router = APIRouter(prefix="/api")


def auth(request: Request) -> None:
    """控制台鉴权：会话 Cookie 或 Bearer（账号 API Key）。"""
    A.current_user(request)


def _user(request: Request) -> dict:
    return A.current_user(request)


def _reg(request: Request):
    """当前请求操作的账号配置：管理员可以用 ?tenant=<id> 管别人的平台。"""
    return request.app.state.tenants.get(A.tenant_of(request) or _user(request)["id"])


def _email_map(store) -> dict[str, str]:
    """user_id -> 邮箱，用于把「按账号」聚合里的 id 换成人看得懂的名字。"""
    return {str(u["id"]): u["email"] for u in store.list_users()}


def _data_user(request: Request) -> int | None:
    """报表看谁的数据：普通账号=自己；管理员不带参数=自己，tenant=all 看全部。

    tenant=0 是「还没注册任何账号」的历史租户，它的数据 user_id 为 NULL，
    所以这里返回 None（不过滤）——注册第一个账号时这些记录会被接管。
    """
    uid = A.tenant_of(request)
    return None if uid == 0 else uid


def _mask(key: str) -> str:
    if len(key) <= 12:
        return "****"
    return f"{key[:6]}…{key[-4:]}"


def _is_mask(key: str) -> bool:
    return "…" in (key or "") or "****" in (key or "")


PRESETS: list[dict[str, Any]] = [
    {"platform": "deepseek", "name": "DeepSeek", "type": "openai",
     "base_url": "https://api.deepseek.com/v1",
     "models": ["deepseek-chat", "deepseek-reasoner"],
     "extra": {}, "thinking_off": "none",
     "note": "deepseek-reasoner 始终思考、没有关闭参数，要快就换 deepseek-chat"},
    {"platform": "siliconflow", "name": "硅基流动", "type": "openai",
     "base_url": "https://api.siliconflow.cn/v1",
     "models": ["deepseek-ai/DeepSeek-V3", "Qwen/Qwen2.5-72B-Instruct", "01ai/Yi-Lightning"],
     "extra": {}, "thinking_off": "enable_thinking",
     "note": "API Key 在 cloud.siliconflow.cn 控制台创建，注册送免费额度，目录里部分模型 0 元可试用；模型名是 组织/模型名 形式，保存后点「重新拉取」选"},
    {"platform": "moonshot", "name": "Kimi / Moonshot", "type": "openai",
     "base_url": "https://api.moonshot.cn/v1",
     "models": ["kimi-k2-0905-preview", "moonshot-v1-8k"],
     "extra": {}, "thinking_off": "none",
     "note": "API Key 在 platform.moonshot.cn「API Key 管理」创建（sk-… 前缀）；Kimi K2 系列有免费额度；思考靠换模型（-thinking 变体），没有请求内开关；旧版 moonshot-v1 按输入长度选 8k/32k/128k 档位"},
    {"platform": "zhipu", "name": "智谱 GLM", "type": "openai",
     "base_url": "https://open.bigmodel.cn/api/paas/v4",
     "models": ["glm-4.6", "glm-4-flash"],
     "extra": {}, "thinking_off": "thinking_type",
     "note": "API Key 在 bigmodel.cn 控制台创建。注意：智谱 key 是两段用点连接（xxxx.yyyy），完整复制、别丢点，丢点后 401；glm-4-flash 免费。GLM-4.6 默认开思考，关闭要传 thinking.type=disabled；GLM-5.3 系列强制思考关不掉"},
    {"platform": "qwen", "name": "阿里百炼 / DashScope", "type": "openai",
     "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
     "models": ["qwen-max", "qwen-plus"],
     "extra": {}, "thinking_off": "enable_thinking",
     "note": "API Key 在百炼控制台（bailian.console.aliyun.com）创建（sk-… 前缀），新用户每个模型送 100 万 tokens 免费额度（90 天）。Qwen3 用 enable_thinking=false 关思考（开思考时必须 stream=true）"},
    {"platform": "openrouter", "name": "OpenRouter", "type": "openai",
     "base_url": "https://openrouter.ai/api/v1",
     "models": ["deepseek/deepseek-chat:free", "meta-llama/llama-3.3-70b-instruct:free"],
     "extra": {}, "thinking_off": "reasoning_effort",
     "note": "聚合站：一个 key 调几十家模型；带 :free 后缀的模型免费（限流较严）；key 在 openrouter.ai 创建；可在 extra 里加 http_referer / x_title 填站点名（影响排行榜归属）；国内直连需代理"},
    {"platform": "openai", "name": "OpenAI 官方", "type": "openai",
     "base_url": "https://api.openai.com/v1",
     "models": ["gpt-4.1", "o4-mini", "gpt-4o-mini"],
     "extra": {}, "thinking_off": "reasoning_effort",
     "note": "推理模型用 reasoning_effort=none 关闭；老模型本来就不思考。国内服务器直连不通，在平台编辑页「代理」一栏填出口代理（http://127.0.0.1:10808 或 socks5h://…），只对这一平台生效"},
    {"platform": "anthropic", "name": "Anthropic 官方", "type": "anthropic",
     "base_url": "https://api.anthropic.com",
     "models": ["claude-sonnet-4-5-20250929", "claude-haiku-4-5-20251001"],
     "extra": {}, "thinking_off": "thinking_type",
     "note": "关闭时显式传 thinking.type=disabled（旧模型不支持 thinking 的话把模式改成『不传字段』）"},
    {"platform": "agnes", "name": "Agnes apihub", "type": "openai",
     "base_url": "https://apihub.agnes-ai.com/v1",
     "models": ["agnes-3.0-flash"], "extra": {}, "thinking_off": "reasoning_effort",
     "note": "实测 reasoning_effort=none 有效；不要用 enable_thinking:false（会反过来开思考）"},
    {"platform": "ollama", "name": "本地 Ollama", "type": "openai",
     "base_url": "http://127.0.0.1:11434/v1", "models": [],
     "extra": {}, "thinking_off": "reasoning_effort",
     "note": "实测 reasoning_effort=none 不会报错"},
    {"platform": "modelscope", "name": "魔塔（ModelScope 推理）", "type": "openai",
     "base_url": "https://api-inference.modelscope.cn/v1", "models": [],
     "extra": {}, "thinking_off": "auto",
     "note": "ModelScope（魔塔）推理服务，OpenAI 兼容；key 在 ModelScope 站点「访问令牌」里创建，模型名是 组织/模型名 形式，保存后点「重新拉取」选模型"},
    {"platform": "huggingface", "name": "HuggingFace（Inference）", "type": "openai",
     "base_url": "https://router.huggingface.co/v1", "models": [],
     "extra": {}, "thinking_off": "auto",
     "note": "HuggingFace Inference（Inference Providers / 自建 Endpoints，OpenAI 兼容）；API Key 填 HF token（hf_… 或 hfp-…，在 HF 网站 Settings → Access tokens 创建）；保存后点「重新拉取」选模型，也可手动输入 组织名/模型名（如 Qwen/Qwen2.5-7B-Instruct）。注意：拉模型免费但实际推理走 Inference Providers 计量——账号没有预充 credits / PRO 额度时「测试」会 402，属 HF 计费问题，不是配置错"},
    {"platform": "nousresearch", "name": "Nous Research（Inference）", "type": "openai",
     "base_url": "https://inference-api.nousresearch.com/v1",
     "models": ["Hermes-4.3-36B", "Hermes-4-70B", "Hermes-4-405B"],
     "extra": {}, "thinking_off": "auto",
     "note": "Nous Research 推理 API（OpenAI 兼容，按 token 计费）；在 portal.nousresearch.com 注册并充值 credits（或订阅套餐）后创建 API Key 填入。模型列表已预置 Hermes 4 系列（128k 上下文），也可点「重新拉取」更新。注意：官方 x402 付费（Solana USDC、免 key）是中转站不支持的 beta 通道，这里必须走 API Key + 额度；账号无额度时「测试」会 402。Hermes 4 是深度思考模型：用 system prompt 触发时，思考内容会放在响应的 reasoning_content 字段"},
    {"platform": "nvidianim", "name": "NVIDIA NIM（Build）", "type": "openai",
     "base_url": "https://integrate.api.nvidia.com/v1",
     "models": ["meta/llama-3.3-70b-instruct", "nvidia/llama-3.1-nemotron-70b-instruct",
                 "nvidia/nemotron-3-super-120b-a12b", "deepseek-ai/deepseek-v3.1",
                 "qwen/qwen2.5-coder-32b-instruct"],
     "extra": {}, "thinking_off": "auto",
     "note": "NVIDIA NIM 托管推理（build.nvidia.com，OpenAI 兼容）；API Key 在 build.nvidia.com/settings/api-keys 创建（nvapi-… 前缀，只显示一次）。模型名是 组织/模型名 形式（如 meta/llama-3.3-70b-instruct），列表已预置常用的，保存后点「重新拉取」可按目录更新。免费额度：注册送 1,000 credits（可申请到 5,000），每个模型限速 40 次/分钟（超了 429）；额度耗尽 402。国内服务器直连不通时，在平台编辑页「代理」一栏填出口代理，只对这一平台生效"},
    {"platform": "google", "name": "Google AI Studio（Gemini）", "type": "openai",
     "base_url": "https://generativelanguage.googleapis.com/v1beta/openai", "models": [],
     "extra": {}, "thinking_off": "auto",
     "note": "AI Studio / Gemini API 的 OpenAI 兼容端点；key 在 aistudio.google.com 的 API key 页创建，模型名如 gemini-2.0-flash / gemini-2.5-pro；「重新拉取」的列表会自动去掉 models/ 前缀（带前缀的名字上游会 400），已存的前缀名调用时也会被自动剥掉。国内服务器直连 Google 不通时，在平台编辑页「代理」一栏填出口代理，如 http://127.0.0.1:10808 或 socks5h://127.0.0.1:10808（socks5h 表示 DNS 也走代理），只对这一平台生效。客户端发 reasoning_effort=max（OpenAI 新值）会被中转线边界自动降级为 high（Google 对 max 报 400），可用 extra.reasoning_effort_map 覆盖。注意 Gemini 思考模型的「关闭思考」走 OpenAI 系 reasoning_effort（Google 兼容端点不再收 google.thinking_config 扩展字段）：2.x/3.0~3.4 用 none 真关死，3.5+ 不认 none（400）自动给最低档 minimal，表没覆盖的新代模型 400 时自动降级 minimal/去掉字段重发一次；工具调用的 thought_signature 由中转自动补齐（历史 assistant 工具调用缺签名时填官方哨兵值，避免 400）；不想自动补可 extra.thought_signature_bypass=false" },
    {"platform": "volcark", "name": "火山方舟（豆包）", "type": "openai",
     "base_url": "https://ark.cn-beijing.volces.com/api/v3",
     "models": ["deepseek-v3.2", "doubao-seed-1-6", "doubao-1.5-pro"],
     "extra": {}, "thinking_off": "auto",
     "note": "火山方舟（豆包 Doubao 系列 + 托管 DeepSeek 等，OpenAI 兼容）；API Key 在火山引擎控制台「火山方舟 → API Key 管理」创建。模型名可直接用模型名（如 deepseek-v3.2）或在线推理接入点 ID（ep-xxxx，控制台创建后填入，可带专属限流/计费），保存后点「重新拉取」核对。各模型有免费额度（新模型通常 50 万 tokens），超出按量计费；深度思考模型（豆包思考 / DeepSeek-R1）思考内容在 reasoning_content 字段"},
    {"platform": "stepfun", "name": "阶跃星辰（StepFun）", "type": "openai",
     "base_url": "https://api.stepfun.com/v1",
     "models": ["step-2.1-16k", "step-2-flash"],
     "extra": {}, "thinking_off": "auto",
     "note": "阶跃星辰 StepFun（Step 系列，OpenAI 兼容）；API Key 在 platform.stepfun.com 创建。step-2.1 是深度思考模型（思考内容在 reasoning_content，可用 thinking 参数开关）；模型名保存后点「重新拉取」选"},
    {"platform": "xfyun", "name": "讯飞星火（Spark）", "type": "openai",
     "base_url": "https://spark-api-open.xf-yun.com/v1",
     "models": ["lite", "generalv3.5", "max-32k", "4.0Ultra"],
     "extra": {}, "thinking_off": "auto",
     "note": "讯飞星火 OpenAI 兼容端点；鉴权用星火开放平台（xfyun.cn）控制台创建的应用 APIPassword（不是 AppId）。lite 长期免费，max/pro 等按量。注意：spark-x 推理模型走不同路径（/v2、/x2），本模板的 /v1 端点不覆盖，推理模型需在平台编辑页手动改 Base URL"},
    {"platform": "minimax", "name": "MiniMax", "type": "openai",
     "base_url": "https://api.minimaxi.com/v1",
     "models": ["MiniMax-M3", "MiniMax-M3.1-Flash", "MiniMax-M2.7-highspeed"],
     "extra": {}, "thinking_off": "auto",
     "note": "MiniMax 国内端点（OpenAI 兼容；海外账号把 Base URL 换成 https://api.minimax.io/v1）。API Key 在 platform.minimaxi.com 创建。M 系列支持 1M 上下文，思考深度可用 reasoning_effort 调（max/high/medium/low）；模型名保存后点「重新拉取」选"},
    {"platform": "xai", "name": "xAI（Grok）", "type": "openai",
     "base_url": "https://api.x.ai/v1",
     "models": ["grok-4", "grok-3", "grok-2-1212"],
     "extra": {}, "thinking_off": "auto",
     "note": "xAI Grok 官方 API（OpenAI 兼容）；key 在 console.x.ai 创建；国内直连需代理（平台编辑页「代理」栏）"},
    {"platform": "mistral", "name": "Mistral", "type": "openai",
     "base_url": "https://api.mistral.ai/v1",
     "models": ["mistral-large-latest", "mistral-small-latest", "codestral-latest"],
     "extra": {}, "thinking_off": "auto",
     "note": "Mistral 官方 API（OpenAI 兼容）；key 在 console.mistral.ai 创建，部分模型有免费额度；国内直连需代理"},
]


# ---------------------------------------------------------------- 概览
@router.get("/overview")
async def overview(request: Request, _: None = Depends(auth)):
    reg = _reg(request)
    store = request.app.state.store
    return {
        "version": config.VERSION,
        "totals": store.totals(_data_user(request)),
        "providers": len(reg.upstreams),
        "enabled_providers": len(reg.enabled()),
        "models": len(reg.all_model_ids()),
        "records": store.record_count(user_id=_data_user(request)),
        "today_bucket": store.summary("day", 1, _data_user(request))[:1],
        "paths": {"db": str(config.DB_FILE), "upstreams": str(config.UPSTREAMS_FILE)},
        "host": config.HOST, "port": config.PORT,
    }


# ---------------------------------------------------------------- 上游平台
def _public(up: Upstream) -> dict:
    data = up.model_dump()
    data["api_key"] = ""
    data["api_keys"] = [_mask(k) for k in up.api_keys]
    data["key_count"] = len(up.api_keys)
    data["urls"] = {"chat": up.url_for("chat"), "messages": up.url_for("messages"),
                    "models": up.url_for("models")}
    return data


@router.get("/providers")
async def list_providers(request: Request, _: None = Depends(auth)):
    reg = _reg(request)
    return {"data": [_public(u) for u in reg.upstreams], "presets": PRESETS}


@router.get("/aggregates")
async def list_aggregates(request: Request, _: None = Depends(auth)):
    reg = _reg(request)
    return {"data": reg.aggregate_info()}


@router.post("/aggregates")
async def save_aggregate(request: Request, _: None = Depends(auth)):
    reg = _reg(request)
    data = await request.json()
    name = str(data.get("name") or "").strip()
    entries = [str(e).strip() for e in (data.get("entries") or []) if str(e).strip()]
    import re as _re
    if not _re.fullmatch(r"[A-Za-z0-9._-]{2,64}", name):
        raise HTTPException(400, "聚合名只允许字母/数字/._-，2-64 位")
    if "/" in name:
        raise HTTPException(400, "聚合名不能包含 /（会被当成平台/模型前缀）")
    if not entries:
        raise HTTPException(400, "至少需要一个「平台/模型」条目")
    if len(entries) > 20:
        raise HTTPException(400, "单个聚合最多 20 个条目")
    if len(set(entries)) != len(entries):
        raise HTTPException(400, "条目有重复")
    bad = [e for e in entries if not reg.resolve(e)]
    if bad:
        raise HTTPException(400, f"这些条目解析不到已启用的上游模型（平台未启用或模型未登记）：{', '.join(bad)}")
    reg.upsert_aggregate(name, entries)
    return {"name": name, "entries": entries}


@router.delete("/aggregates/{name}")
async def delete_aggregate(name: str, request: Request, _: None = Depends(auth)):
    reg = _reg(request)
    if not reg.remove_aggregate(name):
        raise HTTPException(404, f"聚合模型 {name!r} 不存在")
    return {"ok": True}


@router.get("/providers/reasoning-preview")
async def reasoning_preview(request: Request, _: None = Depends(auth),
                            platform: str = "", model: str = "",
                            out_fmt: str = "openai", field: str = "auto",
                            base_url: str = "", extra: str = ""):
    """「关闭思考」解析预览：中转实际发给上游的顶层字段 + 说明。

    UI 在模型级 off 行下面展示它——用户只选「关闭思考」就不用管写法（Hermes 单开关体验）。
    正在编辑的新平台可以带表单里的 base_url/extra(JSON) 直接预览；已存平台可省略。
    走的是和 gateway 完全相同的解析代码（_thinking_off_payload + extra_body 摊平），预览即实发。
    """
    from . import reasoning as R
    from .gateway import Gateway
    ex = {}
    if extra:
        with contextlib.suppress(json.JSONDecodeError):
            ex = json.loads(extra) or {}
    reg = _reg(request)
    row = next((u for u in reg.upstreams if u.platform == platform), None)
    if not extra and row is not None:
        ex = row.extra or {}
    up = SimpleNamespace(platform=platform, name=platform,
                         base_url=base_url or (row.base_url if row else ""),
                         extra=ex or {}, type="openai")
    wire = dict(Gateway._thinking_off_payload(up, out_fmt, field=field, model=model))
    if out_fmt == "openai" and isinstance(wire.get("extra_body"), dict):
        eb = wire.pop("extra_body")
        for k, v in eb.items():
            wire.setdefault(k, v)
    m = (model or "").lower()
    looks_google = bool(R.is_gemini_family(up)) or "gemini" in m
    sig = looks_google and ("gemini-3" in m or "gemini-4" in m)
    notes = []
    if sig:
        notes.append("Gemini 3/4 思考关不彻底（3.5+ 最低档是 minimal，2.x/3.0~3.4 可用 none 关死），"
                     "思考内容对客户端隐藏；带工具调用时历史里的 thought_signature 由中转自动补齐（官方哨兵值）")
    elif looks_google and "gemini" in m:
        notes.append("Gemini 2.x：reasoning_effort=none 可以完全关掉思考")
    if not wire and out_fmt == "anthropic":
        notes.append("Anthropic 原生协议没有「关」的写法：省略 thinking 字段本身就是关")
    return {"wire": wire, "notes": notes, "sig_autofill": sig}


@router.post("/providers")
async def upsert_provider(request: Request, _: None = Depends(auth)):
    reg = _reg(request)
    payload = await request.json()
    platform = (payload.get("platform") or "").strip()
    if not platform:
        raise HTTPException(400, "platform 不能为空")
    existing = reg.find(platform)
    keys = payload.get("api_keys")
    if isinstance(keys, str):
        keys = [k.strip() for k in keys.replace("\r", "").split("\n")]
    if isinstance(keys, list):
        keys = [k for k in keys if k and k.strip()]
        real = [k for k in keys if not _is_mask(k)]
        if not real and existing is not None:
            keys = existing.api_keys  # 前端提交的是掩码，沿用旧 key
        else:
            keys = real
    elif existing is not None:
        keys = existing.api_keys
    else:
        keys = []
    single = payload.get("api_key")
    if isinstance(single, str) and single.strip() and not _is_mask(single):
        keys = [single.strip()] + [k for k in keys if k != single.strip()]

    proxy_raw = payload.get("proxy") if "proxy" in payload and payload.get("proxy") is not None \
        else ((existing.proxy if existing else "") or "")
    proxy = str(proxy_raw or "").strip()
    if proxy:
        scheme = (urlparse(proxy).scheme or "").lower()
        if scheme not in ("http", "https", "socks5", "socks5h"):
            raise HTTPException(400, "代理只支持 http/https/socks5/socks5h 地址，如 http://127.0.0.1:10808")

    row = {
        "platform": platform,
        "name": (payload.get("name") if payload.get("name") is not None
                 else (existing.name if existing else "")) or "",
        "type": payload.get("type") or (existing.type if existing else "openai"),
        "base_url": (payload.get("base_url") or (existing.base_url if existing else "")).strip(),
        "api_keys": keys,
        "models": payload.get("models") if isinstance(payload.get("models"), list)
        else (existing.models if existing else []),
        "model_map": payload.get("model_map") if isinstance(payload.get("model_map"), dict)
        else (existing.model_map if existing else {}),
        "extra": payload.get("extra") if isinstance(payload.get("extra"), dict)
        else (existing.extra if existing else {}),
        "proxy": proxy,
        "enabled": bool(payload.get("enabled", existing.enabled if existing else True)),
        "timeout_s": float(payload.get("timeout_s") or (existing.timeout_s if existing else 300)),
    }
    if not row["base_url"]:
        raise HTTPException(400, "base_url 不能为空")
    if row["type"] not in ("openai", "anthropic"):
        raise HTTPException(400, "type 只能是 openai / anthropic")
    try:
        saved = reg.upsert(row)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"保存失败: {exc}") from exc
    return {"ok": True, "data": _public(Upstream.model_validate(saved))}


@router.delete("/providers/{platform}")
async def delete_provider(platform: str, request: Request, _: None = Depends(auth)):
    reg = _reg(request)
    up = reg.find(platform)
    if not reg.remove(platform):
        raise HTTPException(404, f"平台 {platform} 不存在")
    return {"ok": True}


@router.post("/providers/{platform}/refresh-models")
async def refresh_models(platform: str, request: Request, _: None = Depends(auth)):
    reg = _reg(request)
    gw = request.app.state.gw
    up = reg.find(platform)
    if not up:
        raise HTTPException(404, f"平台 {platform} 不存在")
    if not up.api_keys:
        raise HTTPException(400, "该平台还没有配置 api key")
    try:
        models = await fetch_models(up, gw.client_for(up))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(502, f"拉取模型失败: {exc}") from exc
    if models:
        reg.set_models(platform, models)
    return {"ok": True, "count": len(models), "models": models}


@router.get("/providers/{platform}/available-models")
async def available_models(platform: str, request: Request, _: None = Depends(auth)):
    """拉取上游模型列表但**不写回配置**（编辑平台里模型下拉框用）。"""
    reg = _reg(request)
    gw = request.app.state.gw
    up = reg.find(platform)
    if not up:
        raise HTTPException(404, f"平台 {platform} 不存在")
    if not up.api_keys:
        raise HTTPException(400, "该平台还没有配置 api key")
    try:
        models = await fetch_models(up, gw.client_for(up))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(502, f"拉取模型失败: {exc}") from exc
    return {"ok": True, "count": len(models), "models": models}


@router.post("/providers/{platform}/test")
async def test_provider(platform: str, request: Request, _: None = Depends(auth)):
    """真发一条最小请求，验证 key / base_url / 协议是否可用。"""
    reg = _reg(request)
    gw = request.app.state.gw
    up = reg.find(platform)
    if not up:
        raise HTTPException(404, f"平台 {platform} 不存在")
    if not up.api_keys:
        raise HTTPException(400, "该平台还没有配置 api key")
    body = await request.json() if await request.body() else {}
    model = (body.get("model") or "").strip()
    if not model:
        # 没指定模型时按登记顺序逐个试（最多 5 个）——NVIDIA NIM 这类按账号授权模型清单的
        # 平台，登记列表里的首个模型未必对该 key 开放（404 "Function not found for account"），
        # 试到第一个通的才算平台可用。
        candidates = up.models[:5]
        if not candidates:
            raise HTTPException(400, "该平台没有登记模型，先点『拉取模型』或用 ?model= 指定")
    else:
        candidates = [model]

    url = up.url_for("chat" if up.type == "openai" else "messages")
    tried: list[str] = []
    last_err = ""
    for cand in candidates:
        if up.type == "openai":
            payload: dict[str, Any] = {
                "model": cand, "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 1, "stream": False,
            }
        else:
            payload = {"model": cand, "max_tokens": 1,
                       "messages": [{"role": "user", "content": [{"type": "text", "text": "ping"}]}]}
        t0 = time.perf_counter()
        try:
            resp = await gw.send(_reg(request), up, url, payload, stream=False)
        except Exception as exc:  # noqa: BLE001
            tried.append(cand)
            last_err = f"{cand}: {str(exc)[:200]}"
            continue
        try:
            data = resp.json()
        finally:
            with contextlib.suppress(Exception):
                await resp.aclose()
        usage = data.get("usage") or {}
        model = cand
        if len(tried) > 1:
            tried = []
        break
    else:
        return {"ok": False, "latency_ms": 0,
                "error": f"试了 {len(candidates)} 个模型都不可用。" + (last_err or "")[:400]}
    return {
        "ok": True, "latency_ms": int((time.perf_counter() - t0) * 1000),
        "model": model, "http_status": resp.status_code,
        "usage": usage, "sample": str(data)[:400],
        **({"tried": tried} if len(tried) > 1 else {}),
    }


# 探测 ping 专用短超时：连不通的上游（如本机需要代理才到得了 api.nvidia.com）
# 不能按 300s 平台超时干等——每个模型 20s 封顶，连不通时第一个模型就快速放弃整轮探测。
PROBE_PING_TIMEOUT = httpx.Timeout(20.0, connect=10.0)


async def _client_gone_async(request: Request) -> bool:
    """客户端是否已断开（点取消/关页面）。uvicorn 客户端断开后不会取消 app task、
    只是静默吞掉后续 send，所以探测循环必须自己查，否则「取消」后服务端还在空跑。
    0.1s 超时兜底（TestClient 语义下 receive 会挂起，不能无限等）。"""
    try:
        return await asyncio.wait_for(request.is_disconnected(), 0.1)
    except Exception:  # noqa: BLE001
        return False


async def _probe_model(gw, reg, up, url: str, m: str) -> tuple:
    """ping 单个模型，返回 (result, code, reason)；result ∈ ok/removed/transient/keyerr/connfail。"""
    if up.type == "openai":
        payload: dict[str, Any] = {"model": m,
                                   "messages": [{"role": "user", "content": "ping"}],
                                   "max_tokens": 1, "stream": False}
    else:
        payload = {"model": m, "max_tokens": 1,
                  "messages": [{"role": "user", "content": [{"type": "text", "text": "ping"}]}]}
    try:
        resp = await gw.send(reg, up, url, payload, stream=False,
                             timeout=PROBE_PING_TIMEOUT)
        with contextlib.suppress(Exception):
            await resp.aclose()
        return "ok", resp.status_code, ""
    except UpstreamError as exc:
        if "连接上游失败" in exc.body:      # 连接级失败（超时/拒绝/无代理），不是上游真实 HTTP 响应
            return "connfail", 0, exc.body[:160]
        if exc.status in (401, 403):
            return "keyerr", exc.status, exc.message()[:160]
        if exc.status in (404, 400):
            return "removed", exc.status, exc.message()[:160]
        return "transient", exc.status, exc.message()[:160]
    except Exception as exc:  # noqa: BLE001
        return "connfail", 0, str(exc)[:160]


async def _key_alive(gw, up) -> bool:
    """探测「key 本身是否可用」：用同一 client（同代理）GET {base_url}/models。

    NVIDIA NIM 的 chat 401「Unauthorized access」可能是单模型（function）级未授权，
    而 /models 目录接口仍 200——这种情况不是 key 坏了，不该整轮叫 key 级错误。"""
    cli = gw.client_for(up)
    try:
        resp = await cli.get(up.base_url.rstrip("/") + "/models",
                             headers={"Authorization": "***" + up.api_keys[0]},
                             timeout=httpx.Timeout(10.0, connect=10.0))
        return resp.status_code < 400
    except Exception:  # noqa: BLE001
        return False


@router.post("/providers/{platform}/probe")
async def probe_provider(platform: str, request: Request, _: None = Depends(auth)):
    """探测全部已登记模型并把可用清单写回：每个模型 ping 一条（max_tokens=1）。
    - 2xx（含网关瞬时重试后成功）→ 保留；
    - 模型级 404/400（NVIDIA NIM「Function not found for account」= 账号未授权这类）→ 移除并写回；
    - 429/5xx（重试后仍失败，疑似瞬时故障）→ 保留不误删，transient 里列出；
    - key 级 401/403 → 提前停止（剩下的测了也没意义）、一律保留不动，只报 key_error。
    被移除的模型随时可「拉取模型」恢复（拉的是全量目录）。
    body {"stream": true} 时返回 NDJSON 实时进度流（每行一个模型结果，末行 done + 汇总）。"""
    reg = _reg(request)
    gw = request.app.state.gw
    up = reg.find(platform)
    if not up:
        raise HTTPException(404, f"平台 {platform} 不存在")
    if not up.api_keys:
        raise HTTPException(400, "该平台还没有配置 api key")
    models = up.models[:300]
    if not models:
        raise HTTPException(400, "该平台没有登记模型，先点『拉取模型』")
    body = {}
    raw_body = await request.body()
    if raw_body:
        with contextlib.suppress(json.JSONDecodeError):
            body = json.loads(raw_body) or {}
    url = up.url_for("chat" if up.type == "openai" else "messages")
    # 探测任务句柄：显式「停止」（POST /probe/stop）置 cancel 事件；
    # uvicorn 客户端断开不会取消 app task，is_disconnected 只作尽力而为的补充检查
    job = SimpleNamespace(cancel=asyncio.Event())
    request.app.state.probes[platform] = job

    async def probe_events():
        kept, removed, transient = [], [], []
        connfails: list[str] = []
        key_error, connect_fail, cancelled, skipped = False, False, False, 0
        conn_reason, consec_fail = "", 0
        total = len(models)
        try:
          for i, m in enumerate(models):
            yield {"i": i + 1, "total": total, "model": m, "phase": "start"}
            res, code, reason = await _probe_model(gw, reg, up, url, m)
            if res == "keyerr":
                # 401/403 先分清是 key 坏了还是「这个模型没权限」：
                # 同一 key 能列出 /models 目录 → key 没问题，单模型未授权 → 按「移除」处理继续探测
                if await _key_alive(gw, up):
                    res, reason = "removed", f"{code}: 无此模型授权（key 可用，该模型未开通）"
            if res == "ok":
                kept.append(m)
                consec_fail = 0
            elif res == "removed":
                removed.append({"model": m, "reason": reason})
                consec_fail = 0
            elif res == "transient":
                transient.append(m)
                kept.append(m)
                consec_fail = 0
            elif res == "connfail":
                # 连不通（超时/拒绝/代理抖动）：模型保留不删；
                # 容忍前 2 个（代理可能瞬时抖动），连续 3 个才算上游整体不可达 → 整轮快速放弃
                connfails.append(m)
                kept.append(m)
                consec_fail += 1
                if consec_fail >= 3:
                    connect_fail = True
                    conn_reason = reason
                    skipped = total - i - 1
                    break
            else:  # keyerr：确认 key 级错误，后面全都会 401/403，提前收工
                key_error = True
                kept.append(m)
                skipped = total - i - 1
                break
            yield {"i": i + 1, "total": total, "model": m,
                   "result": res, "code": code, "reason": reason}
            await asyncio.sleep(0.3)   # 限流余量（NVIDIA 免费档 40 RPM）
            if job.cancel.is_set() or await _client_gone_async(request):
                # 用户点了「取消」（显式 stop）或客户端已断开：停止继续 ping，也不写回模型清单
                cancelled = True
                skipped = total - i - 1
                break
        finally:
            # 本轮跑完（含取消/放弃）清掉 job，stop 端点对已结束探测返回 stopped:false
            if getattr(request.app.state, "probes", {}).get(platform) is job:
                del request.app.state.probes[platform]
        written = False
        if removed and not key_error and not connect_fail and not cancelled:
            reg.set_models(platform, kept)
            written = True
        yield {"done": True, "kept": kept, "removed": removed, "transient": transient,
               "connfails": connfails,
               "key_error": key_error, "connect_fail": connect_fail,
               "connect_fail_reason": conn_reason,
               "cancelled": cancelled, "skipped": skipped, "written": written}

    if body.get("stream"):
        async def gen():
            async for ev in probe_events():
                yield json.dumps(ev, ensure_ascii=False) + "\n"
        return StreamingResponse(gen(), media_type="application/x-ndjson")
    events = [ev async for ev in probe_events()]
    return events[-1]


@router.post("/providers/{platform}/probe/stop")
async def probe_stop(platform: str, request: Request, _: None = Depends(auth)):
    """显式停止某平台正在进行的探测（前端「取消」按钮调这个）。
    停止后：不再继续 ping，剩余模型记入 skipped，模型清单不写回。"""
    job = getattr(request.app.state, "probes", {}).get(platform)
    if job is None:
        return {"stopped": False}
    job.cancel.set()
    return {"stopped": True}


@router.get("/models")
async def models(request: Request, _: None = Depends(auth)):
    reg = _reg(request)
    return {"data": reg.model_index(), "ids": reg.all_model_ids()}


# ---------------------------------------------------------------- 统计
@router.get("/stats/summary")
async def stats_summary(request: Request, days: int = 30, group_by: str = "platform",
                        _: None = Depends(auth)):
    store = request.app.state.store
    uid = _data_user(request)
    rows = store.summary(group_by, days, uid)
    if group_by == "account":
        emap = _email_map(store)
        for row in rows:
            row["k"] = emap.get(str(row["k"]), row["k"])
        rows.sort(key=lambda r: r["total_tokens"], reverse=True)
    return {"days": days, "group_by": group_by, "data": rows,
            "totals": store.totals(uid), "tenant": uid}


@router.get("/stats/series")
async def stats_series(request: Request, days: int = 30, group_by: str = "platform",
                       _: None = Depends(auth)):
    store = request.app.state.store
    data = store.series(days, group_by, _data_user(request))
    if group_by == "account":
        emap = _email_map(store)
        reserved = {"day", "total", "requests"}
        data["rows"] = [
            {(k if k in reserved else emap.get(k, k)): v for k, v in row.items()}
            for row in data["rows"]
        ]
        data["keys"] = [emap.get(k, k) for k in data["keys"]]
        data["platforms"] = data["keys"]
    return data


@router.get("/stats/hourly")
async def stats_hourly(request: Request, days: int = 1, group: str = "hour",
                       split: str = "none", _: None = Depends(auth)):
    """近 N 天活动（概览看板）：group=hour 按小时（≤14 天）/ group=day 按天；
    split=platform|model 时每个桶附带该维度的 tokens 拆分。"""
    store = request.app.state.store
    days = max(1, min(int(days), 90))
    group = "day" if group == "day" else "hour"
    split = split if split in ("platform", "model", "dkey", "none") else "none"
    if group == "hour":
        days = min(days, 14)
    return store.hourly(_data_user(request), days=days, group=group, split=split)


@router.get("/stats/regions")
async def stats_regions(request: Request, range: str = "today",
                       _: None = Depends(auth)):
    """客户端 IP 地域分布（概览页「地域访问地图」）。

    range=today|7d|30d（日历天对齐，同概览时间 chip 口径）。返回：
      db: "ok" | "missing"（GeoIP 数据库未配置/加载失败时 missing，前端显示提示）
      data: [{region, kind: cn|world|local|unknown, count}] 按 count 降序
      total: 区间内带 client_ip 的请求合计
    历史数据不改 schema：按 IP 读时解析（离线 mmdb，微秒级）。
    """
    store = request.app.state.store
    days = {"today": 1, "7d": 7, "30d": 30}.get(range, 1)
    db = "ok" if G.available() else "missing"
    data: list[dict] = []
    cities: list[dict] = []
    total = 0
    if G.available():
        agg: dict[tuple[str, str], int] = {}
        city_agg: dict[tuple[str, str, float, float], int] = {}
        city_region: dict[tuple, str] = {}
        for row in store.ip_counts(_data_user(request), days):
            info = G.resolve(row["ip"]) or {"kind": "unknown", "region": "未知"}
            key = (info["kind"], info["region"])
            agg[key] = agg.get(key, 0) + row["n"]
            # 城市热点：仅当解析出城市名 + 坐标（GeoLite2 City 版）；
            # 同一城市可能出现在多省/多国家记录里，按 (kind, city_zh, lat, lon) 归并，
            # 并记住所属省（region），前端「点省下钻城市排名」用它过滤
            if info.get("city") and info.get("lat") is not None:
                ck = (info["kind"], info.get("city_zh") or info["city"],
                      round(info["lat"], 2), round(info["lon"], 2))
                city_agg[ck] = city_agg.get(ck, 0) + row["n"]
                city_region[ck] = info.get("region") or ""
        total = sum(agg.values())
        data = [{"kind": k, "region": name, "count": n}
                for (k, name), n in sorted(agg.items(), key=lambda x: -x[1])]
        cities = [{"kind": k, "region": city_region.get((k, nz, la, lo), ""),
                   "name": nz, "lat": la, "lon": lo, "count": n}
                  for (k, nz, la, lo), n in sorted(city_agg.items(), key=lambda x: -x[1])
                  ][:50]
    return {"range": range, "db": db, "data": data, "cities": cities, "total": total}


@router.get("/stats/month")
async def stats_month(request: Request, year: int, month: int,
                     _: None = Depends(auth)):
    """指定年月的按月活跃统计（概览页「本月活跃情况」日历 + 年月下拉框）。

    返回该月 1 号~月末逐日桶（补 0）、当月合计、active_days，以及该用户
    有数据的所有月份 months（供下拉框）；按 ts 年月前缀精确取数。
    """
    store = request.app.state.store
    if not (1900 <= year <= 2100 and 1 <= month <= 12):
        return JSONResponse({"error": "invalid year/month"}, status_code=400)
    return store.month_activity(_data_user(request), year, month)


@router.get("/stats/year")
async def stats_year(request: Request, year: int,
                    _: None = Depends(auth)):
    """指定年度的「星期 × 月份」活跃统计（概览页活跃度日历 + 年份下拉框）。

    纵轴周一~周日 × 横轴 1~12 月，格子 = 该月该星期几的请求量合计；
    附年度合计、active_days 与 years（有数据的年份，供下拉框）。
    """
    store = request.app.state.store
    if not (1900 <= year <= 2100):
        return JSONResponse({"error": "invalid year"}, status_code=400)
    return store.year_activity(_data_user(request), year)


@router.get("/stats/period")
async def stats_period(request: Request, days: int = 30, _: None = Depends(auth)):
    """区间 KPI（请求/tokens/P95/缓存命中/成功率/流式占比）。"""
    store = request.app.state.store
    return store.period(days, _data_user(request))


@router.get("/stats/latency")
async def stats_latency(request: Request, days: int = 30, _: None = Depends(auth)):
    """每日耗时曲线：avg / p95 / TTFT（仅成功请求）。"""
    store = request.app.state.store
    return store.latency_series(days, _data_user(request))


@router.get("/stats/heatmap")
async def stats_heatmap(request: Request, days: int = 30, _: None = Depends(auth)):
    """星期 × 小时 请求量热力矩阵。"""
    store = request.app.state.store
    return store.heatmap(days, _data_user(request))


@router.get("/stats/mix")
async def stats_mix(request: Request, days: int = 30, _: None = Depends(auth)):
    """区间结构：协议/流式/状态/思考策略/错误 HTTP 码分布。"""
    store = request.app.state.store
    return store.mix(days, _data_user(request))


@router.get("/stats/errors")
async def stats_errors(request: Request, days: int = 30, limit: int = 8,
                       _: None = Depends(auth)):
    """区间 Top 错误原因（按次数）。"""
    store = request.app.state.store
    return {"data": store.top_errors(days, limit, _data_user(request))}


@router.get("/stats/records")
async def stats_records(request: Request, limit: int = 100, offset: int = 0,
                       platform: str | None = None, model: str | None = None,
                       status: str | None = None, days: int = 0,
                       stream: str | None = None, key: str | None = None,
                       ip: str | None = None, users: str | None = None,
                       date_from: str | None = Query(None, alias="from"),
                       date_to: str | None = Query(None, alias="to"),
                       range_: str | None = Query(None, alias="range"),
                       _: None = Depends(auth)):
    """请求日志（全局筛选：每个参数可逗号分隔多选；from/to 支持 YYYY-MM-DD 或到秒的
    YYYY-MM-DDTHH:MM:SS 日历区间；range=15m/30m/1h/6h/today 相对快捷区间，锚定服务端当前时间）。"""
    store = request.app.state.store
    uid = _data_user(request)
    def _csv(v: str | None, cast=None) -> list | None:
        if not v:
            return None
        cast = cast or str
        vals = []
        for part in v.split(","):
            part = part.strip()
            if part:
                vals.append(cast(part))
        return vals or None
    plist = _csv(platform)
    mlist = _csv(model)
    slist = _csv(status)
    stream_list = _csv(stream, lambda p: int(p) if p in ("0", "1") else p)
    uid_list = _csv(users, lambda p: int(p))
    if range_:   # 相对快捷区间优先；显式 from/to 仍然可以叠加收窄
        rf, rt = store.resolve_range(range_)
        date_from = date_from or rf
        date_to = date_to or rt
    fkw = dict(stream=stream_list, key_hint=_csv(key), client_ip=_csv(ip),
               user_ids=uid_list, date_from=date_from, date_to=date_to)
    rows = store.recent(limit, plist, mlist, slist, days, uid, offset, **fkw)
    emap = _email_map(store)
    for r in rows:
        r["user_email"] = emap.get(str(r.get("user_id")), "（历史）" if r.get("user_id") is None else "")
    total = store.record_count(days, uid, platform=plist, model=mlist,
                               status=slist, **fkw)
    return {"data": rows, "total": total, "model_options": store.model_options(uid)}


@router.get("/stats/logs/facets")
async def stats_logs_facets(request: Request, _: None = Depends(auth)):
    """日志页的全局筛选项（跨全量记录，不只当前页）。"""
    store = request.app.state.store
    return store.log_facets(_data_user(request))


@router.get("/stats/export.csv")
async def stats_export(request: Request, days: int = 30, _: None = Depends(auth)):
    store = request.app.state.store
    csv_text = store.export_csv(days, _data_user(request))
    return PlainTextResponse(csv_text, media_type="text/csv; charset=utf-8", headers={
        "content-disposition": f'attachment; filename="relay-usage-{days}d.csv"',
    })


# ---------------------------------------------------------------- 设置
@router.get("/settings")
async def settings(request: Request, _: None = Depends(auth)):
    user = _user(request)
    keys = [{"id": k["id"], "key": A.mask_key(k["key"]), "label": k["label"],
             "created_at": k["created_at"], "last_used": k["last_used"],
             "revoked": bool(k["revoked"]), "expires_at": k.get("expires_at")}
            for k in request.app.state.store.list_api_keys(user["id"])]
    return {
        "version": config.VERSION,
        "user": A.public_user(user),
        "is_admin": user.get("role") == "admin",
        "api_keys": keys,
        "base_url": f"http://{config.HOST}:{config.PORT}",
        "base_path": config.get_base_path(),
        "base_path_env_locked": config.base_path_env_locked(),
    }


@router.post("/settings/base-path")
async def set_base_path(request: Request, _: None = Depends(auth)):
    """设置反向代理子路径，例如 /llmrelay（空 = 挂在根目录）。立即生效，无需重启。仅管理员。"""
    A.require_admin(request)
    body = await request.json() if await request.body() else {}
    raw = str(body.get("base_path") or "")
    if config.base_path_env_locked():
        raise HTTPException(400, "当前由环境变量 RELAY_BASE_PATH 决定，改这里不生效；先去掉环境变量")
    saved = config.set_base_path(raw)
    return {"ok": True, "base_path": saved}


# ---------------------------------------------------------------- 邮箱发信（SMTP）设置
@router.get("/settings/mail")
async def get_mail_settings(request: Request, _: None = Depends(auth)):
    """当前生效的邮箱发信配置（密码打码）。控制台设置 > 环境变量。仅管理员。"""
    A.require_admin(request)
    return M.public_mail_settings(request.app.state.store)


@router.post("/settings/mail")
async def save_mail_settings(request: Request, _: None = Depends(auth)):
    """保存邮箱发信配置。密码留空 = 保持原值（前端只能看到打码）。仅管理员。"""
    A.require_admin(request)
    store = request.app.state.store
    body = await request.json() if await request.body() else {}
    mode = str(body.get("mode") or "smtp").strip().lower()
    if mode not in ("smtp", "log"):
        raise HTTPException(400, "发信模式只能是 smtp 或 log")
    host = str(body.get("host") or "").strip()
    if host and len(host) > 253:
        raise HTTPException(400, "SMTP 主机地址太长了，检查是否填错")
    try:
        port = int(str(body.get("port") or "587").strip() or 587)
    except ValueError:
        raise HTTPException(400, "SMTP 端口必须是数字（如 465 / 587）")
    if not 1 <= port <= 65535:
        raise HTTPException(400, "SMTP 端口要在 1–65535 之间")
    if mode == "smtp" and not host:
        raise HTTPException(400, "SMTP 模式要填 SMTP 服务器主机（或改用日志模式）")
    M.save_mail_settings(store, mode, host, port,
                         user=str(body.get("user") or "").strip(),
                         password=str(body.get("password") or ""),
                         from_addr=str(body.get("from") or "").strip(),
                         starttls=body.get("starttls", True) is not False)
    return {"ok": True, **M.public_mail_settings(store)}


@router.post("/settings/mail/test")
async def test_mail_settings(request: Request, _: None = Depends(auth)):
    """发一封测试邮件验证 SMTP 配置。仅管理员。"""
    A.require_admin(request)
    store = request.app.state.store
    body = await request.json() if await request.body() else {}
    try:
        to = A.normalize_email(str(body.get("to") or ""))
    except HTTPException:
        raise HTTPException(400, "收件人邮箱格式不对")
    r = M.send_test_email(store, to)
    if not r["ok"]:
        return {"ok": False, "mode": r["mode"], "error": r["error"]}
    return {"ok": True, "mode": r["mode"], "error": r["error"],
            "message": (f"测试邮件已发往 {to}" if r["mode"] == "smtp"
                        else "当前为日志模式，没有实际发信（测试内容已写入服务器日志 [mail/log]）")}


# ================================================================ 账号
def _set_session_cookie(resp, request: Request, token: str) -> None:
    resp.set_cookie(A.COOKIE_NAME, token, max_age=A.SESSION_DAYS * 86400,
                    httponly=True, samesite="lax",
                    secure=(request.url.scheme == "https"), path="/")


@router.get("/auth/status")
async def auth_status(request: Request):
    """控制台首屏用：有没有账号、当前是否已登录。"""
    store = request.app.state.store
    user = A.optional_user(request)
    return {
        "has_users": store.count_users() > 0,
        "user": A.public_user(user) if user else None,
        "registration_open": True,   # 新账号由管理员在控制台创建；首个注册者成为管理员
    }


@router.post("/auth/register")
async def auth_register(request: Request):
    body = await request.json() if await request.body() else {}
    store = request.app.state.store
    user = A.register(store, str(body.get("email") or ""), str(body.get("password") or ""))
    base = str(request.base_url).rstrip("/")
    out = {"ok": True, "user": A.public_user(user),
           "first_admin": user["role"] == "admin", "api_key": ""}
    if user["status"] == "pending":
        # 未确认邮箱：不发会话（不能登录），发确认邮件（magic link）
        mail = A.send_verification_email(store, user, base)
        out["pending"] = True
        out["mail_mode"] = mail["mode"]
        out["mail_ok"] = mail["ok"]
        return JSONResponse(out)
    token = A.new_session_token()
    store.create_session(user["id"], token, A.session_expiry(), request.headers.get("user-agent", ""))
    keys = store.list_api_keys(user["id"])
    out["api_key"] = keys[0]["key"] if keys else ""
    resp = JSONResponse(out)
    _set_session_cookie(resp, request, token)
    return resp


@router.get("/auth/verify-email")
async def auth_verify_email(request: Request):
    """邮箱确认（magic link）：校验「签名有效 + 30 分钟未过期 + 邮箱一致」后
    把 pending 账号置为 active 并发默认 API Key。公开端点（邮件里点进来，无会话）。"""
    store = request.app.state.store
    token = request.query_params.get("t", "")
    user, reason = A.check_email_token(store, token)

    def page(title: str, body_html: str, status_code: int = 200) -> HTMLResponse:
        html = (f'<!doctype html><html lang="zh"><head><meta charset="utf-8">'
                f'<meta name="viewport" content="width=device-width,initial-scale=1">'
                f'<title>{title} · LLM Relay</title></head>'
                f'<body style="margin:0;background:#0b1220;color:#dbe6f5;'
                f'font:14px/1.7 -apple-system,"Segoe UI","Microsoft YaHei",sans-serif">'
                f'<div style="max-width:560px;margin:12vh auto;padding:28px 30px;'
                f'background:#111a2c;border:1px solid #22304a;border-radius:12px">'
                f'<h1 style="font-size:17px;margin:0 0 10px">{title}</h1>{body_html}</div></body></html>')
        return HTMLResponse(html, status_code=status_code)

    if user and reason == "":
        base = str(request.base_url).rstrip("/")
        if user["status"] == "pending":
            store.update_user(user["id"], status="active")
            if not store.list_api_keys(user["id"]):
                store.create_api_key(user["id"], A.new_api_key(), label="默认")
            return page("邮箱确认成功",
                        f"邮箱 <b>{html_mod.escape(user['email'])}</b> 已确认，现在可以正常登录使用。"
                        f'<p style="margin-top:16px"><a href="{base}" '
                        f'style="color:#4f8cff">前往登录控制台 →</a></p>')
        return page("邮箱已确认", "这个邮箱之前已经完成确认，直接登录即可。"
                        f'<p style="margin-top:16px"><a href="{base}" '
                        f'style="color:#4f8cff">前往登录控制台 →</a></p>')
    if reason == "expired":
        return page("确认链接已失效",
                    "链接超过 30 分钟有效期，或该待确认账号已超时自动清理。请回到登录页"
                    "重新注册，或点击「重发确认邮件」后打开最新一封邮件里的链接。", 400)
    return page("确认链接无效",
                "链接不完整或已被篡改。请回到登录页点击「重发确认邮件」，"
                "并打开最新一封邮件里的链接。", 400)


@router.post("/auth/resend-verify")
async def auth_resend_verify(request: Request):
    """重发邮箱确认邮件（登录页入口，公开）。限频：同一邮箱 60 秒间隔、
    每日 5 次；不泄露账号是否存在。"""
    store = request.app.state.store
    raw = await request.json() if await request.body() else {}
    try:
        email = A.normalize_email(str(raw.get("email") or ""))
    except HTTPException:
        return {"ok": True, "message": "如果该账号存在且未确认邮箱，已发送确认邮件。"}
    store.purge_stale_pending()   # 30 分钟未确认的 pending 已清理，重发走「重新注册」话术
    user = store.get_user_by_email(email)
    if not user or user["status"] != "pending":
        return {"ok": True, "message": "如果该账号存在且未确认邮箱，已发送确认邮件。"}
    now = int(time.time())
    last = int(store.get_meta(f"resend_last:{email}") or 0)
    if now - last < A.RESEND_COOLDOWN_S:
        return {"ok": True, "message": f"发送过于频繁，请 {A.RESEND_COOLDOWN_S - (now - last)} 秒后再试。"}
    day_key = f"resend_count:{email}:{time.strftime('%Y-%m-%d')}"
    cnt = int(store.get_meta(day_key) or 0)
    if cnt >= A.RESEND_DAILY_LIMIT:
        return {"ok": True, "message": "今日重发次数已用完，请明天再试。"}
    store.set_meta(f"resend_last:{email}", str(now))
    store.set_meta(day_key, str(cnt + 1))
    mail = A.send_verification_email(store, user, str(request.base_url).rstrip("/"))
    note = "" if mail["mode"] == "smtp" or not mail["ok"] else "（当前为日志模式：确认链接已写入服务器日志 [mail/log]）"
    return {"ok": True, "message": f"确认邮件已发往 {email}{note}"}


@router.post("/auth/login")
async def auth_login(request: Request):
    body = await request.json() if await request.body() else {}
    store = request.app.state.store
    user = A.login(store, str(body.get("email") or ""), str(body.get("password") or ""))
    token = A.new_session_token()
    store.create_session(user["id"], token, A.session_expiry(), request.headers.get("user-agent", ""))
    resp = JSONResponse({"ok": True, "user": A.public_user(user)})
    _set_session_cookie(resp, request, token)
    return resp


@router.post("/auth/logout")
async def auth_logout(request: Request):
    token = (request.cookies.get(A.COOKIE_NAME) or "").strip()
    if token:
        request.app.state.store.delete_session(token)
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(A.COOKIE_NAME, path="/")
    return resp


@router.post("/auth/password")
async def change_password(request: Request, _: None = Depends(auth)):
    """改自己的密码。"""
    body = await request.json() if await request.body() else {}
    store = request.app.state.store
    user = _user(request)
    if not A.verify_password(str(body.get("old_password") or ""), user["pass_hash"]):
        raise HTTPException(400, "原密码不对")
    newpw = A.check_password_strength(str(body.get("new_password") or ""))
    store.update_user(user["id"], pass_hash=A.hash_password(newpw))
    store.delete_user_sessions(user["id"])     # 其他设备要重新登录
    return {"ok": True}


# ---------------------------------------------------------------- 我的 API Key
@router.get("/keys")
async def my_keys(request: Request, _: None = Depends(auth)):
    user = _user(request)
    keys = request.app.state.store.list_api_keys(user["id"])
    return {"data": [{**k, "key": A.mask_key(k["key"]), "revoked": bool(k["revoked"])} for k in keys]}


def _parse_expires(v) -> str | None:
    """过期时间入参：空/None = 永不过期；接受 'YYYY-MM-DD'（纯日期 = 当天仍可用，
    按当天 23:59:59 失效）/ 'YYYY-MM-DD HH:MM' / 'YYYY-MM-DD HH:MM:SS' / ISO 带时区。"""
    s = str(v or "").strip()
    if not s:
        return None
    from datetime import datetime, timedelta
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.strptime(s, fmt)
            if fmt == "%Y-%m-%d":
                dt = dt + timedelta(hours=23, minutes=59, seconds=59)
            return dt.astimezone().isoformat(timespec="seconds")
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(s).astimezone().isoformat(timespec="seconds")
    except ValueError:
        raise HTTPException(400, "过期时间格式不对，可用：YYYY-MM-DD 或 YYYY-MM-DD HH:MM")


@router.post("/keys")
async def create_key(request: Request, _: None = Depends(auth)):
    body = await request.json() if await request.body() else {}
    user = _user(request)
    key = request.app.state.store.create_api_key(
        user["id"], A.new_api_key(), str(body.get("label") or "")[:60],
        _parse_expires(body.get("expires_at")))
    return {"ok": True, "key": key["key"], "id": key["id"],   # 明文只在这里返回一次
            "label": key["label"], "created_at": key["created_at"],
            "expires_at": key.get("expires_at")}


@router.get("/keys/{key_id}/reveal")
async def reveal_key(key_id: int, request: Request, _: None = Depends(auth)):
    """把自己的某一把 key 明文取回来（用于「复制」）。只能看自己的。"""
    user = _user(request)
    store = request.app.state.store
    for k in store.list_api_keys(user["id"]):
        if k["id"] == key_id:
            if k["revoked"]:
                raise HTTPException(400, "这把 key 已被吊销")
            return {"ok": True, "id": key_id, "key": k["key"], "label": k["label"]}
    raise HTTPException(404, "key 不存在")


@router.patch("/keys/{key_id}")
async def update_key(key_id: int, request: Request, _: None = Depends(auth)):
    """改 key 的备注 / 过期时间。body: {label?, expires_at?}（expires_at 传 "" 或 null = 永不过期）。"""
    body = await request.json() if await request.body() else {}
    user = _user(request)
    store = request.app.state.store
    k = store.get_api_key(key_id)
    if not k or k["user_id"] != user["id"]:
        raise HTTPException(404, "key 不存在")
    label = body.get("label")
    if label is not None:
        label = str(label)[:60]
        if not store.set_key_label(user["id"], key_id, label):
            raise HTTPException(404, "key 不存在")
    if "expires_at" in body:
        store.update_key_expires(user["id"], key_id, _parse_expires(body.get("expires_at")))
    k2 = store.get_api_key(key_id)
    return {"ok": True, "label": k2["label"], "expires_at": k2.get("expires_at")}


@router.delete("/keys/{key_id}")
async def delete_key(key_id: int, request: Request, _: None = Depends(auth)):
    """彻底删除一把 key（立即失效，记录也不再保留）。"""
    user = _user(request)
    if not request.app.state.store.delete_api_key(user["id"], key_id):
        raise HTTPException(404, "key 不存在")
    return {"ok": True}


# ---------------------------------------------------------------- 管理员：账号管理
@router.get("/users")
async def list_users(request: Request, _: None = Depends(auth)):
    admin = A.require_admin(request)
    store = request.app.state.store
    store.purge_stale_pending()   # 超时未确认的 pending 不再显示
    return {"data": store.list_users(), "me": admin["id"]}


@router.post("/users")
async def create_user(request: Request, _: None = Depends(auth)):
    A.require_admin(request)
    body = await request.json() if await request.body() else {}
    store = request.app.state.store
    email = A.normalize_email(str(body.get("email") or ""))
    password = A.check_password_strength(str(body.get("password") or ""))
    store.purge_stale_pending()
    if store.get_user_by_email(email):
        raise HTTPException(400, "这个邮箱已经注册过了")
    role = "admin" if str(body.get("role") or "") == "admin" else "user"
    user = store.create_user(email, A.hash_password(password), role=role)
    key = store.create_api_key(user["id"], A.new_api_key(), label="默认")
    return {"ok": True, "user": A.public_user(user), "api_key": key["key"]}


@router.patch("/users/{user_id}")
async def update_user(user_id: int, request: Request, _: None = Depends(auth)):
    admin = A.require_admin(request)
    body = await request.json() if await request.body() else {}
    store = request.app.state.store
    target = store.get_user(user_id)
    if not target:
        raise HTTPException(404, "账号不存在")
    def _other_active_admins() -> int:
        return sum(1 for x in store.list_users()
                   if x["role"] == "admin" and x["status"] == "active" and x["id"] != user_id)

    fields: dict = {}
    if "status" in body:
        status = "active" if str(body["status"]) == "active" else "disabled"
        if user_id == admin["id"] and status == "disabled":
            raise HTTPException(400, "不能禁用自己")
        if status == "disabled" and target["role"] == "admin" and _other_active_admins() == 0:
            raise HTTPException(400, "这是最后一个管理员，不能禁用")
        fields["status"] = status
        if status == "disabled":
            store.delete_user_sessions(user_id)
    if "role" in body:
        role = "admin" if str(body["role"]) == "admin" else "user"
        if user_id == admin["id"] and role != "admin":
            raise HTTPException(400, "不能取消自己的管理员身份")
        if role != "admin" and target["role"] == "admin" and _other_active_admins() == 0:
            raise HTTPException(400, "这是最后一个管理员，不能降级")
        fields["role"] = role
    if body.get("password"):
        fields["pass_hash"] = A.hash_password(A.check_password_strength(str(body["password"])))
        store.delete_user_sessions(user_id)
    if not fields:
        raise HTTPException(400, "没有要改的字段")
    store.update_user(user_id, **fields)
    return {"ok": True, "user": A.public_user(store.get_user(user_id))}


@router.delete("/users/{user_id}")
async def delete_user(user_id: int, request: Request, _: None = Depends(auth)):
    """彻底删除账号（会话 / API Key / 用户行），请求历史保留作审计。"""
    admin = A.require_admin(request)
    if user_id == admin["id"]:
        raise HTTPException(400, "不能删除自己")
    store = request.app.state.store
    target = store.get_user(user_id)
    if not target:
        raise HTTPException(404, "账号不存在")
    if target["role"] == "admin" and target["status"] == "active":
        others = sum(1 for x in store.list_users()
                     if x["role"] == "admin" and x["status"] == "active" and x["id"] != user_id)
        if others == 0:
            raise HTTPException(400, "这是最后一个管理员，不能删除")
    store.delete_user_full(user_id)
    return {"ok": True, "note": "账号已彻底删除，请求历史保留"}


@router.get("/health")
async def api_health(request: Request, _: None = Depends(auth)):
    reg = _reg(request)
    return {"ok": True, "providers": [{"platform": u.platform, "enabled": u.enabled,
                                       "keys": len(u.api_keys), "models": len(u.models),
                                       "type": u.type} for u in reg.upstreams],
            "key_status": reg.key_status()}


# ---------------------------------------------------------------- 网页模型会话
@router.post("/{path:path}")
async def _unknown(path: str, request: Request):
    return JSONResponse(status_code=404, content={"detail": f"/api/{path} 不存在"})
