"""账号体系：邮箱+密码注册登录、会话 Cookie、每账号的下游 API Key。

两个入口共用一套判定：
- 控制台（/api/*）：优先会话 Cookie，其次 Bearer（账号 API Key）
- 下游客户端（/v1/*）：Bearer（账号 API Key）

密码用标准库 scrypt，不引第三方依赖。

邮箱确认（magic link）：自助注册的第二个账号起，注册后 status=pending，
确认链接（HMAC 签名、带时间戳、30 分钟有效）发到注册邮箱；点链接通过
/api/auth/verify-email 校验「签名 + 未过期 + 邮箱一致」后置为 active 并发
默认 API Key。首个账号（=admin）豁免，直接 active（零配置引导）。
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
import re
import secrets
import time
from datetime import datetime, timedelta

from fastapi import HTTPException, Request

COOKIE_NAME = "relay_session"
SESSION_DAYS = int(os.getenv("RELAY_SESSION_DAYS", "7"))
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]{2,}$")
MIN_PASSWORD = 8

_SCRYPT_N, _SCRYPT_R, _SCRYPT_P = 2 ** 14, 8, 1


# ---------------------------------------------------------------- 密码
def hash_password(password: str) -> str:
    salt = os.urandom(16)
    dk = hashlib.scrypt(password.encode("utf-8"), salt=salt,
                        n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P, dklen=32)
    return f"scrypt${_SCRYPT_N}${_SCRYPT_R}${_SCRYPT_P}${salt.hex()}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, n, r, p, salt_hex, hash_hex = (stored or "").split("$")
        if algo != "scrypt":
            return False
        dk = hashlib.scrypt(password.encode("utf-8"), salt=bytes.fromhex(salt_hex),
                            n=int(n), r=int(r), p=int(p), dklen=len(hash_hex) // 2)
        return hmac.compare_digest(dk.hex(), hash_hex)
    except Exception:
        return False


def check_password_strength(password: str) -> str:
    if len(password or "") < MIN_PASSWORD:
        raise HTTPException(400, f"密码至少 {MIN_PASSWORD} 位")
    return password


def normalize_email(email: str) -> str:
    email = (email or "").strip().lower()
    if not EMAIL_RE.match(email):
        raise HTTPException(400, "邮箱格式不对")
    return email


# ---------------------------------------------------------------- 会话 / key
def new_session_token() -> str:
    return secrets.token_urlsafe(32)


def session_expiry() -> str:
    return (datetime.now().astimezone() + timedelta(days=SESSION_DAYS)).isoformat(timespec="seconds")


def new_api_key() -> str:
    return "sk-relay-" + secrets.token_urlsafe(30)
def new_password() -> str:
    """管理员重置租户密码用的自动生成密码（12 位，URL 安全字符；长度>=8 满足强度校验）。"""
    return secrets.token_urlsafe(9)



def mask_key(key: str) -> str:
    key = key or ""
    if len(key) > 14:
        return f"{key[:11]}…{key[-4:]}"
    return "…" + key[-4:] if key else ""


def _bearer(request: Request) -> str:
    auth = request.headers.get("authorization") or ""
    if auth.lower().startswith("bearer "):
        token = auth[7:].strip()
        if token:                       # 空的 "Bearer " 不算数，继续看别的头
            return token
    for name in ("x-api-key", "x-admin-key"):
        v = (request.headers.get(name) or "").strip()
        if v:
            return v
    return ""


# ---------------------------------------------------------------- 用户解析
def user_for_token(store, token: str) -> dict | None:
    """API Key → 账号。没有账号或 key 无效都返回 None。"""
    if not token:
        return None
    return store.user_for_api_key(token)


def authenticate(request: Request) -> tuple[dict | None, str]:
    """下游入口（/v1/*）用：只认 Authorization Bearer / x-api-key。

    返回 (user, "") 表示通过；(None, reason) 里 reason 是
    "missing" / "invalid" / "expired"（key 已过期）/ "revoked"（key 被禁用/吊销）。
    expired / revoked 时把审计信息挂在 request.state.auth_audit
    （user_id / dkey_id / expires_at），供 main 写 401 日志与报错回显。
    """
    token = _bearer(request)
    if not token:
        return None, "missing"
    user = user_for_token(request.app.state.store, token)
    if not user:
        return None, "invalid"
    if user.pop("_key_revoked", False):
        # 禁用/吊销的 key 立即拦（客户端路径）；控制台 Cookie 会话不受影响
        request.state.auth_audit = {"user_id": user["id"], "dkey_id": user.get("_key_id"),
                                    "expires_at": ""}
        return None, "revoked"
    if user.pop("_key_expired", False):
        request.state.auth_audit = {"user_id": user["id"], "dkey_id": user.get("_key_id"),
                                    "expires_at": user.get("_key_expires_at") or ""}
        return None, "expired"
    return user, ""


def current_user(request: Request) -> dict:
    """控制台用：Cookie 会话优先，其次 Bearer。没登录直接 401。"""
    store = request.app.state.store
    token = (request.cookies.get(COOKIE_NAME) or "").strip()
    if token:
        sess = store.get_session(token)
        if sess:
            user = store.get_user(sess["user_id"])
            if user and user["status"] == "active":
                return user
    user = user_for_token(store, _bearer(request))
    if user:
        return user
    raise HTTPException(401, "未登录")


def optional_user(request: Request) -> dict | None:
    try:
        return current_user(request)
    except HTTPException:
        return None


def require_admin(request: Request) -> dict:
    user = current_user(request)
    if user.get("role") != "admin":
        raise HTTPException(403, "需要管理员权限")
    return user


def tenant_of(request: Request) -> int | None:
    """管理员可以通过 ?tenant=<id> 看别的账号；普通账号只能看自己。"""
    user = current_user(request)
    if user.get("role") != "admin":
        return int(user["id"])
    raw = (request.query_params.get("tenant") or "").strip()
    if raw in ("", "all", "0"):
        return None if raw == "all" else int(user["id"])
    try:
        return int(raw)
    except ValueError:
        return int(user["id"])


def public_user(user: dict, *, self_view: bool = False) -> dict:
    return {
        "id": user["id"],
        "email": user["email"],
        "role": user["role"],
        "status": user["status"],
        "created_at": user.get("created_at"),
        "last_login": user.get("last_login"),
    }


# ---------------------------------------------------------------- 初始化
def ensure_admin(store) -> dict | None:
    """没有账号时，用环境变量建一个管理员（可选），否则等第一个注册的人成为管理员。"""
    if store.count_users():
        return store.first_admin()
    email = (os.getenv("RELAY_ADMIN_EMAIL") or "").strip().lower()
    password = os.getenv("RELAY_ADMIN_PASSWORD") or ""
    if not email or not password:
        return None
    user = store.create_user(email, hash_password(password), role="admin")
    _after_user_created(store, user)
    return user


def _after_user_created(store, user: dict) -> None:
    """第一个账号（=管理员）接管历史数据，并自动发一把 API Key。"""
    if user.get("role") == "admin" and store.count_users() == 1:
        store.adopt_orphan_records(user["id"])
    store.create_api_key(user["id"], new_api_key(), label="默认")


def register(store, email: str, password: str) -> dict:
    email = normalize_email(email)
    check_password_strength(password)
    # 30 分钟没点确认链接的 pending 账号自动清理，邮箱可重新注册
    store.purge_stale_pending()
    if store.get_user_by_email(email):
        raise HTTPException(400, "这个邮箱已经注册过了")
    role = "admin" if store.count_users() == 0 else "user"
    # 首个账号（=管理员）豁免邮箱确认，直接可用（零配置引导）；
    # 之后的自助注册账号要点击邮件里的确认链接才激活（pending → active）
    status = "active" if role == "admin" and store.count_users() == 0 else "pending"
    user = store.create_user(email, hash_password(password), role=role, status=status)
    if role == "admin" and status == "active":   # 仅首个管理员（豁免确认）自动发 Key
        _after_user_created(store, user)
    return user


def login(store, email: str, password: str) -> dict:
    user = store.get_user_by_email(email)
    if not user or not verify_password(password, user["pass_hash"]):
        raise HTTPException(401, "邮箱或密码不对")
    if user["status"] == "pending":
        raise HTTPException(403, "email_unverified")
    if user["status"] != "active":
        raise HTTPException(403, "该账号已被禁用")
    store.update_user(user["id"], last_login=datetime.now().astimezone().isoformat(timespec="seconds"))
    return store.get_user(user["id"])


# ---------------------------------------------------------------- 邮箱确认（magic link）
EMAIL_TTL_SECONDS = 30 * 60          # 确认链接有效期 30 分钟
RESEND_COOLDOWN_S = 60              # 同一邮箱重发最小间隔
RESEND_DAILY_LIMIT = 5             # 同一邮箱每天最多重发次数
_SECRET_META_KEY = "email_signing_secret"


def get_email_secret(store) -> str:
    """HMAC 签名密钥：持久化在 meta 表（重启不失效），首次自动生成。"""
    secret = store.get_meta(_SECRET_META_KEY)
    if not secret:
        secret = secrets.token_urlsafe(32)
        store.set_meta(_SECRET_META_KEY, secret)
    return secret


def make_email_token(secret: str, user: dict, ttl: int = EMAIL_TTL_SECONDS) -> str:
    """确认链接的 t 参数：base64(uid:email:exp_ts) + '.' + HMAC-SHA256 前 32 hex。

    无状态（不入库）：链接本身带 uid、邮箱、签发时间戳，签名保证不可篡改。
    """
    exp = int(time.time()) + int(ttl)
    payload = f"{user['id']}:{user['email']}:{exp}"
    sig = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()[:32]
    b64 = base64.urlsafe_b64encode(payload.encode()).decode().rstrip("=")
    return f"{b64}.{sig}"


def build_email_link(store, user: dict, base_url: str) -> str:
    return f"{base_url.rstrip('/')}/api/auth/verify-email?t={make_email_token(get_email_secret(store), user)}"


def check_email_token(store, token: str) -> tuple[dict | None, str]:
    """校验确认链接的 t。返回 (user, reason)：reason="" 表示通过。

    校验项：签名有效（防篡改/枚举）→ 未过期（30 分钟时间戳窗口）→
    uid 对应账号存在且邮箱与链接内邮箱一致。
    """
    try:
        b64, sig = (token or "").rsplit(".", 1)
        payload = base64.urlsafe_b64decode(b64 + "=" * (-len(b64) % 4)).decode()
    except Exception:
        return None, "invalid"
    secret = get_email_secret(store)
    expect = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()[:32]
    if not hmac.compare_digest(expect, sig):
        return None, "invalid"
    try:
        uid_s, email, exp_s = payload.split(":")
        uid, exp = int(uid_s), int(exp_s)
    except ValueError:
        return None, "invalid"
    if time.time() > exp:
        return None, "expired"
    user = store.get_user(uid)
    if not user or (user["email"] or "").lower() != email.lower():
        return None, "invalid"
    return user, ""


def send_verification_email(store, user: dict, base_url: str) -> dict:
    """给 pending 账号发确认邮件；返回 mailer 的结果（含 mode: smtp/log）。"""
    from app import mailer
    return mailer.send_verification_email(store, user["email"], build_email_link(store, user, base_url))

