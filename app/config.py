"""全局路径与运行期配置。

所有路径都可以用环境变量覆盖，方便测试时隔离数据：
    RELAY_CONFIG_DIR / RELAY_DATA_DIR / RELAY_HOST / RELAY_PORT
"""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = Path(os.getenv("RELAY_CONFIG_DIR") or (ROOT / "config"))
DATA_DIR = Path(os.getenv("RELAY_DATA_DIR") or (ROOT / "data"))

UPSTREAMS_FILE = CONFIG_DIR / "upstreams.json"
SETTINGS_FILE = CONFIG_DIR / "settings.json"
CLIENTS_FILE = CONFIG_DIR / "clients.json"
DB_FILE = DATA_DIR / "relay.db"

for _d in (CONFIG_DIR, DATA_DIR):
    _d.mkdir(parents=True, exist_ok=True)

HOST = os.getenv("RELAY_HOST", "127.0.0.1")
PORT = int(os.getenv("RELAY_PORT", "8800"))

# 上游请求的默认超时（秒）：connect 短、read 长（流式可能很久没有新 token）
CONNECT_TIMEOUT_S = float(os.getenv("RELAY_CONNECT_TIMEOUT", "20"))
READ_TIMEOUT_S = float(os.getenv("RELAY_READ_TIMEOUT", "600"))

ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_MAX_TOKENS = int(os.getenv("RELAY_DEFAULT_MAX_TOKENS", "8192"))

VERSION = "1.0.2"


# ---- 反向代理子路径 ----
# 站点不是挂在域名根目录时（例如 https://x.com/llmrelay/v1），把前缀设成 /llmrelay，
# 网关会额外接受 /llmrelay/** 的请求（原路径仍然可用，本地直连不受影响）。
def normalize_base_path(raw: str) -> str:
    raw = (raw or "").strip()
    if not raw or raw == "/":
        return ""
    if not raw.startswith("/"):
        raw = "/" + raw
    return raw.rstrip("/")


def _read_settings() -> dict:
    try:
        return json.loads(SETTINGS_FILE.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def get_base_path() -> str:
    """环境变量 RELAY_BASE_PATH 优先，其次 settings.json，最后空（根目录）。

    注意：docker compose 里写 RELAY_BASE_PATH: ${RELAY_BASE_PATH:-} 会让变量
    「存在但是空串」——空串要当作没设，否则 settings.json 里的值永远读不到。
    """
    env = (os.getenv("RELAY_BASE_PATH") or "").strip()
    raw = env or _read_settings().get("base_path", "")
    return normalize_base_path(raw)


def set_base_path(value: str) -> str:
    v = normalize_base_path(value)
    data = _read_settings()
    data["base_path"] = v
    SETTINGS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return v


def base_path_env_locked() -> bool:
    return bool((os.getenv("RELAY_BASE_PATH") or "").strip())

# ---- 生产部署 ----
# 直接 HTTPS（不挂在反向代理后面时用；挂反代时留空，由反代终结 TLS）
SSL_CERTFILE = os.getenv("RELAY_SSL_CERTFILE", "").strip()
SSL_KEYFILE = os.getenv("RELAY_SSL_KEYFILE", "").strip()
# 信任哪些代理发来的 X-Forwarded-For / X-Forwarded-Proto（默认只信本机反代）
FORWARDED_ALLOW_IPS = os.getenv("RELAY_FORWARDED_ALLOW_IPS", "127.0.0.1").strip()
# 控制台跨域白名单，逗号分隔；默认 * （本地工具方便，公网部署建议收紧）
CORS_ORIGINS = [o.strip() for o in os.getenv("RELAY_CORS_ORIGINS", "*").split(",") if o.strip()] or ["*"]


