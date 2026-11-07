"""pytest 公共装置：起一个真实 HTTP 假上游，把中转站指向它。"""
from __future__ import annotations

import json
import os
import socket
import tempfile
import threading
import time
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix="relay-test-"))
os.environ["RELAY_CONFIG_DIR"] = str(_TMP / "config")
os.environ["RELAY_DATA_DIR"] = str(_TMP / "data")

import pytest  # noqa: E402
import uvicorn  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

# 测试账号的 API Key（client fixture 建号时生成）
_TEST_API_KEY: str = ""




def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _wait_port(port: int, timeout: float = 25.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), 0.25):
                return True
        except OSError:
            time.sleep(0.05)
    return False


@pytest.fixture(scope="session")
def upstream_port() -> int:
    from tests.fake_upstream import app as fake_app

    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(fake_app, host="127.0.0.1", port=port,
                                           log_level="warning", lifespan="off"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    assert _wait_port(port), "假上游没起来"

    cfg_dir = Path(os.environ["RELAY_CONFIG_DIR"])
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "upstreams.json").write_text(json.dumps([
        {"platform": "fake", "name": "Fake OpenAI", "type": "openai",
         "base_url": f"http://127.0.0.1:{port}/v1",
         "api_keys": ["sk-fake-openai"], "models": ["fake-gpt", "fake-gpt-2"],
         "model_map": {"gpt-cheap": "fake-gpt"}},
        {"platform": "fakeclaude", "name": "Fake Anthropic", "type": "anthropic",
         "base_url": f"http://127.0.0.1:{port}",
         "api_keys": ["sk-fake-claude"], "models": ["fake-claude"]},
        {"platform": "fakereason", "name": "Fake Reasoner", "type": "openai",
         "base_url": f"http://127.0.0.1:{port}/v1",
         "api_keys": ["sk-reason"], "models": ["fake-reasoner"],
         "extra": {"model_reasoning": {"fake-reasoner": "off"}}},
        {"platform": "fakethink", "name": "Fake Claude Thinking Off", "type": "anthropic",
         "base_url": f"http://127.0.0.1:{port}",
         "api_keys": ["sk-think"], "models": ["fake-claude-thinking"],
         "extra": {"model_reasoning": {"fake-claude-thinking": "off"}}},
        {"platform": "fakethinkon", "name": "Fake Claude Thinking Medium", "type": "anthropic",
         "base_url": f"http://127.0.0.1:{port}",
         "api_keys": ["sk-thinkon"], "models": ["fake-claude-thinking"],
         "extra": {"model_reasoning": {"fake-claude-thinking": "medium"}}},
        {"platform": "fakemodel", "name": "Fake Model Level Reasoning", "type": "openai",
         "base_url": f"http://127.0.0.1:{port}/v1",
         "api_keys": ["sk-model"], "models": ["fake-reasoner", "fake-gpt"],
         "extra": {"reasoning": "off",
                   "model_reasoning": {"fake-reasoner": "off"}}},
        {"platform": "fakefield", "name": "Fake Per-Model Off Field", "type": "openai",
         "base_url": f"http://127.0.0.1:{port}/v1",
         "api_keys": ["sk-field"], "models": ["fake-gpt"],
         "extra": {"model_reasoning": {"fake-gpt": {"policy": "off", "field": "enable_thinking"}}}},
        {"platform": "fakemulti", "name": "Fake Multi Key", "type": "openai",
         "base_url": f"http://127.0.0.1:{port}/v1",
         "api_keys": ["sk-multi-aaaa1111", "sk-multi-bbbb2222"],
         "models": ["multi-gpt"]},
        {"platform": "fakeollama", "name": "Ollama Cloud", "type": "openai",
         "base_url": f"http://127.0.0.1:{port}/v1",
         "api_keys": ["sk-oll"], "models": ["beam"],
         "extra": {"model_reasoning": {"beam": "off"}}},
    ], ensure_ascii=False, indent=2), "utf-8")

    from app.main import registry
    registry.reload()
    yield port
    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture(scope="session")
def client(upstream_port) -> TestClient:
    from app import auth as auth_mod
    from app.main import app as relay_app
    from app.main import store

    # 必须有账号：建一个 admin 并发一把 API Key，下面所有用 hdr（账号 key）的测试都靠它鉴权。
    global _TEST_API_KEY
    if store.count_users() == 0:
        admin = store.create_user("admin@test.local", auth_mod.hash_password("test-password"),
                                  role="admin")
        _TEST_API_KEY = auth_mod.new_api_key()
        store.create_api_key(admin["id"], _TEST_API_KEY, label="测试")

    with TestClient(relay_app) as c:
        yield c


@pytest.fixture(autouse=True)
def _clear_received():
    from tests import fake_upstream
    fake_upstream.RECEIVED.clear()
    yield


@pytest.fixture()
def hdr(client) -> dict:
    assert _TEST_API_KEY, "client fixture 应先建好测试账号"
    return {"authorization": f"Bearer {_TEST_API_KEY}"}


def sse_parse(lines) -> list[tuple[str | None, str]]:
    """把 SSE 行流解析成 (event, data) 列表。"""
    out: list[tuple[str | None, str]] = []
    event = None
    data: list[str] = []
    for line in lines:
        if line == "":
            if data:
                out.append((event, "\n".join(data)))
            event, data = None, []
            continue
        if line.startswith(":"):
            continue
        if line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].lstrip())
    if data:
        out.append((event, "\n".join(data)))
    return out
