"""上游平台注册表：upstreams.json 热加载 + 模型路由 + 多 key 轮换/冷却。"""
from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path
from typing import Optional

import httpx
from pydantic import BaseModel, Field, field_validator


class Upstream(BaseModel):
    platform: str  # 唯一 id，客户端可用 "platform/model" 前缀强制路由
    name: str = ""
    type: str = "openai"  # "openai"（OpenAI 兼容） | "anthropic"
    base_url: str  # 例如 https://api.deepseek.com/v1 或 https://api.anthropic.com
    api_key: str = ""  # 单 key 便捷写法
    api_keys: list[str] = Field(default_factory=list)  # 多 key 轮换
    models: list[str] = Field(default_factory=list)  # 该平台提供的模型名
    model_map: dict[str, str] = Field(default_factory=dict)  # 客户端模型名 -> 上游模型名
    extra: dict = Field(default_factory=dict)  # chat_path / messages_path / headers / default_max_tokens...
    proxy: str = ""  # 可选：该平台专用出口代理，如 http://127.0.0.1:10808 / socks5h://127.0.0.1:10808
    enabled: bool = True
    timeout_s: float = 300.0

    @field_validator("type")
    @classmethod
    def _type(cls, v: str) -> str:
        v = (v or "openai").lower().strip()
        return v if v in ("openai", "anthropic") else "openai"

    @field_validator("platform")
    @classmethod
    def _platform(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            raise ValueError("platform 不能为空")
        return v

    def model_post_init(self, __context) -> None:  # noqa: D401
        merged = ([self.api_key] + list(self.api_keys)) if self.api_key else list(self.api_keys)
        seen: set[str] = set()
        keys: list[str] = []
        for k in merged:
            k = (k or "").strip()
            if k and k not in seen:
                seen.add(k)
                keys.append(k)
        self.api_keys = keys
        self.base_url = (self.base_url or "").strip().rstrip("/")

    @property
    def label(self) -> str:
        return self.name or self.platform

    def url_for(self, kind: str) -> str:
        """kind: chat | messages | models | count_tokens。"""
        default = {
            "chat": "/chat/completions",
            "messages": "/v1/messages",
            "models": "/v1/models" if self.type == "anthropic" else "/models",
            "count_tokens": "/v1/messages/count_tokens",
        }[kind]
        path = str(self.extra.get(f"{kind}_path") or default)
        if not path.startswith("/"):
            path = "/" + path
        return self.base_url + path


class Registry:
    def __init__(self, upstreams_file: Path, clients_file: Path):
        self.upstreams_file = upstreams_file
        self.clients_file = clients_file
        self._lock = threading.RLock()
        self._rr: dict[str, int] = {}
        self._cooldown: dict[tuple[str, int], float] = {}
        self.upstreams: list[Upstream] = []
        self._raw: list[dict] = []
        self.client_map: dict[str, str] = {}
        self._mtime: Optional[float] = None
        self.reload()

    # ---------- 加载 / 热重载 ----------
    def reload(self) -> None:
        with self._lock:
            raw: list = []
            if self.upstreams_file.exists():
                text = self.upstreams_file.read_text(encoding="utf-8").strip()
                if text:
                    raw = json.loads(text)
            self._raw = raw
            self.upstreams = [Upstream.model_validate(x) for x in raw]
            cmap: dict[str, str] = {}
            if self.clients_file.exists():
                text = self.clients_file.read_text(encoding="utf-8").strip()
                if text:
                    for c in json.loads(text):
                        cmap[c["key"]] = c.get("name", c["key"])
            self.client_map = cmap
            try:
                self._mtime = self.upstreams_file.stat().st_mtime if self.upstreams_file.exists() else 0.0
            except OSError:
                self._mtime = 0.0

    def file_changed(self) -> None:
        try:
            m = self.upstreams_file.stat().st_mtime
        except OSError:
            m = 0.0
        with self._lock:
            if m != self._mtime:
                self.reload()

    async def watch(self, interval: float = 5.0) -> None:
        """后台轮询 upstreams.json，手改配置 5s 内生效。"""
        while True:
            await asyncio.sleep(interval)
            try:
                self.file_changed()
            except Exception:
                pass

    # ---------- 持久化 ----------
    def save(self, upstreams: list[dict]) -> None:
        with self._lock:
            self.upstreams_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.upstreams_file.with_suffix(".tmp")
            tmp.write_text(json.dumps(upstreams, ensure_ascii=False, indent=2), "utf-8")
            tmp.replace(self.upstreams_file)
        self.reload()

    def raw(self) -> list[dict]:
        with self._lock:
            return [dict(x) for x in self._raw]

    def upsert(self, payload: dict) -> dict:
        platform = (payload.get("platform") or "").strip()
        if not platform:
            raise ValueError("platform 不能为空")
        rows = self.raw()
        row = Upstream.model_validate({**payload, "platform": platform}).model_dump()
        for i, r in enumerate(rows):
            if r.get("platform") == platform:
                rows[i] = row
                break
        else:
            rows.append(row)
        self.save(rows)
        return row

    def remove(self, platform: str) -> bool:
        rows = self.raw()
        left = [r for r in rows if r.get("platform") != platform]
        if len(left) == len(rows):
            return False
        self.save(left)
        return True

    def set_models(self, platform: str, models: list[str]) -> None:
        rows = self.raw()
        for r in rows:
            if r.get("platform") == platform:
                r["models"] = models
                self.save(rows)
                return
        raise KeyError(platform)

    # ---------- 路由 ----------
    def enabled(self) -> list[Upstream]:
        return [u for u in self.upstreams if u.enabled and u.api_keys]

    def find(self, platform: str) -> Optional[Upstream]:
        for u in self.upstreams:
            if u.platform == platform:
                return u
        return None

    @staticmethod
    def usable(up: Optional[Upstream]) -> bool:
        return bool(up and up.enabled and up.api_keys)

    def resolve(self, model: str, force_platform: Optional[str] = None) -> Optional[tuple[Upstream, str]]:
        """解析下游 model 字段 -> (upstream, 上游真实模型名)。

        规则：
        1. X-Upstream-Platform 头（或 force_platform）强制指定平台，model 取前缀部分；
        2. "platform/model" 前缀形式按平台精确路由（前缀必须命中已知平台）；
        3. 否则按配置顺序找第一个声明了该模型的启用平台。
        """
        model = (model or "").strip()
        if not model:
            return None
        if force_platform:
            up = self.find(force_platform)
            if self.usable(up):
                rest = model.split("/", 1)[1] if "/" in model else model
                return up, up.model_map.get(rest, rest)
        if "/" in model:
            pf, rest = model.split("/", 1)
            up = self.find(pf)
            if self.usable(up):
                return up, up.model_map.get(rest, up.model_map.get(model, rest))
        for up in self.enabled():
            if model in self._names(up):
                return up, up.model_map.get(model, model)
        # 别名直呼：model_map 的 key 也能直接当下游模型名用（不必再写进 models）
        for up in self.enabled():
            alias = model.split("/", 1)[-1] if model.startswith(up.platform + "/") else model
            if alias in (up.model_map or {}):
                return up, up.model_map[alias]
        return None

    @staticmethod
    def _names(up: "Upstream") -> list[str]:
        """这个平台对下游暴露的模型名。"""
        return list(up.models or [])

    def all_model_ids(self) -> list[str]:
        """给下游 /v1/models 用：既给 "model"，也给 "platform/model"，别名也列出来。"""
        ids: list[str] = []
        seen: set[str] = set()
        for up in self.enabled():
            names = self._names(up) + [a for a in (up.model_map or {}) if a not in (up.models or [])]
            for m in names:
                for i in (m, f"{up.platform}/{m}"):
                    if i not in seen:
                        seen.add(i)
                        ids.append(i)
        return ids

    def model_index(self) -> list[dict]:
        out: list[dict] = []
        for up in self.upstreams:
            usable = self.usable(up)
            for m in self._names(up):
                out.append({"id": f"{up.platform}/{m}", "platform": up.platform,
                            "provider_name": up.label, "model": m,
                            "type": up.type, "enabled": usable})
            for alias, target in (up.model_map or {}).items():
                if alias in self._names(up):
                    continue
                out.append({"id": f"{up.platform}/{alias}", "platform": up.platform,
                            "provider_name": up.label, "model": alias, "maps_to": target,
                            "type": up.type, "enabled": usable})
        return out

    # ---------- 思考 / 推理策略 ----------
    @staticmethod
    def reasoning_rule(up: Upstream, model: str) -> tuple[str, str]:
        """模型级思考设置 -> (策略, 关闭写法)。

        extra.model_reasoning = {
            "上游真实模型名": "off"                                  # 写法走自动
            "上游真实模型名": {"policy": "off", "field": "thinking_type"}   # 写法也指定
        }
        平台级没有思考开关，默认永远 auto（跟随客户端）。
        """
        per_model = (up.extra or {}).get("model_reasoning") or {}
        raw = ""
        if isinstance(per_model, dict):
            raw = per_model.get(model) or per_model.get("*") or ""
        policy, field = "", ""
        if isinstance(raw, dict):
            policy = str(raw.get("policy") or "").lower().strip()
            field = str(raw.get("field") or "").lower().strip()
        elif raw:
            policy = str(raw).lower().strip()
        if policy not in ("off", "auto", "low", "medium", "high"):
            policy = "auto"
        if field not in ("", "auto", "reasoning_effort", "enable_thinking",
                         "thinking_type", "chat_template", "none"):
            field = ""
        return policy, field

    @classmethod
    def reasoning_policy(cls, up: Upstream, model: str) -> str:
        return cls.reasoning_rule(up, model)[0]

    # ---------- key 轮换 / 冷却 ----------
    def next_key(self, up: Upstream) -> tuple[str, int]:
        """轮询取一个未冷却的 key；全部冷却时退回轮询位（宁可试错）。"""
        with self._lock:
            n = len(up.api_keys)
            if n == 0:
                raise RuntimeError(f"平台 {up.platform} 没有配置 api key")
            start = self._rr.get(up.platform, 0) % n
            now = time.time()
            for k in range(n):
                idx = (start + k) % n
                if self._cooldown.get((up.platform, idx), 0.0) <= now:
                    self._rr[up.platform] = idx + 1
                    return up.api_keys[idx], idx
            self._rr[up.platform] = start + 1
            return up.api_keys[start], start

    def report(self, up: Upstream, idx: int, ok: bool) -> None:
        with self._lock:
            if not ok:
                self._cooldown[(up.platform, idx)] = time.time() + 60.0
            else:
                self._cooldown.pop((up.platform, idx), None)

    def key_status(self) -> list[dict]:
        now = time.time()
        out: list[dict] = []
        for up in self.upstreams:
            for i, _k in enumerate(up.api_keys):
                cd = self._cooldown.get((up.platform, i), 0.0)
                out.append({"platform": up.platform, "index": i,
                            "cooling": cd > now, "cooldown_left_s": max(0, round(cd - now))})
        return out


async def fetch_models(up: Upstream, client: httpx.AsyncClient) -> list[str]:
    """拉取上游模型列表。失败抛 httpx.HTTPError / ValueError。"""
    key = up.api_keys[0] if up.api_keys else ""
    if up.type == "anthropic":
        headers = {"x-api-key": key, "anthropic-version": "2023-06-01"}
    else:
        headers = {"authorization": f"Bearer {key}"}
    headers.update({str(k): str(v) for k, v in (up.extra.get("headers") or {}).items()})
    resp = await client.get(up.url_for("models"), headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    items = data.get("data") if isinstance(data, dict) else data
    if not isinstance(items, list):
        raise ValueError("上游 /models 返回格式不认识")
    ids: list[str] = []
    for it in items:
        mid = it.get("id") if isinstance(it, dict) else str(it)
        if not mid:
            continue
        if up.type == "openai" and mid.startswith("models/"):
            # Google 等 OpenAI 兼容上游的 /models 返回原生资源名（models/xxx），
            # 但 chat 端点只收无前缀的名字——下拉列表存无前缀，避免配置里存进 400 的名字。
            mid = mid[len("models/"):]
        ids.append(mid)
    return sorted(set(ids))
