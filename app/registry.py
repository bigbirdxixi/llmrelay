"""上游平台注册表：upstreams.json 热加载 + 模型路由 + 多 key 轮换/冷却。"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from pathlib import Path
from typing import Optional

import httpx
from pydantic import BaseModel, Field, field_validator

from .crypto import KeyCipher

log = logging.getLogger("relay")


class Upstream(BaseModel):
    platform: str  # 唯一 id，客户端可用 "platform/model" 前缀强制路由
    name: str = ""
    type: str = "openai"  # "openai"（OpenAI 兼容） | "anthropic"
    base_url: str  # 例如 https://api.deepseek.com/v1 或 https://api.anthropic.com
    api_key: str = ""  # 单 key 便捷写法
    api_keys: list[str] = Field(default_factory=list)  # 多 key 轮换
    enc_keys: list[str] = Field(default_factory=list)  # v1.1.1：加密存储的 key（base64url AES-GCM 密文，K_t 派生自主密钥 S）
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
        """kind: chat | messages | models | count_tokens | images | videos | videos_status
        | audio_speech | audio_trans | embeddings。

        anthropic 协议默认路径带 /v1 前缀（按官方 base_url 不带 /v1 设计）；
        若 base_url 已以 /v1 结尾（如 Agnes 等 OpenAI 风格 hub 同时提供 Messages），
        去掉路径里的前导 /v1，避免拼出 /v1/v1/… → 404。base_url 因此永远不用动。
        """
        default = {
            "chat": "/chat/completions",
            "messages": "/v1/messages",
            "models": "/v1/models" if self.type == "anthropic" else "/models",
            "count_tokens": "/v1/messages/count_tokens",
            # 生成类端点（v2.0.0 透传）：默认 OpenAI/通用风格路径，
            # 平台可用 extra 的 images_path / videos_path / audio_speech_path / audio_trans_path 覆盖
            "images": "/images/generations",
            "videos": "/videos",
            # 视频任务状态查询（GET，v2.0.0）：默认与创建同路径；上游查询地址独立时
            # （如 Agnes 走 /agnesapi?video_id=..）用 extra.videos_status_path 给完整 URL
            "videos_status": "/videos",
            "audio_speech": "/audio/speech",
            "audio_trans": "/audio/transcriptions",
            # embeddings（v2.0.2 透传）：平台可用 extra 的 embeddings_path 覆盖
            # （如 NVIDIA NIM 走单数 /embedding）
            "embeddings": "/embeddings",
        }[kind]
        override = self.extra.get(f"{kind}_path")
        if override:
            o = str(override)
            if o.startswith(("http://", "https://")):
                return o  # 完整 URL 覆盖：上游查询端点不与 base_url 同前缀时用
            path = o if o.startswith("/") else "/" + o
        else:
            path = default
            if not path.startswith("/"):
                path = "/" + path
        base = self.base_url
        if base.endswith("/v1") and path.startswith("/v1/"):
            path = path[len("/v1"):]
        return base + path


class Registry:
    def __init__(self, upstreams_file: Path, clients_file: Path, cipher: Optional[KeyCipher] = None):
        self.upstreams_file = upstreams_file
        self.clients_file = clients_file
        self._cipher = cipher  # v1.1.1：该租户的 K_t（KeyCipher）；None=不加密（测试/老代码直建）
        self._lock = threading.RLock()
        self._rr: dict[str, int] = {}
        self._cooldown: dict[tuple[str, int], float] = {}
        self.upstreams: list[Upstream] = []
        self._raw: list[dict] = []
        self.client_map: dict[str, str] = {}
        self._mtime: Optional[float] = None
        # 聚合模型：{聚合名: ["平台/模型", ...]}，条目在文件里以 {"_kind":"aggregates"} 标记项存放
        self.aggregates: dict[str, list[str]] = {}
        self._agg_rr: dict[str, int] = {}
        self._agg_cd: dict[tuple[str, str], float] = {}
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
            marker = [x for x in raw if isinstance(x, dict) and x.get("_kind") == "aggregates"]
            ups: list[Upstream] = []
            for x in raw:
                if isinstance(x, dict) and x.get("_kind") == "aggregates":
                    continue
                enc = x.get("enc_keys") if isinstance(x, dict) else None
                if enc:
                    # v1.1.1：密文行——用本租户 K_t 解密进内存（磁盘不留明文）
                    dec: list[str] = []
                    if self._cipher is not None:
                        try:
                            dec = [self._cipher.open_str(s) for s in enc if isinstance(s, str)]
                        except Exception:
                            log.warning("平台 %s 的加密 key 解不开（主密钥不对或密文损坏），按无 key 处理",
                                        x.get("platform"))
                    elif not self._cipher:
                        log.warning("平台 %s 的 key 是加密存储，但本实例没有主密钥，按无 key 处理",
                                    x.get("platform"))
                    x = {**x, "api_key": "", "api_keys": dec}
                ups.append(Upstream.model_validate(x))
            self.upstreams = ups
            aggs = {}
            for mk in marker:
                if isinstance(mk.get("aggregates"), dict):
                    aggs.update({str(k): [str(e) for e in v]
                                 for k, v in mk["aggregates"].items() if isinstance(v, list)})
            self.aggregates = aggs
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
    def _encrypt_row(self, row: dict) -> dict:
        """v1.1.1：落盘前把明文 key 换成密文（enc_keys）；磁盘不留明文。

        已是密文行（无明文、有 enc_keys）原样保留，避免每次保存都重封 nonce；
        标记项（_kind）与非 dict 行不动。
        """
        if not isinstance(row, dict) or row.get("_kind"):
            return row
        keys: list[str] = []
        single = row.get("api_key")
        if isinstance(single, str) and single.strip():
            keys.append(single.strip())
        for k in (row.get("api_keys") or []):
            if isinstance(k, str) and k.strip() and k.strip() not in keys:
                keys.append(k.strip())
        if not keys and row.get("enc_keys"):
            return row
        out = dict(row)
        if keys:
            out["enc_keys"] = [self._cipher.seal_str(k) for k in keys]
        out["api_key"] = ""
        out["api_keys"] = []
        return out

    def _write(self, rows: list[dict]) -> None:
        with self._lock:
            if self._cipher is not None:
                rows = [self._encrypt_row(r) for r in rows]
            self.upstreams_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.upstreams_file.with_suffix(".tmp")
            tmp.write_text(json.dumps(rows, ensure_ascii=False, indent=2), "utf-8")
            tmp.replace(self.upstreams_file)
        self.reload()

    def save(self, upstreams: list[dict]) -> None:
        self._write(upstreams)

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
        # v2.0.1: 模型清单/别名变更 → 自动清掉指向不再登记模型的聚合条目（Bug：
        # 平台设置里去掉模型后，聚合里还保留着它的配置）
        self.prune_stale_aggregate_entries(platform)
        return row

    def remove(self, platform: str) -> bool:
        rows = self.raw()
        left = [r for r in rows if r.get("platform") != platform]
        if len(left) == len(rows):
            return False
        self.save(left)
        # v2.0.1: 平台删除 → 其「平台/*」聚合条目全部失效，清理（聚合丢空则删）
        self.prune_stale_aggregate_entries(platform)
        return True

    # ---------- 聚合模型（一个聚合名 -> 多个 平台/模型 条目，轮询 + 失败冷却自动切换） ----------
    def _persist_aggregates(self) -> None:
        with self._lock:
            base = [x for x in self._raw
                   if not (isinstance(x, dict) and x.get("_kind") == "aggregates")]
            if self.aggregates:
                base.append({"_kind": "aggregates", "aggregates": self.aggregates})
            self._write(base)

    def upsert_aggregate(self, name: str, entries: list[str]) -> None:
        with self._lock:
            self.aggregates[name] = entries
        self._persist_aggregates()

    def remove_aggregate(self, name: str) -> bool:
        with self._lock:
            if name not in self.aggregates:
                return False
            del self.aggregates[name]
            self._agg_rr.pop(name, None)
            for k in [k for k in self._agg_cd if k[0] == name]:
                self._agg_cd.pop(k, None)
        self._persist_aggregates()
        return True

    def entry_fresh(self, e: str) -> bool:
        """单条聚合条目在配置层是否有效（v2.0.1，比 resolve 的透传更严）：
        「平台/模型」必须指向该平台**当前登记的**模型（models 列表或 model_map 别名）；
        裸名沿用宽松语义（可解析即有效）。用于 /api/aggregates 保存校验、
        aggregate_info 可用判定、与平台模型清单变更时的陈旧条目清理。"""
        res = self.resolve(e)
        if res is None:
            return False
        up, _ = res
        if "/" not in e:
            return True
        pf, m = e.split("/", 1)
        if e in (up.model_map or {}):      # 整串「平台/模型」本身是 model_map 别名键
            return True
        exposed = set(up.models or []) | set((up.model_map or {}).keys())
        return m in exposed

    def prune_stale_aggregate_entries(self, platform: str | None = None) -> int:
        """平台模型清单变更 / 平台删除后，自动清掉聚合里指向失效目标的条目（v2.0.1）：
        目标平台已删、或模型不再登记在该平台 models/model_map 里的「平台/模型」条目被移除；
        丢掉全部条目的聚合随之删除。platform=None 查全部条目，否则只查该平台前缀的。
        裸名条目不参与清理（宽松语义不变）。返回移除条数（>0 时持久化聚合行）。"""
        removed = 0
        with self._lock:
            for agg_name in list(self.aggregates):
                entries = self.aggregates[agg_name]
                kept = []
                for e in entries:
                    if "/" not in e:
                        kept.append(e)
                        continue
                    pf = e.split("/", 1)[0]
                    if platform is not None and pf != platform:
                        kept.append(e)      # 与本次变更无关的平台，不动
                        continue
                    if self.entry_fresh(e):
                        kept.append(e)
                    else:
                        removed += 1
                if kept == entries:
                    continue
                if kept:
                    self.aggregates[agg_name] = kept
                    self._agg_rr[agg_name] = 0     # 条目清单变了，轮询指针归头
                else:
                    del self.aggregates[agg_name]
                    self._agg_rr.pop(agg_name, None)
                kept_names = set(self.aggregates.get(agg_name) or [])
                for ck in [k for k in self._agg_cd
                           if k[0] == agg_name and k[1] not in kept_names]:
                    self._agg_cd.pop(ck, None)
            if removed:
                self._persist_aggregates()
                log.info("聚合模型：移除 %d 条失效条目（platform=%s，目标平台/模型已不登记）",
                         removed, platform or "-")
        return removed

    def aggregate_entries(self, name: str) -> list[str]:
        return list(self.aggregates.get(name) or [])

    def next_aggregate(self, name: str) -> Optional[tuple[Upstream, str, str]]:
        """轮询取一个可用条目：跳过冷却中的；全冷却则从头再试（宁可试错）。
        返回 (upstream, 上游真实模型名, 条目串)；没有可解析条目返回 None。"""
        resolved: list[tuple[str, tuple[Upstream, str]]] = []
        for e in self.aggregate_entries(name):
            r = self.resolve(e)
            if r:
                resolved.append((e, r))
        n = len(resolved)
        if n == 0:
            return None
        with self._lock:
            now = time.time()
            start = self._agg_rr.get(name, 0) % n
            for k in range(n):
                idx = (start + k) % n
                if self._agg_cd.get((name, resolved[idx][0]), 0.0) <= now:
                    self._agg_rr[name] = idx + 1
                    up, up_model = resolved[idx][1]
                    return up, up_model, resolved[idx][0]
            idx = start % n
            self._agg_rr[name] = idx + 1
            up, up_model = resolved[idx][1]
            return up, up_model, resolved[idx][0]

    def report_aggregate(self, name: str, entry: str, ok: bool) -> None:
        with self._lock:
            if not ok:
                self._agg_cd[(name, entry)] = time.time() + 60.0
            else:
                self._agg_cd.pop((name, entry), None)

    def aggregate_info(self) -> list[dict]:
        """给 UI / API：每个聚合名 + 各条目的解析状态。"""
        out = []
        for name, entries in self.aggregates.items():
            rows = []
            for e in entries:
                r = self.resolve(e)
                rows.append({"entry": e, "ok": self.entry_fresh(e),
                             "up_model": r[1] if r else "",
                             "platform": r[0].platform if r else e.split("/", 1)[0]})
            out.append({"name": name, "entries": rows,
                       "usable": sum(1 for r in rows if r["ok"])})
        return out

    def set_models(self, platform: str, models: list[str]) -> None:
        rows = self.raw()
        for r in rows:
            if r.get("platform") == platform:
                r["models"] = models
                self.save(rows)
                # v2.0.1: 模型清单被改小（探测移除无效模型、UI 手改）→ 清理聚合里的失效条目
                self.prune_stale_aggregate_entries(platform)
                return
        raise KeyError(platform)

    def set_model_kind(self, platform: str, kind_map: dict[str, str]) -> None:
        """合并写 extra.model_kind（模型名 -> image|video|audio_tts|audio_trans）：
        探测识别出生成类签名时自动回填，也可被 UI 保存覆盖。空串值 = 清掉该模型的分类。"""
        rows = self.raw()
        for r in rows:
            if r.get("platform") == platform:
                extra = r.get("extra") or {}
                if not isinstance(extra, dict):
                    extra = {}
                mk = dict(extra.get("model_kind") or {})
                for m, k in kind_map.items():
                    if k:
                        mk[m] = k
                    else:
                        mk.pop(m, None)
                extra["model_kind"] = mk
                r["extra"] = extra
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
        """给下游 /v1/models 用：既给 "model"，也给 "platform/model"，别名也列出来，
        外加聚合模型名（聚合名以裸名暴露）。"""
        ids: list[str] = []
        seen: set[str] = set()
        for up in self.enabled():
            names = self._names(up) + [a for a in (up.model_map or {}) if a not in (up.models or [])]
            for m in names:
                for i in (m, f"{up.platform}/{m}"):
                    if i not in seen:
                        seen.add(i)
                        ids.append(i)
        for name in self.aggregates:
            if name not in seen:
                seen.add(name)
                ids.append(name)
        return ids

    def binding_options(self) -> dict[str, list[str]]:
        """给控制台「模型绑定」多选弹窗用（v1.2.2 起，与 /v1/models 的 all_model_ids 分家）。
        返回三组、组内按首字母（字典序）排序：
          full = 「平台/模型」全名（每模型/别名一行）
          bare = 裸模型名（客户端可能用裸名调用，绑裸名才放行裸名调用）
          aggs = 聚合模型名（裸名）
        绑定语义不变：门禁按精确字符串匹配；旧裸名绑定与裸名 API 入参都合法。"""
        full, bare = [], []
        seen_f, seen_b = set(), set()
        for up in self.enabled():
            names = self._names(up) + [a for a in (up.model_map or {}) if a not in (up.models or [])]
            for m in names:
                f = f"{up.platform}/{m}"
                if f not in seen_f:
                    seen_f.add(f)
                    full.append(f)
                if m not in seen_b:
                    seen_b.add(m)
                    bare.append(m)
        aggs = [n for n in self.aggregates if n not in seen_f and n not in seen_b]
        return {"full": sorted(full), "bare": sorted(bare), "aggs": sorted(aggs)}

    @staticmethod
    def _context_length(up: Upstream, model: str, target: str = "") -> int:
        """取 extra.model_context_length 的上下文窗口值（v2.0.2：/v1/models 暴露 context_length）。

        extra.model_context_length = {"上游真实模型名": 524288, "*": 32768}：
        精确模型名优先，其次别名指向的真实模型名，最后 '*' 通配兜底。
        值须为正整数（int / 数字字符串均可）；缺失 / 非正 / 非数值一律不返回（0 = 未配置）。
        """
        per_model = ((up.extra or {}).get("model_context_length")) or {}
        if not isinstance(per_model, dict):
            return 0
        for key in (model, target, "*"):
            if not key or key not in per_model:
                continue
            raw = per_model[key]
            try:
                val = int(raw) if isinstance(raw, (int, float)) else int(str(raw).strip())
            except (TypeError, ValueError):
                continue
            if val > 0:
                return val
        return 0

    def model_index(self) -> list[dict]:
        out: list[dict] = []
        for up in self.upstreams:
            usable = self.usable(up)
            for m in self._names(up):
                item = {"id": f"{up.platform}/{m}", "platform": up.platform,
                        "provider_name": up.label, "model": m,
                        "type": up.type, "enabled": usable}
                ctx = self._context_length(up, m)
                if ctx:
                    item["context_length"] = ctx
                out.append(item)
            for alias, target in (up.model_map or {}).items():
                if alias in self._names(up):
                    continue
                item = {"id": f"{up.platform}/{alias}", "platform": up.platform,
                        "provider_name": up.label, "model": alias, "maps_to": target,
                        "type": up.type, "enabled": usable}
                ctx = self._context_length(up, alias, target)
                if ctx:
                    item["context_length"] = ctx
                out.append(item)
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
