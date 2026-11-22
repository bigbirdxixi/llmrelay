"""token 用量统计：SQLite（WAL），按 平台 / 模型 / 天 聚合。

表结构按需自增列（PRAGMA table_info 比对后 ALTER TABLE），老库可平滑升级。
"""
from __future__ import annotations

import csv
import io
import json
import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path

COLUMNS: dict[str, str] = {
    "in_format": "TEXT",       # 下游协议：openai | anthropic
    "model_alias": "TEXT",     # 下游请求里写的 model
    "client": "TEXT",
    "stream": "INTEGER",
    "status": "TEXT",
    "http_status": "INTEGER",
    "error": "TEXT",
    "latency_ms": "INTEGER",
    "ttft_ms": "INTEGER",
    "prompt_tokens": "INTEGER",
    "completion_tokens": "INTEGER",
    "total_tokens": "INTEGER",
    "cached_tokens": "INTEGER",
    "cache_write_tokens": "INTEGER",
    "reasoning_tokens": "INTEGER",
    "estimated": "INTEGER",
    "client_ip": "TEXT",
    "reasoning_policy": "TEXT",
    "key_hint": "TEXT",       # 这次请求用的是哪个上游 api key（掩码后）
    "key_index": "INTEGER",   # 该 key 在平台 key 列表里的序号（0 起）
    "user_id": "INTEGER",     # 属于哪个账号（NULL = 多账号改造前的历史数据）
    "dkey_id": "INTEGER",     # 这次请求用的下游客户端 key（api_keys.id，NULL = 历史数据）
    "output_info": "TEXT",    # 生成类结果（v2.0.0 图像生成透传）：{"images": N, "urls": [...]}，chat 请求为空
    # ---- 费用（v2.1.0 R153，记账时按当时的价格配置冻结；成功请求计费，失败 0）----
    "cost": "REAL",           # 本次调用总费用（元）
    "cost_req": "REAL",       # 其中：请求费（每次调用固定金额）
    "cost_input": "REAL",     # 其中：输入费（非缓存输入 token）
    "cost_cache": "REAL",     # 其中：缓存命中费（cached_tokens）
    "cost_output": "REAL",    # 其中：输出费（completion_tokens）
}

BASE_DDL = """
CREATE TABLE IF NOT EXISTS requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    request_id TEXT NOT NULL,
    platform TEXT NOT NULL,
    model TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_req_ts ON requests (ts);
CREATE INDEX IF NOT EXISTS idx_req_plat ON requests (platform, model);

-- ---------- 多账号 ----------
CREATE TABLE IF NOT EXISTS users (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    email       TEXT NOT NULL UNIQUE,
    pass_hash   TEXT NOT NULL,
    role        TEXT NOT NULL DEFAULT 'user',     -- admin | user
    status      TEXT NOT NULL DEFAULT 'active',   -- active | disabled
    created_at  TEXT NOT NULL,
    last_login  TEXT
);

CREATE TABLE IF NOT EXISTS sessions (
    token       TEXT PRIMARY KEY,
    user_id     INTEGER NOT NULL,
    created_at  TEXT NOT NULL,
    expires_at  TEXT NOT NULL,
    ua          TEXT
);
CREATE INDEX IF NOT EXISTS idx_sess_user ON sessions (user_id);

CREATE TABLE IF NOT EXISTS api_keys (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL,
    key         TEXT NOT NULL UNIQUE,
    label       TEXT,
    created_at  TEXT NOT NULL,
    last_used   TEXT,
    revoked     INTEGER NOT NULL DEFAULT 0,
    expires_at  TEXT,              -- 过期时间（ISO 带时区；NULL/空 = 永不过期）
    model_bindings TEXT,           -- 模型绑定（JSON 数组：该 key 可调的客户端模型名；NULL/空 = 不限制）v1.2.0
    balance       REAL,           -- 余额（元，v2.1.0 R154）：NULL = 不限额只统计消费；>0 = 余额模式，成功调用原子扣费，扣成负数后拦截后续请求
    key_mac     BLOB,             -- HMAC-SHA256(K_mac, key)：验签专用（确定性、建索引，O(1)）
    key_enc     BLOB             -- AES-GCM(K_enc, key)：可逆，复制按钮取回用（NULL = 尚未迁移）
);
CREATE INDEX IF NOT EXISTS idx_key_user ON api_keys (user_id);
-- 注意：idx_key_mac（key_mac 唯一索引）不在此处建——老库的 api_keys 表可能还没有
-- key_mac 列，要先等 _migrate() 的 ALTER 加列之后才能建（否则「no such column」）。

-- ---------- key 明文取回审计（复制按钮 reveal） ----------
CREATE TABLE IF NOT EXISTS key_reveal_log (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    key_id  INTEGER NOT NULL,
    ts      TEXT NOT NULL,
    ip      TEXT,
    ua      TEXT
);

-- ---------- 邮件验证（magic link）/ 杂项 ----------
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


#: 下游 key 名称归并表达式（{p} = requests 表名/别名；v1.2.1 起原「备注」列改名「名称」）：
#: 有名称→名称；活着的空名称→(无名称)；key 已删→(key 已删除)；历史未记录→(未记录)
_DKEY_LABEL_EXPR = (
    "CASE WHEN ak.id IS NOT NULL AND COALESCE(NULLIF(TRIM(ak.label), ''), '') <> '' "
    "THEN ak.label "
    "WHEN ak.id IS NOT NULL THEN '(无名称)' "
    "WHEN {p}.dkey_id IS NOT NULL THEN '(key 已删除)' "
    "ELSE '(未记录)' END")


class Store:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(BASE_DDL)
        self._migrate()
        self._conn.commit()
        self._key_cipher = None     # app.crypto.KeyCipher（启动期注入）：验签走 key_mac，复制走 key_enc

    # ---------- schema ----------
    def _migrate(self) -> None:
        have = {r["name"] for r in self._conn.execute("PRAGMA table_info(requests)")}
        for name, typ in COLUMNS.items():
            if name not in have:
                self._conn.execute(f"ALTER TABLE requests ADD COLUMN {name} {typ}")
        # api_keys 的列级迁移（老库没有 expires_at / 加密双列）
        khave = {r["name"] for r in self._conn.execute("PRAGMA table_info(api_keys)")}
        if "expires_at" not in khave:
            self._conn.execute("ALTER TABLE api_keys ADD COLUMN expires_at TEXT")
        if "key_mac" not in khave:
            self._conn.execute("ALTER TABLE api_keys ADD COLUMN key_mac BLOB")
        if "key_enc" not in khave:
            self._conn.execute("ALTER TABLE api_keys ADD COLUMN key_enc BLOB")
        if "model_bindings" not in khave:
            self._conn.execute("ALTER TABLE api_keys ADD COLUMN model_bindings TEXT")
        if "balance" not in khave:
            self._conn.execute("ALTER TABLE api_keys ADD COLUMN balance REAL")
        # 列齐了才建唯一索引（NULL 可重复：未迁移的老行不受影响；SQLite 里 NULL 互不冲突）
        self._conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_key_mac ON api_keys (key_mac)")
        # dkey_id 是 COLUMNS 迁移出来的列（建表语句没有）→ 索引也必须在列齐后建（v2.1.0：key 消费/余额查询）
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_req_dkey ON requests (dkey_id)")

    # ---------- 写入 ----------
    def record(self, **kw) -> None:
        payload = {
            "ts": kw.get("ts") or datetime.now().astimezone().isoformat(timespec="seconds"),
            "request_id": kw.get("request_id") or "",
            "platform": kw.get("platform") or "",
            "model": kw.get("model") or "",
            "in_format": kw.get("in_format") or "openai",
            "model_alias": kw.get("model_alias") or "",
            "client": kw.get("client") or "master",
            "stream": int(bool(kw.get("stream"))),
            "status": kw.get("status") or "ok",
            "http_status": int(kw.get("http_status") or 200),
            "error": (kw.get("error") or "")[:800],
            "latency_ms": int(kw.get("latency_ms") or 0),
            "ttft_ms": int(kw.get("ttft_ms") or 0),
            "prompt_tokens": int(kw.get("prompt_tokens") or 0),
            "completion_tokens": int(kw.get("completion_tokens") or 0),
            "total_tokens": int(kw.get("total_tokens") or 0),
            "cached_tokens": int(kw.get("cached_tokens") or 0),
            "cache_write_tokens": int(kw.get("cache_write_tokens") or 0),
            "reasoning_tokens": int(kw.get("reasoning_tokens") or 0),
            "estimated": int(bool(kw.get("estimated"))),
            "client_ip": (kw.get("client_ip") or "")[:64],
            "reasoning_policy": (kw.get("reasoning_policy") or "")[:16],
            "key_hint": (kw.get("key_hint") or "")[:32],
            "key_index": int(kw.get("key_index") or 0),
            "output_info": (kw.get("output_info") or "")[:4000],
            "user_id": kw.get("user_id"),
            "dkey_id": kw.get("dkey_id"),
            # 费用（v2.1.0 R153）：gateway.record 在成功路径上按当时价格算好传入；本地 4xx 等直接
            # store.record 的路径不带 → 0。历史行缺列前为 NULL → 查询侧一律 COALESCE。
            "cost": round(float(kw.get("cost") or 0.0), 6),
            "cost_req": round(float(kw.get("cost_req") or 0.0), 6),
            "cost_input": round(float(kw.get("cost_input") or 0.0), 6),
            "cost_cache": round(float(kw.get("cost_cache") or 0.0), 6),
            "cost_output": round(float(kw.get("cost_output") or 0.0), 6),
        }
        if not payload["total_tokens"]:
            payload["total_tokens"] = payload["prompt_tokens"] + payload["completion_tokens"]
        cols = ", ".join(payload)
        marks = ", ".join("?" * len(payload))
        with self._lock:
            self._conn.execute(f"INSERT INTO requests ({cols}) VALUES ({marks})", tuple(payload.values()))
            self._conn.commit()

    # ---------- 查询 ----------
    @staticmethod
    def _since(days: int) -> str:
        days = max(1, min(int(days or 1), 3650))
        return (datetime.now() - timedelta(days=days)).isoformat(timespec="seconds")

    @staticmethod
    def _user_cond(user_id, prefix: str = "", sep: str = "AND") -> tuple[str, tuple]:
        """账号过滤片段（支持多选汇总）：None → 无条件；int → ` {sep} {prefix}user_id = ?`；
        非空 list → ` {sep} {prefix}user_id IN (?,…)`；空 list 按无过滤处理。返回 (cond, args)。"""
        if user_id is None:
            return "", ()
        if isinstance(user_id, (list, tuple, set)):
            ids = [int(u) for u in user_id]
            if not ids:
                return "", ()
            return (f" {sep} {prefix}user_id IN ({','.join('?' * len(ids))})", tuple(ids))
        return f" {sep} {prefix}user_id = ?", (int(user_id),)

    @staticmethod
    def _user_clause(user_id, sep: str = "WHERE") -> tuple[str, tuple]:
        """user_id=None 表示不按账号过滤（admin 看全站）；list 表示多账号汇总视图（IN）。"""
        return Store._user_cond(user_id, "", sep)

    def totals(self, user_id: int | list | None = None) -> dict:
        today = datetime.now().strftime("%Y-%m-%d")
        where, args = self._user_clause(user_id)
        where_today, args_today = self._user_clause(user_id)
        if where_today:
            where_today += " AND substr(ts,1,10) = ?"
            args_today = args_today + (today,)
        else:
            where_today = " WHERE substr(ts,1,10) = ?"
            args_today = (today,)
        with self._lock:
            t = self._conn.execute(
                "SELECT COUNT(*) AS r, COALESCE(SUM(total_tokens),0) AS tk,"
                " COALESCE(SUM(prompt_tokens),0) AS pt, COALESCE(SUM(completion_tokens),0) AS ct,"
                " COALESCE(SUM(cached_tokens),0) AS cache,"
                " COALESCE(AVG(latency_ms),0) AS lat"
                " FROM requests" + where, args).fetchone()
            d = self._conn.execute(
                "SELECT COUNT(*) AS r, COALESCE(SUM(total_tokens),0) AS tk,"
                " COALESCE(SUM(prompt_tokens),0) AS pt, COALESCE(SUM(completion_tokens),0) AS ct,"
                " COALESCE(SUM(CASE WHEN status!='ok' THEN 1 ELSE 0 END),0) AS errs"
                " FROM requests" + where_today, args_today).fetchone()
        return {
            "today_requests": d["r"], "today_tokens": d["tk"],
            "today_prompt_tokens": d["pt"], "today_completion_tokens": d["ct"],
            "today_errors": d["errs"],
            "total_requests": t["r"], "total_tokens": t["tk"],
            "total_prompt_tokens": t["pt"], "total_completion_tokens": t["ct"],
            "total_cached_tokens": t["cache"],
            "avg_latency_ms": round(t["lat"] or 0),
        }

    def _split_map(self, split: str, group: str, cond: str, args: list) -> dict:
        """按 桶键×(platform|model|dkey) 聚合 tokens，返回 {桶键: {名: tokens}}。"""
        if split not in ("platform", "model", "dkey"):
            return {}
        keyexpr = ("substr(ts,1,10)" if group == "day"
                   else "substr(ts,1,10) || ' ' || substr(ts,12,2)")
        if split == "dkey":
            join = " LEFT JOIN api_keys ak ON ak.id = requests.dkey_id"
            col = _DKEY_LABEL_EXPR.format(p="requests")
        else:
            join = ""
            col = "platform" if split == "platform" else "model"
        if join and cond:
            # JOIN api_keys 后 user_id 列有歧义，给 requests 加上表前缀
            cond = cond.replace("user_id =", "requests.user_id =", 1)
        with self._lock:
            srows = self._conn.execute(
                f"SELECT {keyexpr} AS k, {col} AS c, COALESCE(SUM(total_tokens),0) AS tk"
                f" FROM requests{join}{cond} GROUP BY k, c", args).fetchall()
        byk: dict = {}
        for r in srows:
            byk.setdefault(r["k"], {})[r["c"] or "unknown"] = r["tk"]
        return byk

    def hourly(self, user_id: int | list | None = None, days: int = 1,
               group: str = "hour", split: str = "none") -> dict:
        """近 N 天的活动 + 区间合计（概览看板用）。

        group="hour"：按小时分桶（近 days 天 × 24，label 为 "HH" 或 "MM-DD HH"）
        group="day"：按天分桶（label 为 "YYYY-MM-DD"）
        split="platform"/"model"：每个桶额外带 by_platform / by_model 的 tokens 拆分
        days=1 且 group="hour" 时额外返回 "today"（语义即今日合计）。
        """
        now = datetime.now()
        days = max(1, min(int(days), 90))
        where, args = self._user_clause(user_id)
        total_sql = ("SELECT COUNT(*) AS r, COALESCE(SUM(total_tokens),0) AS tk,"
                     " COALESCE(SUM(CASE WHEN status!='ok' THEN 1 ELSE 0 END),0) AS er,"
                     " COALESCE(AVG(latency_ms),0) AS lat")
        if group == "day":
            start = (now - timedelta(days=days - 1)).strftime("%Y-%m-%d")
            cond = (f"{where} AND substr(ts,1,10) >= ?" if where
                    else " WHERE substr(ts,1,10) >= ?")
            args = args + (start,)
            with self._lock:
                rows = self._conn.execute(
                    "SELECT substr(ts,1,10) AS k, COUNT(*) AS r,"
                    " COALESCE(SUM(total_tokens),0) AS tk,"
                    " COALESCE(SUM(CASE WHEN status!='ok' THEN 1 ELSE 0 END),0) AS er"
                    f" FROM requests{cond} GROUP BY k", args).fetchall()
                t = self._conn.execute(f"{total_sql} FROM requests{cond}", args).fetchone()
            got = {r["k"]: r for r in rows}
            sk = self._split_map(split, "day", cond, args) if split in ("platform", "model", "dkey") else {}
            skey = {"platform": "by_platform", "model": "by_model", "dkey": "by_dkey"}.get(split, "by_model")
            buckets = []
            for i in range(days):
                d = (now - timedelta(days=days - 1 - i)).strftime("%Y-%m-%d")
                g = got.get(d)
                buckets.append({"h": d, "requests": g["r"] if g else 0,
                                "tokens": g["tk"] if g else 0,
                                "errors": g["er"] if g else 0,
                                skey: sk.get(d, {})})
        else:
            # 日历日起始：今日 00:00 往前推 (days-1) 天，今天剩余小时补 0
            # ts 为 ISO 格式（2026-11-03T16:41:43+08:00）：日期 substr(ts,1,10)、小时 substr(ts,12,2)
            start_hour = (now.replace(hour=0, minute=0, second=0, microsecond=0)
                          - timedelta(days=days - 1))
            cutoff = start_hour.strftime("%Y-%m-%d")
            cond = (f"{where} AND substr(ts,1,10) >= ?" if where
                    else " WHERE substr(ts,1,10) >= ?")
            args = args + (cutoff,)
            with self._lock:
                rows = self._conn.execute(
                    "SELECT substr(ts,1,10) AS d, substr(ts,12,2) AS hh,"
                    " COUNT(*) AS r, COALESCE(SUM(total_tokens),0) AS tk,"
                    " COALESCE(SUM(CASE WHEN status!='ok' THEN 1 ELSE 0 END),0) AS er"
                    f" FROM requests{cond} GROUP BY d, hh", args).fetchall()
                t = self._conn.execute(f"{total_sql} FROM requests{cond}", args).fetchone()
            got = {f"{r['d']} {r['hh']}": r for r in rows}
            sk = self._split_map(split, "hour", cond, args) if split in ("platform", "model", "dkey") else {}
            skey = {"platform": "by_platform", "model": "by_model", "dkey": "by_dkey"}.get(split, "by_model")
            buckets = []
            for i in range(days * 24):
                ts_h = start_hour + timedelta(hours=i)
                g = got.get(ts_h.strftime("%Y-%m-%d %H"))
                label = ts_h.strftime("%H") if days == 1 else ts_h.strftime("%m-%d %H")
                buckets.append({"h": label, "requests": g["r"] if g else 0,
                                "tokens": g["tk"] if g else 0,
                                "errors": g["er"] if g else 0,
                                skey: sk.get(ts_h.strftime("%Y-%m-%d %H"), {})})
        total = {"requests": t["r"], "tokens": t["tk"], "errors": t["er"],
                 "avg_latency_ms": round(t["lat"] or 0)}
        out = {"group": group if group == "day" else "hour", "days": days,
               "buckets": buckets, "total": total,
               "today": total if days == 1 and group != "day" else None}
        return out

    def month_activity(self, user_id: int | list | None, year: int, month: int,
                      limit_months: int = 12) -> dict:
        """指定年月的按月活跃统计（概览页「本月活跃情况」日历）。

        返回该月 1 号 ~ 月末的逐日桶（无数据的日期补 0）+ 当月合计 +
        active_days（有请求的天数）+ months（该用户有数据的所有月份，
        倒序、上限 limit_months 个，供前端年月下拉框）。
        按 ts 前缀精确匹配年月（substr(ts,1,7)='YYYY-MM'），不受
        「近 N 天」窗口与时区漂移影响——选哪个月就只统计哪个月。
        """
        from calendar import monthrange
        ym = f"{year:04d}-{month:02d}"
        _, last_day = monthrange(year, month)
        where, args = self._user_clause(user_id)
        cond = (f"{where} AND substr(ts,1,7) = ?" if where else " WHERE substr(ts,1,7) = ?")
        args = args + (ym,)
        with self._lock:
            rows = self._conn.execute(
                "SELECT substr(ts,1,10) AS k, COUNT(*) AS r,"
                " COALESCE(SUM(total_tokens),0) AS tk,"
                " COALESCE(SUM(CASE WHEN status!='ok' THEN 1 ELSE 0 END),0) AS er"
                f" FROM requests{cond} GROUP BY k", args).fetchall()
            mw, margs = self._user_cond(user_id, "", "WHERE")
            months = [r[0] for r in self._conn.execute(
                "SELECT DISTINCT substr(ts,1,7) AS m FROM requests"
                + mw + " GROUP BY m ORDER BY m DESC LIMIT ?",
                margs + (limit_months,)).fetchall()]
        got = {r["k"]: r for r in rows}
        buckets = []
        for day in range(1, last_day + 1):
            key = f"{ym}-{day:02d}"
            g = got.get(key)
            buckets.append({"h": key, "requests": g["r"] if g else 0,
                            "tokens": g["tk"] if g else 0,
                            "errors": g["er"] if g else 0})
        total_requests = sum(b["requests"] for b in buckets)
        total_tokens = sum(b["tokens"] for b in buckets)
        active_days = sum(1 for b in buckets if b["requests"] > 0)
        return {"ym": ym, "year": year, "month": month, "days": last_day,
                "buckets": buckets, "total_requests": total_requests,
                "total_tokens": total_tokens, "active_days": active_days,
                "months": months}

    def year_activity(self, user_id: int | list | None, year: int,
                     limit_years: int = 5) -> dict:
        """指定年度的按「星期 × 月份」活跃统计（概览页活跃度日历）。

        纵轴 7 行（周一~周日）× 横轴 12 列（1~12 月），格子 = 该月该星期几的
        请求量合计；无数据的 (月,星期) 补 0。另返回年度合计、active_days
        （有请求的自然日天数）与 years（该用户有数据的年份倒序，供下拉框）。
        按 ts 年份前缀（substr(ts,1,4)）精确取数。
        """
        where, args = self._user_clause(user_id)
        cond = (f"{where} AND substr(ts,1,4) = ?" if where else " WHERE substr(ts,1,4) = ?")
        args = args + (f"{year:04d}",)
        ym = f"{year:04d}-"
        with self._lock:
            rows = self._conn.execute(
                "SELECT substr(ts,1,10) AS d, COUNT(*) AS r,"
                " COALESCE(SUM(total_tokens),0) AS tk,"
                " COALESCE(SUM(CASE WHEN status!='ok' THEN 1 ELSE 0 END),0) AS er"
                f" FROM requests{cond} GROUP BY d", args).fetchall()
            yw, yargs = self._user_cond(user_id, "", "WHERE")
            years = [r[0] for r in self._conn.execute(
                "SELECT DISTINCT substr(ts,1,4) AS y FROM requests"
                + yw + " GROUP BY y ORDER BY y DESC LIMIT ?",
                yargs + (limit_years,)).fetchall()]
        cells = [[{"requests": 0, "tokens": 0, "errors": 0} for _ in range(7)]
                 for _ in range(12)]
        active_days = 0
        total_requests = total_tokens = 0
        for r in rows:
            d = r["d"]
            if not d.startswith(ym):
                continue
            mm, dd = int(d[5:7]), int(d[8:10])
            if not (1 <= mm <= 12 and 1 <= dd <= 31):
                continue
            try:
                dow = datetime(year, mm, dd).weekday()
            except ValueError:
                continue
            c = cells[mm - 1][dow]
            c["requests"] += r["r"]
            c["tokens"] += r["tk"]
            c["errors"] += r["er"]
            total_requests += r["r"]
            total_tokens += r["tk"]
            if r["r"] > 0:
                active_days += 1
        return {"year": year, "cells": cells,
                "total_requests": total_requests, "total_tokens": total_tokens,
                "active_days": active_days, "years": years}

    def ip_counts(self, user_id: int | list | None, days: int = 1) -> list[dict]:
        """近 N 个日历天（日历天对齐，同 hourly group=day 口径）内按客户端 IP 聚合请求数。

        返回 [{"ip": str, "n": int}]（n 降序）；空 client_ip 不计入。
        供概览页地域地图：调用方再逐 IP 走 geoip.resolve 归并到省/国家。
        """
        days = max(1, min(int(days), 90))
        start = (datetime.now() - timedelta(days=days - 1)).strftime("%Y-%m-%d")
        where, args = self._user_clause(user_id)
        cond = (f"{where} AND substr(ts,1,10) >= ?" if where
                else " WHERE substr(ts,1,10) >= ?")
        args = args + (start,)
        with self._lock:
            rows = self._conn.execute(
                "SELECT client_ip AS ip, COUNT(*) AS n FROM requests"
                f"{cond} AND client_ip != '' GROUP BY client_ip"
                " ORDER BY n DESC", args).fetchall()
        return [{"ip": r["ip"], "n": r["n"]} for r in rows]

    def summary(self, group_by: str = "platform", days: int = 30,
                user_id: int | list | None = None) -> list[dict]:
        since = self._since(days)
        if group_by == "dkey":
            # 按下游客户端 key（用 api_keys.label 名称；历史无 dkey_id 的归「未记录」）
            dkey_expr = _DKEY_LABEL_EXPR.format(p="r")
            ucond, uargs = self._user_cond(user_id, "r.", "AND")
            q = f"""
                SELECT COALESCE({dkey_expr}, '') AS k,
                       COUNT(*) AS requests,
                       COALESCE(SUM(r.prompt_tokens),0) AS prompt_tokens,
                       COALESCE(SUM(r.completion_tokens),0) AS completion_tokens,
                       COALESCE(SUM(r.total_tokens),0) AS total_tokens,
                       COALESCE(SUM(r.cached_tokens),0) AS cached_tokens,
                       COALESCE(SUM(CASE WHEN r.status!='ok' THEN 1 ELSE 0 END),0) AS errors,
                       COALESCE(ROUND(AVG(r.latency_ms)),0) AS avg_latency_ms
                FROM requests r LEFT JOIN api_keys ak ON ak.id = r.dkey_id
                WHERE r.ts >= ?{ucond}
                GROUP BY 1 ORDER BY total_tokens DESC
            """
            params = (since,) + uargs
            with self._lock:
                return [dict(r) for r in self._conn.execute(q, params)]
        group = {
            "platform": "platform AS k",
            "model": "platform || '/' || model AS k",
            "model_only": "model AS k",
            "day": "substr(ts,1,10) AS k",
            "format": "in_format AS k",
            "key": "platform || ' · ' || COALESCE(NULLIF(key_hint, ''), '(未记录)') AS k",
            "account": "COALESCE(CAST(user_id AS TEXT), '(历史)') AS k",
        }.get(group_by, "platform AS k")
        ucond, uargs = self._user_cond(user_id)
        q = f"""
            SELECT {group},
                   COUNT(*) AS requests,
                   COALESCE(SUM(prompt_tokens),0) AS prompt_tokens,
                   COALESCE(SUM(completion_tokens),0) AS completion_tokens,
                   COALESCE(SUM(total_tokens),0) AS total_tokens,
                   COALESCE(SUM(cached_tokens),0) AS cached_tokens,
                   COALESCE(SUM(CASE WHEN status!='ok' THEN 1 ELSE 0 END),0) AS errors,
                   COALESCE(ROUND(AVG(latency_ms)),0) AS avg_latency_ms
            FROM requests WHERE ts >= ?{ucond}
            GROUP BY 1 ORDER BY total_tokens DESC
        """
        params = (since,) + uargs
        with self._lock:
            return [dict(r) for r in self._conn.execute(q, params)]

    #: 曲线分组表达式（键 → SQL）。前端「多选联动」靠这个把任意维度画成多条线。
    SERIES_GROUPS: dict[str, str] = {
        "platform": "platform",
        "model": "platform || '/' || model",
        "key": "platform || ' · ' || COALESCE(NULLIF(key_hint, ''), '(未记录)')",
        "dkey": _DKEY_LABEL_EXPR.format(p="r"),
        "account": "COALESCE(CAST(user_id AS TEXT), '(历史)')",
        "total": "'合计'",
    }

    def series(self, days: int = 30, group_by: str = "platform",
               user_id: int | list | None = None) -> dict:
        """按天 × 分组的时间序列。

        返回 {"days": [...], "keys": [...], "rows": [{day, total, requests, <key>: tokens}]}。
        `platforms` 是 `keys` 的别名，老前端代码继续可用。
        """
        since = self._since(days)
        dkey = group_by == "dkey"
        expr = self.SERIES_GROUPS.get(group_by, self.SERIES_GROUPS["platform"])
        if dkey:
            from_clause = "FROM requests r LEFT JOIN api_keys ak ON ak.id = r.dkey_id"
            where, uargs = self._user_cond(user_id, "r.", "AND")
            args = (since,) + uargs
            with self._lock:
                rows = [dict(r) for r in self._conn.execute(
                    f"SELECT substr(r.ts,1,10) AS day, {expr} AS k,"
                    " COALESCE(SUM(r.total_tokens),0) AS tokens,"
                    " COALESCE(SUM(r.prompt_tokens),0) AS prompt_tokens,"
                    " COALESCE(SUM(r.completion_tokens),0) AS completion_tokens,"
                    " COUNT(*) AS requests"
                    f" {from_clause} WHERE r.ts >= ?{where} GROUP BY day, k ORDER BY day",
                    args)]
        else:
            where, uargs = self._user_cond(user_id)
            args = (since,) + uargs
            with self._lock:
                rows = [dict(r) for r in self._conn.execute(
                    f"SELECT substr(ts,1,10) AS day, {expr} AS k,"
                    " COALESCE(SUM(total_tokens),0) AS tokens,"
                    " COALESCE(SUM(prompt_tokens),0) AS prompt_tokens,"
                    " COALESCE(SUM(completion_tokens),0) AS completion_tokens,"
                    " COUNT(*) AS requests"
                    f" FROM requests WHERE ts >= ?{where} GROUP BY day, k ORDER BY day",
                    args)]
        start = (datetime.now() - timedelta(days=max(1, int(days)) - 1)).date()
        day_list = [(start + timedelta(days=i)).isoformat() for i in range(max(1, int(days)))]
        keys = sorted({r["k"] or "" for r in rows})
        index = {(r["day"], r["k"] or ""): r for r in rows}
        out_rows = []
        for d in day_list:
            item = {"day": d, "total": 0, "requests": 0}
            for k in keys:
                r = index.get((d, k))
                item[k] = r["tokens"] if r else 0
                item["total"] += r["tokens"] if r else 0
                item["requests"] += r["requests"] if r else 0
            out_rows.append(item)
        return {"days": day_list, "group_by": group_by, "keys": keys,
                "platforms": keys, "rows": out_rows}

    def cost_series(self, days: int = 30, group_by: str = "model",
                   user_id: int | list | None = None) -> dict:
        """费用时间序列（v2.1.0 R155）：按天 × 分组，值 = SUM(cost)（元）+ 4 个费用分项
        （请求费/输入/缓存命中/输出，供「费用构成」堆叠展示）。结构同 series()
        （rows 每行：day + 每 key 的 cost + cost/分项合计 + requests），group_by 复用
        SERIES_GROUPS（platform/model/key/dkey/account/total）。无计费（未配价格）= 全 0。"""
        since = self._since(days)
        dkey = group_by == "dkey"
        expr = self.SERIES_GROUPS.get(group_by, self.SERIES_GROUPS["platform"])
        if dkey:
            from_clause = "FROM requests r LEFT JOIN api_keys ak ON ak.id = r.dkey_id"
            where, uargs = self._user_cond(user_id, "r.", "AND")
            tscol = "r.ts"
        else:
            from_clause = "FROM requests"
            where, uargs = self._user_cond(user_id)
            tscol = "ts"
        args = (since,) + uargs
        with self._lock:
            rows = [dict(r) for r in self._conn.execute(
                f"SELECT substr({tscol},1,10) AS day, {expr} AS k,"
                " COALESCE(SUM(cost),0) AS cost,"
                " COALESCE(SUM(cost_req),0) AS cost_req,"
                " COALESCE(SUM(cost_input),0) AS cost_input,"
                " COALESCE(SUM(cost_cache),0) AS cost_cache,"
                " COALESCE(SUM(cost_output),0) AS cost_output,"
                " COUNT(*) AS requests"
                f" {from_clause} WHERE {tscol} >= ?{where} GROUP BY day, k ORDER BY day",
                args)]
        start = (datetime.now() - timedelta(days=max(1, int(days)) - 1)).date()
        day_list = [(start + timedelta(days=i)).isoformat() for i in range(max(1, int(days)))]
        keys = sorted({r["k"] or "" for r in rows})
        index = {(r["day"], r["k"] or ""): r for r in rows}
        out_rows = []
        for d in day_list:
            item = {"day": d, "cost": 0.0, "cost_req": 0.0, "cost_input": 0.0,
                   "cost_cache": 0.0, "cost_output": 0.0, "requests": 0}
            for k in keys:
                r = index.get((d, k))
                item[k] = round(float(r["cost"] or 0), 6) if r else 0.0
                item["cost"] += float(r["cost"] or 0) if r else 0
                item["cost_req"] += float(r["cost_req"] or 0) if r else 0
                item["cost_input"] += float(r["cost_input"] or 0) if r else 0
                item["cost_cache"] += float(r["cost_cache"] or 0) if r else 0
                item["cost_output"] += float(r["cost_output"] or 0) if r else 0
                item["requests"] += int(r["requests"] or 0) if r else 0
            for f in ("cost", "cost_req", "cost_input", "cost_cache", "cost_output"):
                item[f] = round(item[f], 6)
            out_rows.append(item)
        return {"days": day_list, "group_by": group_by, "keys": keys,
                "platforms": keys, "rows": out_rows}

    @staticmethod
    def _p95(vals: list[float]) -> int:
        if not vals:
            return 0
        vs = sorted(vals)
        idx = min(len(vs) - 1, int(0.95 * len(vs)))
        return int(round(vs[idx]))

    def period(self, days: int = 30, user_id: int | list | None = None) -> dict:
        """区间 KPI：请求/tokens/平均&P95 耗时/TTFT/缓存命中/错误/流式/推理 + 费用（v2.1.0）。"""
        since = self._since(days)
        where, args = self._user_clause(user_id)
        cond = (f"{where} AND ts >= ?" if where else " WHERE ts >= ?")
        params = args + (since,)
        with self._lock:
            t = self._conn.execute(
                "SELECT COUNT(*) AS r, COALESCE(SUM(total_tokens),0) AS tk,"
                " COALESCE(SUM(cached_tokens),0) AS cache,"
                " COALESCE(SUM(reasoning_tokens),0) AS reason,"
                " COALESCE(SUM(CASE WHEN status!='ok' THEN 1 ELSE 0 END),0) AS er,"
                " COALESCE(SUM(CASE WHEN stream=1 THEN 1 ELSE 0 END),0) AS st,"
                " COALESCE(AVG(CASE WHEN status='ok' THEN latency_ms END),0) AS lat,"
                " COALESCE(AVG(CASE WHEN status='ok' THEN ttft_ms END),0) AS ttft,"
                " COALESCE(SUM(cost),0) AS cost,"
                " COALESCE(SUM(cost_req),0) AS creq, COALESCE(SUM(cost_input),0) AS cin,"
                " COALESCE(SUM(cost_cache),0) AS ccache, COALESCE(SUM(cost_output),0) AS cout"
                f" FROM requests{cond}", params).fetchone()
            # P95 只算成功请求（失败请求的耗时没有统计意义）
            ok_cond = (f"{where} AND status='ok' AND ts >= ?" if where
                       else " WHERE status='ok' AND ts >= ?")
            lats = [row[0] for row in self._conn.execute(
                f"SELECT latency_ms FROM requests{ok_cond}", params).fetchall()
                if row[0] is not None]
        total, cached = t["tk"] or 0, t["cache"] or 0
        requests_n = t["r"] or 0
        cost = round(float(t["cost"] or 0), 6)
        return {"requests": requests_n, "tokens": total,
                "cached_tokens": cached,
                "cache_rate": round(cached / total, 4) if total else 0,
                "reasoning_tokens": t["reason"] or 0,
                "errors": t["er"] or 0,
                "success_rate": round((requests_n - (t["er"] or 0)) / requests_n, 4) if requests_n else 1.0,
                "stream_requests": t["st"] or 0,
                "stream_rate": round((t["st"] or 0) / requests_n, 4) if requests_n else 0,
                "avg_latency_ms": round(t["lat"] or 0),
                "p95_latency_ms": self._p95(lats),
                "avg_ttft_ms": round(t["ttft"] or 0),
                # v2.1.0 R155：区间费用（成功请求计费合计；未配置价格 = 0）
                "cost": cost,
                "cost_req": round(float(t["creq"] or 0), 6),
                "cost_input": round(float(t["cin"] or 0), 6),
                "cost_cache": round(float(t["ccache"] or 0), 6),
                "cost_output": round(float(t["cout"] or 0), 6),
                "cost_per_request": round(cost / requests_n, 6) if requests_n else 0.0}

    def latency_series(self, days: int = 30, user_id: int | list | None = None) -> dict:
        """每日耗时序列（仅成功请求）：avg / p95 / TTFT 平均。"""
        since = self._since(days)
        where, args = self._user_clause(user_id)
        cond = (f"{where} AND ts >= ?" if where else " WHERE ts >= ?")
        params = args + (since,)
        with self._lock:
            rows = [dict(r) for r in self._conn.execute(
                "SELECT substr(ts,1,10) AS d, latency_ms AS l, ttft_ms AS t"
                f" FROM requests{cond} AND status='ok'", params)]
        start = (datetime.now() - timedelta(days=max(1, int(days)) - 1)).date()
        day_list = [(start + timedelta(days=i)).isoformat() for i in range(max(1, int(days)))]
        by_day: dict[str, list] = {}
        for r in rows:
            by_day.setdefault(r["d"], []).append(r)
        out_rows = []
        for d in day_list:
            grp = by_day.get(d, [])
            lats = [x["l"] for x in grp if x["l"] is not None]
            tfts = [x["t"] for x in grp if x["t"]]
            out_rows.append({"day": d,
                             "avg": int(round(sum(lats) / len(lats))) if lats else 0,
                             "p95": self._p95(lats),
                             "ttft": int(round(sum(tfts) / len(tfts))) if tfts else 0,
                             "requests": len(lats)})
        return {"days": day_list, "group_by": "day",
                "keys": ["avg", "p95", "ttft"], "rows": out_rows}

    def heatmap(self, days: int = 30, user_id: int | list | None = None) -> dict:
        """星期 × 小时 请求量矩阵（7 行周一..周日 × 24 列）。"""
        since = self._since(days)
        where, args = self._user_clause(user_id)
        cond = (f"{where} AND ts >= ?" if where else " WHERE ts >= ?")
        params = args + (since,)
        with self._lock:
            rows = [dict(r) for r in self._conn.execute(
                "SELECT substr(ts,1,10) AS d, CAST(substr(ts,12,2) AS INTEGER) AS h,"
                " COUNT(*) AS c FROM requests"
                f"{cond} GROUP BY d, h", params)]
        matrix = [[0] * 24 for _ in range(7)]
        mx = 0
        for r in rows:
            try:
                wd = datetime.fromisoformat(r["d"]).weekday()
                h = int(r["h"] or 0)
            except (ValueError, TypeError):
                continue
            if 0 <= wd < 7 and 0 <= h < 24:
                matrix[wd][h] = int(r["c"])
                mx = max(mx, matrix[wd][h])
        return {"days": days, "matrix": matrix, "max": mx,
                "labels": ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]}

    def mix(self, days: int = 30, user_id: int | list | None = None) -> dict:
        """区间结构：下游协议 / 流式 / 状态 / 思考策略 / 错误 HTTP 码分布。"""
        since = self._since(days)
        where, args = self._user_clause(user_id)
        cond = (f"{where} AND ts >= ?" if where else " WHERE ts >= ?")
        params = args + (since,)
        with self._lock:
            def _by(col_sql: str, where_extra: str = "") -> dict:
                rows = [dict(r) for r in self._conn.execute(
                    f"SELECT {col_sql} AS k, COUNT(*) AS c FROM requests"
                    f"{cond}{where_extra} GROUP BY k", params)]
                return {str(r["k"] or "(未记录)"): r["c"] for r in rows}
            by_in = _by("in_format")
            by_stream = { "流式": 0, "普通": 0}
            by_status = _by("status")
            by_reason = _by("reasoning_policy")
            http = [dict(r) for r in self._conn.execute(
                "SELECT http_status AS k, COUNT(*) AS c FROM requests"
                f"{cond} AND status!='ok' AND http_status IS NOT NULL GROUP BY k", params)]
            for r in [dict(x) for x in self._conn.execute(
                    "SELECT stream AS k, COUNT(*) AS c FROM requests" + cond + " GROUP BY k", params)]:
                by_stream["流式" if int(r["k"] or 0) == 1 else "普通"] = r["c"]
        return {"by_in_format": by_in, "by_stream": by_stream, "by_status": by_status,
                "by_reasoning": {k: v for k, v in by_reason.items() if k != "(未记录)"},
                "by_http_error": {str(h["k"]): h["c"] for h in http}}

    def top_errors(self, days: int = 30, limit: int = 8, user_id: int | list | None = None) -> list[dict]:
        """区间内 Top 错误原因（截断 140 字符），按次数降序。"""
        since = self._since(days)
        where, args = self._user_clause(user_id)
        cond = (f"{where} AND ts >= ?" if where else " WHERE ts >= ?")
        params = args + (since,) + (max(1, min(int(limit), 50)),)
        with self._lock:
            rows = [dict(r) for r in self._conn.execute(
                "SELECT substr(error,1,140) AS reason, http_status,"
                " COUNT(*) AS c, MAX(ts) AS last FROM requests"
                f"{cond} AND status!='ok' AND error IS NOT NULL AND error != ''"
                " GROUP BY reason, http_status ORDER BY c DESC LIMIT ?", params)]
        for r in rows:
            r["reason"] = r["reason"] or "(无错误详情)"
        return rows

    def _in_clause(self, col: str, vals: list, args: list, conds: list) -> None:
        """vals 非空时生成 IN (?) 子句；空列表不加条件。
        "" 是合法筛选值（「（未解析）」「-」占位经 stats_records 转回空串后按空匹配，R141/R142），
        只跳过 None，不再静默丢掉空串。"""
        vals = [v for v in (vals or []) if v is not None]
        if vals:
            conds.append(f"{col} IN ({', '.join('?' * len(vals))})")
            args.extend(vals)

    def _log_filters(self, user_id: int | list | None, days: int,
                     platform: str | list | None, model: str | list | None,
                     status: str | list | None,
                     stream: list | None = None, key_hint: list | None = None,
                     client_ip: list | None = None, user_ids: list | None = None,
                     date_from: str | None = None, date_to: str | None = None,
                     dkey_ids: list | None = None
                     ) -> tuple[list[str], list]:
        """统一的日志筛选条件构造（recent / record_count 共用，保证总数与列表一致）。
        dkey_ids = 下游 key id 列表；0 是哨兵 = 「未记录」（dkey_id IS NULL，历史数据）。"""
        conds: list[str] = []
        args: list = []
        uc, ua = self._user_cond(user_id, "", "")
        if uc:
            conds.append(uc)
            args.extend(ua)
        if days:
            conds.append("ts >= ?")
            args.append(self._since(days))
        if isinstance(platform, list):
            self._in_clause("platform", platform, args, conds)
        elif platform:
            conds.append("platform = ?")
            args.append(platform)
        if isinstance(model, list):
            self._in_clause("model", model, args, conds)
        elif model:
            conds.append("model = ?")
            args.append(model)
        if isinstance(status, list):
            self._in_clause("status", status, args, conds)
        elif status:
            conds.append("status = ?")
            args.append(status)
        self._in_clause("stream", stream, args, conds)
        self._in_clause("key_hint", key_hint, args, conds)
        self._in_clause("client_ip", client_ip, args, conds)
        self._in_clause("user_id", user_ids, args, conds)
        if dkey_ids:
            d_in = [i for i in dkey_ids if i]
            d_null = 0 in dkey_ids
            parts: list[str] = []
            if d_in:
                parts.append(f"dkey_id IN ({','.join('?' * len(d_in))})")
                args.extend(d_in)
            if d_null:
                parts.append("dkey_id IS NULL")
            conds.append("(" + " OR ".join(parts) + ")")
        if date_from:
            conds.append("ts >= ?")
            args.append(self._ts_boundary(date_from, end=False))
        if date_to:
            conds.append("ts <= ?")
            args.append(self._ts_boundary(date_to, end=True))
        return conds, args

    @staticmethod
    def _ts_boundary(value: str, end: bool) -> str:
        """前端 from/to（纯日期或到秒的时间、无时区）→ ts 可比较边界。
        ts 存的是本地时区 ISO（YYYY-MM-DDTHH:MM:SS+08:00），边界按同样规则生成：
        - 纯日期 YYYY-MM-DD：起边界取当天 00:00:00、终边界取当天 23:59:59（兼容旧行为）；
        - 日期时间 YYYY-MM-DDTHH:MM[:SS]：原样使用（精确到秒）。"""
        s = str(value or "").strip()[:19]
        if not s:
            return ""
        if len(s) == 10:
            s += "T23:59:59" if end else "T00:00:00"
        elif len(s) == 16:
            s += ":00"
        return datetime.fromisoformat(s).astimezone().isoformat(timespec="seconds")

    @staticmethod
    def resolve_range(r: str | None) -> tuple[str | None, str | None]:
        """相对快捷区间（15m/30m/1h/6h/today）→ (from, to) 两个 ts 可比较边界；
        以服务端当前时间为锚点；未知值返回 (None, None)。"""
        r = (r or "").strip().lower()
        span = {"15m": 15 * 60, "30m": 30 * 60, "1h": 3600, "6h": 6 * 3600}
        now = datetime.now()
        to = now.astimezone().isoformat(timespec="seconds")
        if r == "today":
            lo = now.replace(hour=0, minute=0, second=0, microsecond=0)
            return lo.astimezone().isoformat(timespec="seconds"), to
        if r in span:
            lo = now - timedelta(seconds=span[r])
            return lo.astimezone().isoformat(timespec="seconds"), to
        return None, None

    def _log_q(self, conds: list[str], q: str = "") -> str:
        return (q + (" WHERE " + " AND ".join(conds) if conds else ""))

    def recent(self, limit: int = 100, platform: str | None = None, model: str | None = None,
               status: str | None = None, days: int = 0,
               user_id: int | list | None = None, offset: int = 0,
               stream: list | None = None, key_hint: list | None = None,
               client_ip: list | None = None, user_ids: list | None = None,
               date_from: str | None = None, date_to: str | None = None,
               dkey_ids: list | None = None) -> list[dict]:
        conds, args = self._log_filters(
            user_id, days, platform, model, status, stream, key_hint,
            client_ip, user_ids, date_from, date_to, dkey_ids)
        q = self._log_q(conds, "SELECT * FROM requests")
        q += " ORDER BY id DESC LIMIT ? OFFSET ?"
        args.append(max(1, min(int(limit), 1000)))
        args.append(max(0, int(offset)))
        with self._lock:
            rows = [dict(r) for r in self._conn.execute(q, args)]
        # 下游 key 名称：dkey_id → api_keys.label（key 被删了显示提示）
        ids = {r.get("dkey_id") for r in rows if r.get("dkey_id")}
        lab: dict[int, str] = {}
        if ids:
            marks = ",".join("?" * len(ids))
            for r in self._conn.execute(
                    f"SELECT id, label FROM api_keys WHERE id IN ({marks})", tuple(ids)):
                lab[r["id"]] = r["label"] or ""
        for r in rows:
            if r.get("dkey_id"):
                # 与 _DKEY_LABEL_EXPR 保持一致：活着的空名称→（无名称）；已删→（key 已删除）
                d = r["dkey_id"]
                r["dkey_label"] = (lab[d] or "（无名称）") if d in lab else "（key 已删除）"
            else:
                r["dkey_label"] = ""
        return rows

    def record_count(self, days: int = 0, user_id: int | list | None = None,
                    **kw) -> int:
        """kw 同 recent() 的筛选参数（platform/model/status 列表、stream、key_hint、
        client_ip、user_ids、date_from、date_to）。"""
        conds, args = self._log_filters(
            user_id, days,
            kw.get("platform"), kw.get("model"), kw.get("status"),
            kw.get("stream"), kw.get("key_hint"), kw.get("client_ip"),
            kw.get("user_ids"), kw.get("date_from"), kw.get("date_to"),
            kw.get("dkey_ids"))
        q = self._log_q(conds, "SELECT COUNT(*) FROM requests")
        with self._lock:
            return int(self._conn.execute(q, args).fetchone()[0])

    def log_facets(self, user_id: int | list | None = None,
                   date_from: str | None = None, date_to: str | None = None,
                   **filters) -> dict:
        """日志筛选项。

        v2.0.1 起不再全量：选项按「当前时间范围 + 其他已生效筛选」罗列（即当前筛选
        结果集），选项与表内数据对应，不再出现「选项有、勾选后 0 条」；计算某一维度
        的选项时不加该维度自己的筛选（自身排除，勾了值仍可切换其他值——标准 facet 语义）。
        也顺带避免日志页每次翻页/改筛选都对全表跑一遍 distinct（性能）。
        filters：与 recent() 同语义的列表——platform/model/status/stream/key_hint/
        client_ip/user_ids/dkey_ids。"""
        def _conds(exclude: tuple) -> tuple[list, list]:
            kw = dict(stream=filters.get("stream"), key_hint=filters.get("key_hint"),
                      client_ip=filters.get("client_ip"), user_ids=filters.get("user_ids"),
                      dkey_ids=filters.get("dkey_ids"))
            for name in ("platform", "model", "status"):
                kw[name] = filters.get(name)
            for name in exclude:
                kw[name] = None
            return self._log_filters(
                user_id, 0, kw["platform"], kw["model"], kw["status"],
                stream=kw["stream"], key_hint=kw["key_hint"],
                client_ip=kw["client_ip"], user_ids=kw["user_ids"],
                date_from=date_from, date_to=date_to, dkey_ids=kw["dkey_ids"])
        with self._lock:
            def _vals(col: str, default: str = "", exclude: tuple = ()) -> list:
                conds, args = _conds(exclude)
                rows = [r[0] for r in self._conn.execute(
                    self._log_q(conds, f"SELECT DISTINCT {col} FROM requests"), args)]
                return [r if r not in (None, "") else default for r in rows]
            stream_vals = _vals("stream", "0", ("stream",))
            conds, args = _conds(("user_ids",))
            uid_rows = [r[0] for r in self._conn.execute(
                self._log_q(conds, "SELECT DISTINCT user_id FROM requests"), args)]
            conds, args = _conds(("dkey_ids",))
            # 下游 key 筛选项：distinct dkey_id（0 = 未记录/NULL），标签解析与 recent() 一致
            dk_rows = [r[0] for r in self._conn.execute(
                self._log_q(conds, "SELECT DISTINCT dkey_id FROM requests"), args)]
            dk_ids = sorted({int(x) for x in dk_rows if x is not None})
            dk_map: dict[int, str] = {}
            if dk_ids:
                marks = ",".join("?" * len(dk_ids))
                for r in self._conn.execute(
                        f"SELECT id, label FROM api_keys WHERE id IN ({marks})", tuple(dk_ids)):
                    dk_map[r["id"]] = r["label"] or ""
            # 选项 distinct 也统一在锁内做完（_vals 是闭包，锁外再调等于裸跑 SQL）
            plat_vals = _vals("platform", "（未解析）")
            model_vals = _vals("model", "（未解析）")
            status_vals = _vals("status")
            key_vals = _vals("key_hint", "-")
            ip_vals = _vals("client_ip", "-")
            # min/max/total：「范围 + 全部筛选」的当前结果集（与 records 同口径）
            conds, args = _conds(())
            rng = self._conn.execute(
                self._log_q(conds, "SELECT MIN(ts), MAX(ts) FROM requests"), args).fetchone()
            total = int(self._conn.execute(
                self._log_q(conds, "SELECT COUNT(*) FROM requests"), args).fetchone()[0])
        # 全量映射所有用户（小表）：单账号视图（int）原来没有映射分支 → accounts 全空 email
        # → 日志页「账号」筛选框一个选项都没有（R140）。list_users 取锁，必须在 with 外。
        emails = {str(u["id"]): u["email"] for u in self.list_users()}
        # user_id 指向已删除账号：带编号占位（唯一，前端可按标签反查 id 继续筛选）。
        accounts = [{"id": i, "email": emails.get(str(i), f"（用户已删除 #{i}）")}
                    for i in sorted(x for x in uid_rows if x is not None)]
        dkeys: list[dict] = []
        if any(x is None for x in dk_rows):
            dkeys.append({"id": 0, "label": "（未记录）"})
        for i in dk_ids:
            dkeys.append({"id": i, "label": (dk_map[i] or "（无名称）") if i in dk_map else "（key 已删除）"})
        return {
            # 本地审计行（401/403/生成类本地 4xx 等）此时尚未解析到平台，DB 里存空串；
            # 映射为「（未解析）」占位，筛选项不再出现空白项（R141）。筛选侧由
            # admin.py 的 stats_records 把占位转回空串去匹配。
            "platform": plat_vals,
            "model": model_vals,
            "status": status_vals,
            # stream 列是 INTEGER（0/1/NULL）：原来只认字符串 "0"/"1"，int 值永远匹配不上
            # → facet 恒为空 → 日志页「类型」筛选框一个选项都没有（R140）
            "stream": sorted({int(x) for x in stream_vals if str(x) in ("0", "1")}),
            "key_hint": key_vals,
            "client_ip": ip_vals,
            "accounts": accounts,
            "dkeys": dkeys,
            "min_ts": rng[0] or "", "max_ts": rng[1] or "",
            "total": total,
        }

    def model_options(self, user_id: int | list | None = None) -> list[dict]:
        where, args = self._user_clause(user_id)
        with self._lock:
            return [dict(r) for r in self._conn.execute(
                "SELECT platform, model, COUNT(*) AS requests,"
                " COALESCE(SUM(total_tokens),0) AS total_tokens"
                " FROM requests" + where +
                " GROUP BY platform, model ORDER BY total_tokens DESC", args)]

    def export_csv(self, days: int = 30, user_id: int | list | None = None) -> str:
        since = self._since(days)
        sql = "SELECT * FROM requests WHERE ts >= ?"
        params: tuple = (since,)
        uc, ua = self._user_cond(user_id)
        sql += uc
        params = params + ua
        with self._lock:
            rows = [dict(r) for r in self._conn.execute(sql + " ORDER BY id", params)]
        buf = io.StringIO()
        if rows:
            w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        return buf.getvalue()

    # ================= 账号 / 会话 / API Key =================
    def _now(self) -> str:
        return datetime.now().astimezone().isoformat(timespec="seconds")

    def create_user(self, email: str, pass_hash: str, role: str = "user",
                    status: str = "active") -> dict:
        email = (email or "").strip().lower()
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO users (email, pass_hash, role, status, created_at) VALUES (?,?,?,?,?)",
                (email, pass_hash, role, status, self._now()))
            self._conn.commit()
            uid = cur.lastrowid
        return self.get_user(uid)

    # ---------- meta（杂项键值：邮件验证签名密钥、重发限频等） ----------
    def get_meta(self, key: str) -> str | None:
        with self._lock:
            r = self._conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return r["value"] if r else None

    def set_meta(self, key: str, value: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO meta (key, value) VALUES (?,?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value", (key, value))
            self._conn.commit()

    def get_user(self, user_id: int) -> dict | None:
        with self._lock:
            r = self._conn.execute("SELECT * FROM users WHERE id = ?", (int(user_id),)).fetchone()
        return dict(r) if r else None

    def get_user_by_email(self, email: str) -> dict | None:
        with self._lock:
            r = self._conn.execute("SELECT * FROM users WHERE email = ?",
                                   ((email or "").strip().lower(),)).fetchone()
        return dict(r) if r else None

    def first_admin(self) -> dict | None:
        """取第一个管理员账号（ensure_admin / 遗留租户认领用）。"""
        with self._lock:
            r = self._conn.execute(
                "SELECT * FROM users WHERE role = 'admin' AND status = 'active' ORDER BY id LIMIT 1"
            ).fetchone()
        return dict(r) if r else None

    def list_users(self) -> list[dict]:
        with self._lock:
            rows = [dict(r) for r in self._conn.execute(
                "SELECT id, email, role, status, created_at, last_login FROM users ORDER BY id")]
        # 顺带带上每个账号的用量概览，admin 页面直接用
        for u in rows:
            agg = self.conn_exec(
                "SELECT COUNT(*) AS r, COALESCE(SUM(total_tokens),0) AS tk FROM requests WHERE user_id = ?",
                (u["id"],))
            u["requests"] = agg["r"] if agg else 0
            u["total_tokens"] = agg["tk"] if agg else 0
        return rows

    def conn_exec(self, sql: str, args: tuple = ()):
        with self._lock:
            r = self._conn.execute(sql, args).fetchone()
        return dict(r) if r else None

    def count_users(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]

    def update_user(self, user_id: int, **fields) -> dict | None:
        allowed = {"email", "pass_hash", "role", "status", "last_login"}
        sets = {k: v for k, v in fields.items() if k in allowed}
        if not sets:
            return self.get_user(user_id)
        cols = ", ".join(f"{k} = ?" for k in sets)
        with self._lock:
            self._conn.execute(f"UPDATE users SET {cols} WHERE id = ?",
                               tuple(sets.values()) + (int(user_id),))
            self._conn.commit()
        return self.get_user(user_id)

    def adopt_orphan_records(self, user_id: int) -> int:
        """把多账号改造前的历史记录归到指定账号名下。"""
        with self._lock:
            cur = self._conn.execute("UPDATE requests SET user_id = ? WHERE user_id IS NULL OR user_id = 0",
                                     (int(user_id),))
            self._conn.commit()
            return cur.rowcount

    def create_session(self, user_id: int, token: str, expires_at: str, ua: str = "") -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO sessions (token, user_id, created_at, expires_at, ua)"
                " VALUES (?,?,?,?,?)", (token, int(user_id), self._now(), expires_at, ua[:200]))
            self._conn.commit()

    def get_session(self, token: str) -> dict | None:
        if not token:
            return None
        with self._lock:
            r = self._conn.execute("SELECT * FROM sessions WHERE token = ?", (token,)).fetchone()
        if not r:
            return None
        if str(r["expires_at"]) <= self._now():
            self.delete_session(token)
            return None
        return dict(r)

    def delete_session(self, token: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
            self._conn.commit()

    def delete_user_sessions(self, user_id: int) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM sessions WHERE user_id = ?", (int(user_id),))
            self._conn.commit()

    def purge_expired_sessions(self) -> int:
        with self._lock:
            cur = self._conn.execute("DELETE FROM sessions WHERE expires_at <= ?", (self._now(),))
            self._conn.commit()
            return cur.rowcount

    def purge_stale_pending(self, max_age_s: int = 30 * 60) -> int:
        """清理注册后超时未确认邮箱的 pending 账号（连同其会话与 Key），
        释放邮箱使其可以重新注册。返回被清理的账号数。"""
        cutoff = (datetime.now().astimezone() - timedelta(seconds=max_age_s)).isoformat(timespec="seconds")
        with self._lock:
            rows = self._conn.execute(
                "SELECT id FROM users WHERE status = 'pending' AND created_at < ?", (cutoff,)).fetchall()
            for r in rows:
                uid = r[0]
                self._conn.execute("DELETE FROM sessions WHERE user_id = ?", (uid,))
                self._conn.execute("DELETE FROM api_keys WHERE user_id = ?", (uid,))
                self._conn.execute("DELETE FROM users WHERE id = ?", (uid,))
            self._conn.commit()
        return len(rows)

    def delete_user_full(self, user_id: int) -> bool:
        """彻底删除账号：会话、API Key、用户行（请求历史保留作审计）。"""
        with self._lock:
            r = self._conn.execute("SELECT id FROM users WHERE id = ?", (int(user_id),)).fetchone()
            if not r:
                return False
            self._conn.execute("DELETE FROM sessions WHERE user_id = ?", (int(user_id),))
            self._conn.execute("DELETE FROM api_keys WHERE user_id = ?", (int(user_id),))
            self._conn.execute("DELETE FROM users WHERE id = ?", (int(user_id),))
            self._conn.commit()
        return True

    # ---------- 密钥加密（v1.1.0：key_mac 验签索引 / key_enc 可逆取回） ----------
    def set_key_cipher(self, cipher) -> None:
        """启动期注入 KeyCipher；之后 create/验签/reveal 全部走加密双列。"""
        self._key_cipher = cipher

    def has_encrypted_keys(self) -> bool:
        """库里是否已有加密存储的 key（主密钥缺失时用于 fail-fast 判断）。"""
        with self._lock:
            return self._conn.execute(
                "SELECT COUNT(*) FROM api_keys WHERE key_mac IS NOT NULL").fetchone()[0] > 0

    def migrate_keys(self) -> int:
        """一次性把历史明文 key 补上 key_mac/key_enc（幂等；未迁移的行仍走明文兜底验签）。"""
        if self._key_cipher is None:
            return 0
        c = self._key_cipher
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, key FROM api_keys WHERE key_mac IS NULL AND key IS NOT NULL").fetchall()
            n = 0
            for r in rows:
                self._conn.execute(
                    "UPDATE api_keys SET key_mac = ?, key_enc = ? WHERE id = ?",
                    (c.mac(r["key"]), c.seal(r["key"]), r["id"]))
                n += 1
            self._conn.commit()
            if n:
                # 回读 round-trip 校验：抽一条确认密文解得回来（全表小，直接全验）
                for r in rows:
                    enc = self._conn.execute(
                        "SELECT key_enc FROM api_keys WHERE id = ?", (r["id"],)).fetchone()[0]
                    assert c.open(enc) == r["key"], f"迁移后 round-trip 校验失败: api_keys.id={r['id']}"
        return n

    def reveal_api_key(self, key_id: int) -> str | None:
        """复制按钮：解出某把 key 的明文。未迁移的历史行兜底读旧明文列。"""
        with self._lock:
            r = self._conn.execute(
                "SELECT key, key_enc FROM api_keys WHERE id = ?", (int(key_id),)).fetchone()
        if not r:
            return None
        if r["key_enc"] is not None and self._key_cipher is not None:
            return self._key_cipher.open(r["key_enc"])
        return r["key"]

    def log_key_reveal(self, user_id: int, key_id: int, ip: str, ua: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO key_reveal_log (user_id, key_id, ts, ip, ua) VALUES (?,?,?,?,?)",
                (int(user_id), int(key_id), self._now(), (ip or "")[:64], (ua or "")[:120]))
            self._conn.commit()

    def count_key_reveals(self, key_id: int) -> int:
        with self._lock:
            return self._conn.execute(
                "SELECT COUNT(*) FROM key_reveal_log WHERE key_id = ?", (int(key_id),)).fetchone()[0]

    # ---------- 模型绑定（key 白名单）存取 ----------
    @staticmethod
    def _models_to_json(models) -> str | None:
        """绑定列表 → 存储形式：None/空列表 = NULL（不限制）；否则 JSON 串（去重 + 排序，展示稳定）。"""
        if not models:
            return None
        clean = sorted({str(m).strip() for m in models if str(m).strip()})
        return json.dumps(clean, ensure_ascii=False) if clean else None

    @staticmethod
    def _models_from_json(raw) -> list | None:
        """存储形式 → 绑定列表：NULL/空/坏值 = None（不限制）。"""
        if not raw:
            return None
        try:
            v = json.loads(raw)
            if isinstance(v, list):
                out = [str(x).strip() for x in v if str(x).strip()]
                return out or None
        except (TypeError, ValueError):
            return None
        return None

    def create_api_key(self, user_id: int, key: str, label: str = "",
                       expires_at: str | None = None,
                       model_bindings: list | None = None,
                       balance: float | None = None) -> dict:
        """expires_at：ISO 带时区字符串；None/空 = 永不过期。
        model_bindings：该 key 可调的客户端模型/聚合名（原始请求串形式）；None/空 = 不限制。
        balance（v2.1.0 R154）：余额（元）；None/0 = 不限额只统计消费；>0 = 余额模式（成功调用扣费，
        扣成负数后拦截）。加密双列（key_mac/key_enc）随主键一起写（cipher 未注入时留 NULL，验签走明文兜底）。"""
        c = self._key_cipher
        mac = c.mac(key) if c else None
        enc = c.seal(key) if c else None
        bindings = self._models_to_json(model_bindings)
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO api_keys (user_id, key, label, created_at, expires_at, key_mac, key_enc, model_bindings, balance)"
                " VALUES (?,?,?,?,?,?,?,?,?)",
                (int(user_id), key, (label or "")[:60], self._now(),
                 (expires_at or "").strip() or None, mac, enc, bindings,
                 self._normalize_balance(balance)))
            self._conn.commit()
            kid = cur.lastrowid
        return self.get_api_key(kid)

    @staticmethod
    def _normalize_balance(balance) -> float | None:
        """余额入参归一：None/空/0 → NULL（不限额，只统计消费）；>0 → float（元，6 位小数）。
        负数拒绝：负余额是扣减产生的内部状态（限额标志），不允许手工设置——
        误传负数若归一成 NULL 会静默取消限额，必须 400。"""
        if balance is None:
            return None
        s = str(balance).strip()
        if not s:
            return None
        try:
            v = float(s)
        except ValueError:
            raise ValueError("余额必须是数字（元）")
        if v < 0:
            raise ValueError("余额不能为负数（负余额是扣减产生的内部状态，不能手工设置）")
        return None if v == 0 else round(v, 6)

    def update_key_expires(self, user_id: int, key_id: int,
                           expires_at: str | None) -> bool:
        """改 key 过期时间（None/空 = 永不过期）。"""
        with self._lock:
            cur = self._conn.execute(
                "UPDATE api_keys SET expires_at = ? WHERE id = ? AND user_id = ?",
                ((expires_at or "").strip() or None, int(key_id), int(user_id)))
            self._conn.commit()
            return cur.rowcount > 0

    def set_key_label(self, user_id: int, key_id: int, label: str) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "UPDATE api_keys SET label = ? WHERE id = ? AND user_id = ?",
                ((label or "").strip()[:60], int(key_id), int(user_id)))
            self._conn.commit()
            return cur.rowcount > 0

    def set_key_balance(self, user_id: int, key_id: int, balance) -> bool:
        """改 key 余额（v2.1.0 R154）：None/空/0 = 不限额（NULL）；>0 = 余额模式。
        管理操作是覆盖写（直接设值），不是增量充值。"""
        with self._lock:
            cur = self._conn.execute(
                "UPDATE api_keys SET balance = ? WHERE id = ? AND user_id = ?",
                (self._normalize_balance(balance), int(key_id), int(user_id)))
            self._conn.commit()
            return cur.rowcount > 0

    def charge_key(self, key_id: int, cost: float) -> None:
        """成功调用后的余额扣减（v2.1.0 R154）：单条原子 UPDATE（无读改写竞态），
        可扣成负数——负数 = 限额标志，下一次请求入口预检拦截。balance 为 NULL（不限额）不动。
        SQL 内 round(·,6) 防多次扣减的浮点噪声累积（SQLite 内置 round 函数）。"""
        if cost is None or cost <= 0:
            return
        with self._lock:
            self._conn.execute(
                "UPDATE api_keys SET balance = round(balance - ?, 6)"
                " WHERE id = ? AND balance IS NOT NULL",
                (round(float(cost), 6), int(key_id)))
            self._conn.commit()

    def key_spends(self, user_id: int | None = None) -> dict[int, float]:
        """各下游 key 的累计消费（元，全时段；v2.1.0）：{api_keys.id: spent}。
        一次 GROUP BY 查完（避免逐 key N+1）；key 列表页与管理员汇总视图共用。"""
        with self._lock:
            if user_id is None:
                rows = self._conn.execute(
                    "SELECT dkey_id, COALESCE(SUM(cost), 0) AS s FROM requests"
                    " WHERE dkey_id IS NOT NULL GROUP BY dkey_id").fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT r.dkey_id, COALESCE(SUM(r.cost), 0) AS s FROM requests r"
                    " JOIN users u ON u.id = r.user_id WHERE r.dkey_id IS NOT NULL AND u.id = ?"
                    " GROUP BY r.dkey_id", (int(user_id),)).fetchall()
        return {int(r["dkey_id"]): round(float(r["s"] or 0), 6) for r in rows}

    def key_spend(self, key_id: int) -> float:
        """单把 key 的累计消费（元）——key 行展示用（列表页走 key_spends 批量）。"""
        with self._lock:
            r = self._conn.execute(
                "SELECT COALESCE(SUM(cost), 0) AS s FROM requests WHERE dkey_id = ?",
                (int(key_id),)).fetchone()
        return round(float(r["s"] or 0), 6)

    def update_key_models(self, user_id: int, key_id: int,
                          model_bindings: list | None) -> bool:
        """改 key 的模型绑定（None/空列表 = 清空 → 不限制）。"""
        with self._lock:
            cur = self._conn.execute(
                "UPDATE api_keys SET model_bindings = ? WHERE id = ? AND user_id = ?",
                (self._models_to_json(model_bindings), int(key_id), int(user_id)))
            self._conn.commit()
            return cur.rowcount > 0

    def get_api_key(self, key_id: int) -> dict | None:
        """取一条 api_keys（BLOB 双列剥掉：bytes 进不了 JSON 序列化，对外永远只给明文 key 列）。"""
        with self._lock:
            r = self._conn.execute("SELECT * FROM api_keys WHERE id = ?", (int(key_id),)).fetchone()
        return self._clean_key_row(r) if r else None

    def list_api_keys(self, user_id: int | None) -> list[dict]:
        """user_id=None 时列全部账号的 key（管理员汇总视图用；行里自带 user_id 归属）。"""
        with self._lock:
            if user_id is None:
                rows = self._conn.execute(
                    "SELECT * FROM api_keys ORDER BY user_id, id").fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM api_keys WHERE user_id = ? ORDER BY id", (int(user_id),)).fetchall()
        return [self._clean_key_row(r) for r in rows]

    @staticmethod
    def _clean_key_row(r) -> dict:
        d = dict(r)
        d.pop("key_mac", None)    # 32B HMAC：服务端内部用，不外泄
        d.pop("key_enc", None)    # AES-GCM 密文：服务端内部用，不外泄
        d["models"] = Store._models_from_json(d.pop("model_bindings", None))  # 解析后外泄：list 或 None（不限制）
        return d

    def revoke_api_key(self, user_id: int, key_id: int) -> bool:
        with self._lock:
            cur = self._conn.execute("UPDATE api_keys SET revoked = 1 WHERE id = ? AND user_id = ?",
                                     (int(key_id), int(user_id)))
            self._conn.commit()
            return cur.rowcount > 0

    def delete_api_key(self, user_id: int, key_id: int) -> bool:
        """彻底删掉一把 key（不是只打吊销标记）。"""
        with self._lock:
            cur = self._conn.execute("DELETE FROM api_keys WHERE id = ? AND user_id = ?",
                                     (int(key_id), int(user_id)))
            self._conn.commit()
            return cur.rowcount > 0

    def count_active_keys(self, user_id: int) -> int:
        with self._lock:
            return self._conn.execute(
                "SELECT COUNT(*) FROM api_keys WHERE user_id = ? AND revoked = 0",
                (int(user_id),)).fetchone()[0]

    def user_for_api_key(self, key: str) -> dict | None:
        """下游请求带 key 进来 → 反查账号（顺便摸一下 last_used）。

        返回的 dict 上附带：
        _key_id（api_keys.id，供记账 dkey_id 维度）、
        _key_revoked（该 key 已被禁用/吊销 → 调用方拦 401「已禁用」）、
        _key_expired（过期时间已到 → 调用方拦 401「已过期」）、
        _key_expires_at（原始过期时间 ISO 串，报错信息里回显用；无过期 = ""）、
        _key_models（模型绑定白名单，list；None = 不限制 → 调用方做路由门禁）、
        _key_balance（余额（元），None = 不限额只统计；< 0 = 扣成负数 → 调用方入口拦 403）。
        """
        key = (key or "").strip()
        if not key:
            return None
        r = None
        with self._lock:
            # 主路径：MAC 索引一击命中（O(1)；加密双列是 v1.1.0 起默认路径）
            if self._key_cipher is not None:
                r = self._conn.execute(
                    "SELECT u.*, k.id AS key_id, k.expires_at AS key_expires, k.revoked AS key_revoked,"
                    " k.model_bindings AS key_models, k.balance AS key_balance"
                    " FROM api_keys k JOIN users u ON u.id = k.user_id"
                    " WHERE k.key_mac = ? AND u.status = 'active'",
                    (self._key_cipher.mac(key),)).fetchone()
            # 兜底：历史未迁移行（key_mac NULL）仍按明文列验签，迁移完成后此分支自然无人命中
            if r is None:
                r = self._conn.execute(
                    "SELECT u.*, k.id AS key_id, k.expires_at AS key_expires, k.revoked AS key_revoked,"
                    " k.model_bindings AS key_models, k.balance AS key_balance"
                    " FROM api_keys k JOIN users u ON u.id = k.user_id"
                    " WHERE k.key = ? AND u.status = 'active'", (key,)).fetchone()
            if r:
                self._conn.execute("UPDATE api_keys SET last_used = ? WHERE id = ?",
                                    (self._now(), r["key_id"]))
                self._conn.commit()
        if not r:
            return None
        user = dict(r)
        user["_key_id"] = user.pop("key_id")
        exp = user.pop("key_expires") or ""
        user["_key_expires_at"] = exp
        user["_key_revoked"] = bool(user.pop("key_revoked"))
        user["_key_expired"] = (not user["_key_revoked"]) and bool(exp) and self._now() >= exp
        user["_key_models"] = self._models_from_json(user.pop("key_models", None))
        # v2.1.0 R154：key 余额（None = 不限额只统计消费；扣成负数后调用方入口预检拦截 403）
        user["_key_balance"] = user.pop("key_balance", None)
        return user

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass
