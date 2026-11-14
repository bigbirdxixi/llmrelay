"""每个账号一份独立的上游配置（平台 / key / 模型 / 思考设置）。

- 存储位置：config/tenants/u<id>/upstreams.json
- 第一次取某个账号时，如果文件不存在：老的单体 config/upstreams.json 会被复制给第一个
  取它的账号（多账号改造前的配置不丢），之后的账号从空配置开始。
- 保留原来的 Registry（文件 + 热加载 + 多 key 轮换）不动，只是变成「一账号一实例」。
"""
from __future__ import annotations

import asyncio
import contextlib
import shutil
from pathlib import Path

from .registry import Registry
from .crypto import tenant_cipher

import json
import logging

log = logging.getLogger("relay")


class TenantRegistries:
    def __init__(self, root: Path, legacy_file: Path | None = None, store=None):
        self.root = Path(root)
        self.legacy_file = Path(legacy_file) if legacy_file else None
        self.store = store                      # 用来判断「谁是第一个管理员」
        self.root.mkdir(parents=True, exist_ok=True)
        self._cache: dict[int, Registry] = {}
        self._legacy_claimed = False
        self._secret: bytes | None = None       # v1.1.1：主密钥 S（lifespan 注入后各租户派生 K_t）
        # 认领标记要落盘：重启后不能忘，否则下一个新账号又会捞一份管理员的配置
        self._marker = self.root / ".legacy_claimed"

    # ---------- v1.1.1：主密钥 ----------
    def set_secret(self, secret: bytes) -> None:
        """lifespan 里注入 S。之前按「无 cipher」建好的缓存实例作废重建。"""
        self._secret = bytes(secret)
        self._cache.clear()

    def _cipher_for(self, uid: int):
        return tenant_cipher(self._secret, uid) if self._secret is not None else None

    @property
    def claimed(self) -> bool:
        """老 config 是否已认领给第一个管理员（标记落盘，重启不丢）。"""
        return self._legacy_claimed or self._marker.exists()

    def _may_seed(self, uid: int) -> bool:
        """只有「第一个管理员」能继承老的 config/upstreams.json，且一辈子只发生一次。"""
        if self._legacy_claimed or self._marker.exists():
            return False
        if not (self.legacy_file and self.legacy_file.exists()):
            return False
        if self.store is not None:
            admin = self.store.first_admin()
            if not admin or int(admin["id"]) != int(uid):
                return False
        return True

    # ---------- 路径 ----------
    def path(self, user_id: int) -> Path:
        return self.root / f"u{int(user_id)}" / "upstreams.json"

    # ---------- 取实例 ----------
    def get(self, user_id: int) -> Registry:
        uid = int(user_id)
        if uid == 0 and self.legacy_file is not None:
            # 还没有任何账号时的保底租户：直接读写老的单体配置，避免客户端断线
            reg = self._cache.get(0)
            if reg is None:
                reg = Registry(self.legacy_file, self._clients_file(),
                                cipher=self._cipher_for(0))
                self._cache[0] = reg
            return reg
        reg = self._cache.get(uid)
        if reg is not None:
            return reg
        file = self.path(uid)
        file.parent.mkdir(parents=True, exist_ok=True)
        if not file.exists():
            if self._may_seed(uid):
                # 老配置整体归第一个管理员；标记落盘，之后任何新账号都不会再拿到
                shutil.copy2(self.legacy_file, file)
                self._legacy_claimed = True
                with contextlib.suppress(Exception):
                    self._marker.write_text(f"claimed_by={uid}", encoding="utf-8")
                if self._secret is not None:
                    # 刚拷进来的文件可能是明文，也可能是老文件已按 K_0 加密——
                    # 一律归一化成本租户 K_uid（明文直接封，K_0 密文先解后封）
                    migrate_one_tenant_file(file, uid, self._secret, rekey_from=0)
            else:
                file.write_text("[]", encoding="utf-8")
        reg = Registry(file, self._clients_file(), cipher=self._cipher_for(uid))
        self._cache[uid] = reg
        return reg

    def _clients_file(self) -> Path:
        # 老的 clients.json 现在已经不参与鉴权了，但 Registry 构造需要它
        return self.root / "_clients.json"

    # ---------- 维护 ----------
    def reload(self) -> None:
        """测试与手工重载用：把所有已加载的账号配置重读一遍。"""
        for reg in list(self._cache.values()):
            reg.reload()

    def drop(self, user_id: int) -> None:
        self._cache.pop(int(user_id), None)

    async def watch(self, interval: float = 5.0) -> None:
        while True:
            await asyncio.sleep(interval)
            for reg in list(self._cache.values()):
                try:
                    if reg.file_changed():
                        reg.reload()
                except Exception:
                    continue


# ---------- v1.1.1：租户配置（upstreams.json）上游 key 加密迁移 ----------
def _row_plain_keys(row: dict) -> list[str]:
    """取一行里的明文 key（api_key 便捷写法 + api_keys 多 key），去重去空。"""
    keys: list[str] = []
    single = row.get("api_key")
    if isinstance(single, str) and single.strip():
        keys.append(single.strip())
    for k in (row.get("api_keys") or []):
        if isinstance(k, str) and k.strip() and k.strip() not in keys:
            keys.append(k.strip())
    return keys


def _load_rows(path: Path) -> list:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    data = json.loads(text)
    return data if isinstance(data, list) else []


def migrate_one_tenant_file(path: Path, uid: int, secret: bytes, rekey_from: int | None = None) -> bool:
    """把单个租户 upstreams.json 的 key 归一化为「本域 K_uid 加密、磁盘无明文」（幂等、原子、round-trip 校验）。

    - 明文行（api_key/api_keys 非空）→ 用 K_uid 封成 enc_keys
    - rekey_from 给定时：其他域（K_from）封的密文行 → 解出再用 K_uid 重封（legacy 认领拷贝用）
    - 已是 K_uid 密文且无明文 → 原样保留（不重封 nonce，幂等）
    - 写临时文件 → 重新解析 → 用 K_uid 解回逐项比对 → 全对才 replace；失败保留原文件
    返回是否改了文件。
    """
    try:
        rows = _load_rows(path)
    except Exception:
        return False
    cipher = tenant_cipher(secret, uid)
    src = tenant_cipher(secret, rekey_from) if rekey_from is not None else None
    changed = False
    expected: list[tuple[int, list[str]]] = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict) or row.get("_kind"):
            continue
        plain = _row_plain_keys(row)
        enc = row.get("enc_keys") or []
        if plain:
            row["enc_keys"] = [cipher.seal_str(k) for k in plain]
            row["api_key"] = ""
            row["api_keys"] = []
            expected.append((i, plain))
            changed = True
        elif enc and src is not None:
            # 其他域的密文：解出来再用本域重封（K_from → K_uid）
            try:
                dec = [src.open_str(s) for s in enc if isinstance(s, str)]
            except Exception:
                log.warning("租户文件 %s 平台 %s 的密文不是 K_%s 封的，跳过该行",
                            path.name, row.get("platform"), rekey_from)
                continue
            row["enc_keys"] = [cipher.seal_str(k) for k in dec]
            expected.append((i, dec))
            changed = True
    if not changed:
        return False
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(rows, ensure_ascii=False, indent=2), "utf-8")
    # round-trip：重读 + 解回，逐项比对，全对才生效
    try:
        check = _load_rows(tmp)
        for i, plain in expected:
            r = check[i]
            got = [cipher.open_str(s) for s in r.get("enc_keys", [])]
            if got != plain:
                raise ValueError("round-trip 不一致")
    except Exception:
        with contextlib.suppress(OSError):
            tmp.unlink()
        log.error("租户 %s 配置加密迁移失败（%s），保留原文件", uid, path.name)
        return False
    tmp.replace(path)
    log.info("租户 %s 的上游 key 已加密迁移（%s，%d 个平台）", uid, path.name, len(expected))
    return True


def migrate_tenant_files(root: Path, secret: bytes) -> int:
    """启动期一次性迁移：扫描 root/u<uid>/upstreams.json，幂等。返回改动的文件数。"""
    n = 0
    for d in sorted(Path(root).glob("u*")):
        f = d / "upstreams.json"
        if d.is_dir() and f.exists():
            try:
                uid = int(d.name[1:])
            except ValueError:
                continue
            if migrate_one_tenant_file(f, uid, secret):
                n += 1
    return n


def has_encrypted_tenant_files(root: Path, legacy_file: Path | None = None) -> bool:
    """fail-fast 扫描：任何租户配置里已有 enc_keys（加密材料）？"""
    def any_enc(path: Path) -> bool:
        try:
            return any(isinstance(r, dict) and r.get("enc_keys") for r in _load_rows(path))
        except Exception:
            return False
    files = [d / "upstreams.json" for d in sorted(Path(root).glob("u*")) if d.is_dir()]
    if legacy_file and Path(legacy_file).exists():
        files.append(Path(legacy_file))
    return any(any_enc(f) for f in files if f.exists())
