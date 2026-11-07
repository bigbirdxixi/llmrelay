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


class TenantRegistries:
    def __init__(self, root: Path, legacy_file: Path | None = None, store=None):
        self.root = Path(root)
        self.legacy_file = Path(legacy_file) if legacy_file else None
        self.store = store                      # 用来判断「谁是第一个管理员」
        self.root.mkdir(parents=True, exist_ok=True)
        self._cache: dict[int, Registry] = {}
        self._legacy_claimed = False
        # 认领标记要落盘：重启后不能忘，否则下一个新账号又会捞一份管理员的配置
        self._marker = self.root / ".legacy_claimed"

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
                reg = Registry(self.legacy_file, self._clients_file())
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
            else:
                file.write_text("[]", encoding="utf-8")
        reg = Registry(file, self._clients_file())
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
