"""密钥加解密（v1.1.0）：下游 API Key 不再明文落库。

密钥层级（定稿设计，见 CHANGELOG [Unreleased]）：
- S（主密钥 master secret）：32 随机字节。来源优先级
  ① 环境变量 RELAY_KEY_SECRET（hex 或 base64，运维自己生成的）
  ② config/master.key（首次启动自动生成并落盘 0600，大声警告）
  ③ 库里已有加密 key 而两者都找不到 → 拒绝启动（宁可停服也不悄悄换一把钥匙）。
  S 是「DB/备份泄露 ≠ key 泄露」的最后一道墙：S 永不进数据备份、永不进发布包。

- api_keys 表（下游 key）用 S 派生的全局工作密钥：
    key_mac = HMAC-SHA256(K_mac, key)  → 确定性，建唯一索引，验签专用（每请求 O(1)）
    key_enc = AES-GCM(K_enc, key)      → 可逆（随机 12B nonce），复制按钮专用
  为什么是全局而不是「每用户一把 K_t」：验签时只知道 key 不知道属于谁，
  全局 MAC 密钥才能做到索引一击命中；每账号 K_t 会退化成 O(n) 逐个算 MAC
  （设计红线：禁止全量解密比对）。「每租户一把随机密钥」的诉求在下一阶段
  用于加密各账号的上游配置（config/tenants/u<id>/upstreams.json）——那时
  解密时 user_id 已知，不存在热路径索引问题。
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import logging
import os
import secrets
from pathlib import Path

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from . import config

log = logging.getLogger("relay")

ENV_SECRET = "RELAY_KEY_SECRET"


def _derive(secret: bytes, info: bytes) -> bytes:
    """HKDF-SHA256：从同一把 S 派生用途不同的 32B 子密钥（MAC / GCM 分域）。"""
    return HKDF(algorithm=hashes.SHA256(), length=32, salt=None,
                info=info).derive(secret)


class KeyCipher:
    """下游 API Key 的 MAC + 可逆封装（AES-GCM）。线程安全（无内部状态）。"""

    def __init__(self, master: bytes):
        master = bytes(master)
        if len(master) < 16:
            raise ValueError("主密钥太短（至少 16 字节）")
        self._mac_key = _derive(master, b"llm-relay:api-keys:mac:v1")
        self._gcm = AESGCM(_derive(master, b"llm-relay:api-keys:enc:v1"))

    # ---- 验签（确定性，走索引） ----
    def mac(self, key: str) -> bytes:
        return hmac.new(self._mac_key, (key or "").encode("utf-8"), hashlib.sha256).digest()

    # ---- 存储/取回（可逆） ----
    def seal(self, key: str) -> bytes:
        """明文 → 12B 随机 nonce + AES-GCM 密文（含 16B tag）。"""
        nonce = os.urandom(12)
        return nonce + self._gcm.encrypt(nonce, (key or "").encode("utf-8"), None)

    def open(self, blob: bytes) -> str:
        """seal 的逆。失败抛 cryptography.exceptions.InvalidTag。"""
        nonce, ct = bytes(blob[:12]), bytes(blob[12:])
        return self._gcm.decrypt(nonce, ct, None).decode("utf-8")


def _parse_env_secret(raw: str) -> bytes:
    raw = raw.strip()
    try:
        b = bytes.fromhex(raw)
        if len(b) >= 16:
            return b
    except (ValueError, binascii.Error):
        pass
    try:
        b = base64.b64decode(raw, validate=True)
        if len(b) >= 16:
            return b
    except Exception:
        pass
    return raw.encode("utf-8")   # 原样当字节用（>=16 才放行，KeyCipher 兜底校验）


def resolve_master_secret(store) -> bytes:
    """启动期解析主密钥 S。store 需有 has_encrypted_keys()。

    缺 S 且库里已有加密 key → raise RuntimeError（拒绝启动，fail-fast）。
    缺 S 且库是纯明文/空 → 自动生成并持久化到 config/master.key（0600）。
    """
    env = (os.getenv(ENV_SECRET) or "").strip()
    if env:
        return _parse_env_secret(env)

    f: Path = config.MASTER_KEY_FILE
    if f.exists():
        got = f.read_bytes().strip()
        if len(got) >= 16:
            return got
        log.warning("主密钥文件 %s 损坏（<16 字节），按缺失处理", f)

    if store.has_encrypted_keys():
        raise RuntimeError(
            "找不到主密钥（RELAY_KEY_SECRET 环境变量或 config/master.key），"
            "但数据库里已有加密存储的 API Key——换一把新钥匙解不开旧密文。"
            "请补回原主密钥后重启。")

    got = secrets.token_bytes(32)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(got)
    try:
        f.chmod(0o600)
    except Exception:
        pass
    log.warning("已自动生成主密钥并保存到 %s（0600）。请单独妥善保存它："
                "丢失 = 所有已加密的 API Key 无法恢复；它不进数据备份、不进发布包。", f)
    return got
