"""离线 IP2Region xdb IP 定位（纯 Python 二进制查询，无第三方依赖）。

- 数据库文件：环境变量 RELAY_IP2REGION_XDB，默认 <DATA_DIR>/IP2Region.xdb
  （IP2Region v3.x 版 ip2region_v4.xdb，约 11MB，可从 https://github.com/lionsoul2014/ip2region
  的 data/ip2region_v4.xdb 下载；国内镜像 gitee.com/lionsoul/ip2region）。
- 文件缺失时功能自动关闭（静默降级），调用方先走本库、未命中再落 GeoLite2。
- 返回 details 字符串，字段以 | 分隔：国家|省份|城市|ISP|ISO（如 中国|四川省|成都市|电信|CN）；
  内网/保留段为 0|0|内网IP|内网IP|0；未命中 / IPv6 / 文件缺失返回 ""。

xdb 结构（v3.x，见 ip2region.net/doc/xdb/structure）：
  Header 256B（version u16 / cache u16 / genTime u32 / indexStart u32 / indexEnd u32 / ipVer u16 / ptrByte u16）
  Vector 索引 512KiB（256×256 单元格 × 8B：区域索引起/止指针，file offset）
  地域信息段（utf-8 details，去重存储）
  二分索引段（IPv4 每项 14B：startIP u32LE + endIP u32LE + dataLen u16 + dataPtr u32）
"""
from __future__ import annotations

import ipaddress
import os
from pathlib import Path

from app.config import DATA_DIR

_HEADER = 256
_VEC_SIZE = 256 * 256 * 8        # 512 KiB
_ITEM_V4 = 14                   # IPv4 二分索引项字节数

_buf: bytes | None = None
_loaded_for: str | None = None
_load_failed = False
_avail_checked = False


def xdb_path() -> Path:
    raw = (os.getenv("RELAY_IP2REGION_XDB") or "").strip()
    if raw:
        return Path(raw).expanduser()
    return DATA_DIR / "IP2Region.xdb"


def _load() -> bytes | None:
    """整文件读入内存（约 11MB，一次性 ~50ms）；缺失/损坏返回 None。"""
    global _buf, _loaded_for, _load_failed
    p = xdb_path()
    key = str(p)
    if _buf is not None and _loaded_for == key and p.exists():
        return _buf
    if _load_failed and _loaded_for == key:
        return None
    if not p.exists():
        return None
    try:
        b = p.read_bytes()
        if len(b) < _HEADER + _VEC_SIZE:
            return None
        # 校验头：version=3，ipVersion=4（IPv4 xdb），否则视为格式不符，禁用
        ver = b[0] | (b[1] << 8)
        ipver = b[16] | (b[17] << 8)
        if ver != 3 or ipver != 4:
            return None
        _buf = b
        _loaded_for = key
        return b
    except Exception:
        _load_failed = True
        _buf = None
        return None


def available() -> bool:
    return _load() is not None


def reset() -> None:
    """测试用：清缓存（换文件后重新加载）。"""
    global _buf, _loaded_for, _load_failed
    _buf = None
    _loaded_for = None
    _load_failed = False


def query(ip: str) -> str:
    """IPv4 查询 → details（国家|省份|城市|ISP|ISO）；未命中/IPv6/无库 → ""。"""
    b = _load()
    if b is None:
        return ""
    s = (ip or "").strip()
    if not s:
        return ""
    try:
        a = ipaddress.ip_address(s)
    except ValueError:
        return ""
    if a.version != 4:
        return ""
    ip_int = int(a)
    # 1) vector 索引：IP 前 2 字节定位分区
    i = ((ip_int >> 24) & 0xFF) * 256 * 8 + ((ip_int >> 16) & 0xFF) * 8
    vec = b[_HEADER:_HEADER + _VEC_SIZE]
    sPtr = int.from_bytes(vec[i:i + 4], "little")
    ePtr = int.from_bytes(vec[i + 4:i + 8], "little")
    if sPtr >= ePtr:
        return ""
    # 2) 二分索引拆半查找（14B 项：startIP/endIP 小端 u32，dataLen u16，dataPtr u32）
    l, h = 0, (ePtr - sPtr) // _ITEM_V4
    found = -1
    while l < h:
        m = (l + h) >> 1
        p = sPtr + m * _ITEM_V4
        start_i = int.from_bytes(b[p:p + 4], "little")
        end_i = int.from_bytes(b[p + 4:p + 8], "little")
        if ip_int < start_i:
            h = m
        elif ip_int > end_i:
            l = m + 1
        else:
            found = p
            break
    if found < 0:
        return ""
    # 3) 地域信息
    data_len = int.from_bytes(b[found + 8:found + 10], "little")
    data_ptr = int.from_bytes(b[found + 10:found + 14], "little")
    if data_len <= 0 or data_ptr + data_len > len(b):
        return ""
    return b[data_ptr:data_ptr + data_len].decode("utf-8", "replace")
