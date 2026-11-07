"""离线 GeoIP 地理编码（GeoLite2-City.mmdb + IP2Region xdb 双库）。

客户端 IP → 地区（中国到省、境外到国家），供概览页「地域访问地图」聚合。
- 中国段优先 IP2Region xdb（离线、城市级、免费、周更）：
  文件路径环境变量 RELAY_IP2REGION_XDB，默认 <DATA_DIR>/IP2Region.xdb；
  来源：github.com/lionsoul2014/ip2region 的 data/ip2region_v4.xdb（约 11MB），
  国内可用 gitee.com/lionsoul/ip2region（大文件 raw 需登录，走 GitHub 镜像下载）。
- 境外与 xdb 未命中落 GeoLite2-City.mmdb：环境变量 RELAY_GEOIP_DB，
  默认 <DATA_DIR>/GeoLite2-City.mmdb，需从 MaxMind 免费账号下载（每月可更新），
  见 DEPLOY.md。
- 读时解析（查询时内存查，微秒级）；两个文件都缺失时功能自动关闭，
  端点返回 db="missing"，前端显示配置提示，不影响其他页面。
"""
from __future__ import annotations

import ipaddress
import os
from pathlib import Path

from app.config import DATA_DIR

_reader = None          # maxminddb.Reader 或 None
_loaded_for: str | None = None
_load_failed = False

# 中国省级行政区划：GeoLite2 的 subdivision.iso_code（FIPS 10-4）→ 中文名
# 注：部分 GeoLite2 版本的 subdivision.iso_code 不是 FIPS 数字码（如四川返回 "SC"），
# 所以 resolve() 还兜底按 subdivision 英文名映射（CN_PROVINCES_EN），两者缺一都不丢。
CN_PROVINCES = {
    "11": "北京", "12": "天津", "13": "河北", "14": "山西", "15": "内蒙古",
    "21": "辽宁", "22": "吉林", "23": "黑龙江",
    "31": "上海", "32": "江苏", "33": "浙江", "34": "安徽", "35": "福建",
    "36": "江西", "37": "山东",
    "41": "河南", "42": "湖北", "43": "湖南", "44": "广东", "45": "广西",
    "46": "海南",
    "50": "重庆", "51": "四川", "52": "贵州", "53": "云南", "54": "西藏",
    "61": "陕西", "62": "甘肃", "63": "青海", "64": "宁夏", "65": "新疆",
    "71": "台湾", "81": "香港", "82": "澳门",
}

# GeoLite2 subdivision 英文名 → 中文省名（主用键：不同库版本 iso_code 形态不一致，
# 英文名稳定；与前端 CHINA_PROVINCE_PATHS 的键严格一致）
CN_PROVINCES_EN = {
    "Beijing": "北京", "Tianjin": "天津", "Hebei": "河北", "Shanxi": "山西",
    "Inner Mongolia": "内蒙古", "Liaoning": "辽宁", "Jilin": "吉林",
    "Heilongjiang": "黑龙江",
    "Shanghai": "上海", "Jiangsu": "江苏", "Zhejiang": "浙江", "Anhui": "安徽",
    "Fujian": "福建", "Jiangxi": "江西", "Shandong": "山东",
    "Henan": "河南", "Hubei": "湖北", "Hunan": "湖南", "Guangdong": "广东",
    "Guangxi": "广西", "Hainan": "海南",
    "Chongqing": "重庆", "Sichuan": "四川", "Guizhou": "贵州",
    "Yunnan": "云南", "Tibet": "西藏",
    "Shaanxi": "陕西", "Gansu": "甘肃", "Qinghai": "青海",
    "Ningxia": "宁夏", "Xinjiang": "新疆",
    "Taiwan": "台湾", "Hong Kong": "香港", "Macao": "澳门", "Macau": "澳门",
}

# 中国 IP 查不到 subdivision 时的兜底（移动/运营商/IPv6 段常见）
CN_FALLBACK = "中国大陆"

# GeoLite2 city 英文名 → 中文城市名（城市热点用；未收录的城市显示英文名）
# 覆盖 31 个省级行政区首府/主要城市 + 直辖市 + 常用城市 + 主要境外城市
CITY_ZH = {
    # 直辖市 / 特别行政区
    "Beijing": "北京", "Shanghai": "上海", "Tianjin": "天津", "Chongqing": "重庆",
    "Hong Kong": "香港", "Macao": "澳门", "Macau": "澳门",
    # 河北/山西/内蒙古
    "Shijiazhuang": "石家庄", "Tangshan": "唐山", "Qinhuangdao": "秦皇岛",
    "Taiyuan": "太原", "Hohhot": "呼和浩特", "Baotou": "包头",
    # 东北
    "Shenyang": "沈阳", "Dalian": "大连", "Changchun": "长春", "Harbin": "哈尔滨",
    # 江苏/浙江/安徽
    "Nanjing": "南京", "Suzhou": "苏州", "Wuxi": "无锡", "Changzhou": "常州",
    "Nantong": "南通", "Xuzhou": "徐州", "Yangzhou": "扬州",
    "Hangzhou": "杭州", "Ningbo": "宁波", "Wenzhou": "温州", "Jiaxing": "嘉兴",
    "Shaoxing": "绍兴", "Jinhua": "金华", "Hefei": "合肥",
    # 福建/江西/山东
    "Fuzhou": "福州", "Xiamen": "厦门", "Quanzhou": "泉州",
    "Nanchang": "南昌", "Jinan": "济南", "Qingdao": "青岛", "Yantai": "烟台",
    "Weifang": "潍坊",
    # 河南/湖北/湖南
    "Zhengzhou": "郑州", "Wuhan": "武汉", "Changsha": "长沙", "Zhuzhou": "株洲",
    "Xiangtan": "湘潭",
    # 广东/广西/海南
    "Guangzhou": "广州", "Shenzhen": "深圳", "Zhuhai": "珠海", "Dongguan": "东莞",
    "Foshan": "佛山", "Zhongshan": "中山", "Huizhou": "惠州", "Shanwei": "汕尾",
    "Zhanjiang": "湛江", "Nanning": "南宁", "Haikou": "海口", "Sanya": "三亚",
    # 西南
    "Chengdu": "成都", "Mianyang": "绵阳", "Guiyang": "贵阳", "Kunming": "昆明",
    "Dali": "大理", "Lhasa": "拉萨",
    # 西北
    "Xi'an": "西安", "Xian": "西安", "Lanzhou": "兰州", "Xining": "西宁",
    "Yinchuan": "银川", "Urumqi": "乌鲁木齐", "Ürümqi": "乌鲁木齐",
    # 主要境外城市
    "Tokyo": "东京", "Osaka": "大阪", "Seoul": "首尔", "Busan": "釜山",
    "Singapore": "新加坡", "Bangkok": "曼谷", "Kuala Lumpur": "吉隆坡",
    "Jakarta": "雅加达", "Manila": "马尼拉", "Delhi": "新德里",
    "Mumbai": "孟买", "New Delhi": "新德里", "Taipei": "台北",
    "Sydney": "悉尼", "Melbourne": "墨尔本", "Auckland": "奥克兰",
    "Toronto": "多伦多", "Vancouver": "温哥华", "Montreal": "蒙特利尔",
    "New York": "纽约", "Los Angeles": "洛杉矶", "San Francisco": "旧金山",
    "Chicago": "芝加哥", "Seattle": "西雅图", "Houston": "休斯顿", "Miami": "迈阿密",
    "London": "伦敦", "Paris": "巴黎", "Berlin": "柏林", "Madrid": "马德里",
    "Rome": "罗马", "Amsterdam": "阿姆斯特丹", "Brussels": "布鲁塞尔",
    "Dublin": "都柏林", "Frankfurt": "法兰克福", "Zurich": "苏黎世",
    "Vienna": "维也纳", "Stockholm": "斯德哥尔摩", "Oslo": "奥斯陆",
    "Helsinki": "赫尔辛基", "Copenhagen": "哥本哈根", "Munich": "慕尼黑",
    "Istanbul": "伊斯坦布尔", "Tel Aviv": "特拉维夫", "Doha": "多哈",
    "Dubai": "迪拜", "Abu Dhabi": "阿布扎比", "Cairo": "开罗",
    "Johannesburg": "约翰内斯堡", "São Paulo": "圣保罗", "Sao Paulo": "圣保罗",
    "Buenos Aires": "布宜诺斯艾利斯", "Santiago": "圣地亚哥",
    "Moscow": "莫斯科", "St. Petersburg": "圣彼得堡", "Almaty": "阿拉木图",
    "Tehran": "德黑兰", "Karachi": "卡拉奇", "Islamabad": "伊斯兰堡",
    "Dhaka": "达卡", "Colombo": "科伦坡", "Kathmandu": "加德满都",
    "Phnom Penh": "金边", "Vientiane": "万象", "Hanoi": "河内",
    "Ho Chi Minh City": "胡志明市",
}


def city_info(rec: dict) -> dict:
    """GeoLite2 record → {city, city_zh, lat, lon}；无城市数据返回空 dict。

    city_zh：CITY_ZH 收录的转中文，未收录的原样（拼音/英文名）。
    lat/lon 仅在 city 存在时有效（国家级兜底坐标无定位意义）。
    """
    city = ((rec.get("city") or {}).get("names") or {}).get("en") or ""
    if not city:
        return {}
    loc = rec.get("location") or {}
    lat, lon = loc.get("latitude"), loc.get("longitude")
    return {"city": city, "city_zh": CITY_ZH.get(city, city), "lat": lat, "lon": lon}


# 常见国家 ISO → 中文名（其余回退英文名）
COUNTRY_ZH = {
    "US": "美国", "JP": "日本", "KR": "韩国", "SG": "新加坡", "HK": "香港",
    "MO": "澳门", "TW": "台湾", "GB": "英国", "DE": "德国",
    "FR": "法国", "IT": "意大利", "ES": "西班牙", "NL": "荷兰", "RU": "俄罗斯",
    "IN": "印度", "VN": "越南", "TH": "泰国", "MY": "马来西亚", "ID": "印度尼西亚",
    "PH": "菲律宾", "AU": "澳大利亚", "NZ": "新西兰", "CA": "加拿大",
    "BR": "巴西", "MX": "墨西哥", "AE": "阿联酋", "IL": "以色列", "ZA": "南非",
    "CH": "瑞士", "SE": "瑞典", "NO": "挪威", "DK": "丹麦", "FI": "芬兰",
    "PL": "波兰", "UA": "乌克兰", "TR": "土耳其", "SA": "沙特阿拉伯",
    "EG": "埃及", "AR": "阿根廷", "CL": "智利", "CO": "哥伦比亚",
    "PE": "秘鲁", "KE": "肯尼亚", "NG": "尼日利亚", "MA": "摩洛哥",
    "AT": "奥地利", "BE": "比利时", "IE": "爱尔兰", "PT": "葡萄牙",
    "GR": "希腊", "CZ": "捷克", "RO": "罗马尼亚", "HU": "匈牙利",
    "KH": "柬埔寨", "LA": "老挝", "MM": "缅甸",
    "BN": "文莱", "MN": "蒙古", "KZ": "哈萨克斯坦",
    "UZ": "乌兹别克斯坦", "BD": "孟加拉国", "PK": "巴基斯坦", "LK": "斯里兰卡",
    "NP": "尼泊尔", "AF": "阿富汗", "IR": "伊朗",
}


# IP2Region xdb 中国城市 → 经纬度（地图城市热点橙点用；未收录的城市落到省会坐标）
CN_CITY_COORDS: dict[str, tuple[float, float]] = {
    # 直辖市 / 特别行政区 / 省会
    "北京": (39.90, 116.40), "天津": (39.10, 117.20), "上海": (31.20, 121.50),
    "重庆": (29.60, 106.50), "香港": (22.30, 114.20), "澳门": (22.20, 113.50),
    "台北": (25.00, 121.50),
    "石家庄": (38.30, 114.50), "太原": (37.90, 112.50), "呼和浩特": (40.80, 111.70),
    "沈阳": (41.80, 123.40), "长春": (43.90, 125.30), "哈尔滨": (45.80, 126.50),
    "南京": (32.10, 118.80), "杭州": (30.30, 120.20), "合肥": (31.80, 117.20),
    "福州": (26.10, 119.30), "厦门": (24.50, 118.10), "南昌": (28.70, 115.90),
    "济南": (36.70, 117.00), "青岛": (36.10, 120.40), "郑州": (34.70, 113.70),
    "武汉": (30.60, 114.30), "长沙": (28.20, 112.90), "广州": (23.10, 113.30),
    "深圳": (22.50, 114.10), "南宁": (22.80, 108.40), "海口": (20.00, 110.30),
    "成都": (30.60, 104.10), "贵阳": (26.70, 106.70), "昆明": (25.00, 102.70),
    "拉萨": (29.70, 91.10), "西安": (34.30, 108.90), "兰州": (36.10, 103.80),
    "西宁": (36.60, 101.80), "银川": (38.50, 106.20), "乌鲁木齐": (43.80, 87.60),
    # 主要城市
    "苏州": (31.30, 120.60), "无锡": (31.50, 120.30), "常州": (31.80, 119.90),
    "南通": (32.00, 120.90), "扬州": (32.40, 119.40), "徐州": (34.30, 117.20),
    "金华": (29.10, 119.60), "宁波": (29.90, 121.60), "温州": (28.00, 120.70),
    "嘉兴": (30.80, 120.80), "绍兴": (30.00, 120.60), "东莞": (23.00, 113.70),
    "佛山": (23.00, 113.10), "珠海": (22.30, 113.60), "中山": (22.50, 113.40),
    "惠州": (23.10, 114.40), "汕头": (23.40, 116.70), "湛江": (21.20, 110.40),
    "泉州": (24.90, 118.60), "赣州": (25.90, 115.40), "烟台": (37.50, 121.40),
    "潍坊": (36.70, 119.10), "绵阳": (31.50, 104.70), "德阳": (31.10, 104.40),
    "遵义": (27.70, 106.90), "大理": (25.60, 100.20), "丽江": (26.90, 100.30),
    "大理市": (25.60, 100.20), "咸阳": (34.30, 108.70), "宝鸡": (34.40, 107.20),
    "天水": (34.60, 105.70), "包头": (40.70, 107.20), "鄂尔多斯": (39.60, 109.80),
    "赤峰": (43.90, 118.90), "大连": (38.90, 121.60), "秦皇岛": (39.90, 119.60),
    "柳州": (24.30, 109.40), "桂林": (25.30, 110.30), "三亚": (18.30, 109.50),
}

# 省份（归一后名）→ 省会坐标（城市未收录时兜底，保证橙点落在省轮廓内）
CN_PROV_COORDS: dict[str, tuple[float, float]] = {
    "北京": (39.90, 116.40), "天津": (39.10, 117.20), "河北": (38.30, 114.50),
    "山西": (37.90, 112.50), "内蒙古": (40.80, 111.70), "辽宁": (41.80, 123.40),
    "吉林": (43.90, 125.30), "黑龙江": (45.80, 126.50), "上海": (31.20, 121.50),
    "江苏": (32.10, 118.80), "浙江": (30.30, 120.20), "安徽": (31.80, 117.20),
    "福建": (26.10, 119.30), "江西": (28.70, 115.90), "山东": (36.70, 117.00),
    "河南": (34.70, 113.70), "湖北": (30.60, 114.30), "湖南": (28.20, 112.90),
    "广东": (23.10, 113.30), "广西": (22.80, 108.40), "海南": (20.00, 110.30),
    "重庆": (29.60, 106.50), "四川": (30.60, 104.10), "贵州": (26.70, 106.70),
    "云南": (25.00, 102.70), "西藏": (29.70, 91.10), "陕西": (34.30, 108.90),
    "甘肃": (36.10, 103.80), "青海": (36.60, 101.80), "宁夏": (38.50, 106.20),
    "新疆": (43.80, 87.60), "台湾": (25.00, 121.50), "香港": (22.30, 114.20),
    "澳门": (22.20, 113.50),
}


def _prov_norm(p: str) -> str:
    """IP2Region 省份名 → 前端 CHINA_PROVINCE_PATHS 键（四川省→四川 / 广西壮族自治区→广西）。"""
    for suf in ("特别行政区", "壮族自治区", "维吾尔自治区", "回族自治区", "自治区", "省", "市"):
        if p.endswith(suf) and len(p) > len(suf):
            return p[: -len(suf)]
    return p


def _city_norm(c: str) -> str:
    """IP2Region 城市名 → 简写（成都市→成都；'0'/空 → ''）。"""
    if not c or c == "0":
        return ""
    return c[:-1] if c.endswith("市") and len(c) > 1 else c


def _city_coords(city: str, prov: str) -> tuple[float | None, float | None]:
    """中国城市坐标：城市表 → 省会兜底；都没有 → (None, None)（不画橙点）。"""
    return (CN_CITY_COORDS.get(city)
            or CN_CITY_COORDS.get(city + "市")
            or CN_PROV_COORDS.get(prov, (None, None)))


def _xdb_result(details: str) -> dict | None:
    """IP2Region details（国家|省份|城市|ISP|ISO）→ resolve() 结果；
    仅中国段采信（境外仍走 GeoLite2，能带坐标画城市点）；Reserved/0 段与空值返回 None。"""
    parts = (details or "").split("|")
    if not parts or parts[0] != "中国":
        return None
    prov_raw = parts[1] if len(parts) > 1 else ""
    city = _city_norm(parts[2] if len(parts) > 2 else "")
    if prov_raw in ("", "0"):
        out: dict = {"kind": "cn", "region": CN_FALLBACK}
    else:
        prov = _prov_norm(prov_raw)
        out = {"kind": "cn", "region": prov}
    if city:
        out["city"] = city
        out["city_zh"] = city
        lat, lon = _city_coords(city, out["region"])
        out["lat"], out["lon"] = lat, lon
    return out


def db_path() -> Path:
    raw = (os.getenv("RELAY_GEOIP_DB") or "").strip()
    if raw:
        return Path(raw).expanduser()
    return DATA_DIR / "GeoLite2-City.mmdb"


def _load() -> object | None:
    """惰性加载 mmdb；文件缺失/损坏返回 None（功能静默关闭）。"""
    global _reader, _loaded_for, _load_failed
    p = db_path()
    if _reader is not None and _loaded_for == str(p) and p.exists():
        return _reader
    if _load_failed:
        return None
    if not p.exists():
        return None
    try:
        import maxminddb

        _reader = maxminddb.Reader(str(p))
        _loaded_for = str(p)
        return _reader
    except Exception:
        _load_failed = True
        _reader = None
        return None


def available() -> bool:
    """mmdb 或 IP2Region xdb 任一可用即可（中国段 xdb 优先，境外走 mmdb）。"""
    if _load() is not None:
        return True
    try:
        from app import ip2xdb
        return ip2xdb.available()
    except Exception:
        return False


def reset() -> None:
    """测试用：清缓存（换 db 文件后重新加载）。"""
    global _reader, _loaded_for, _load_failed
    _reader = None
    _loaded_for = None
    _load_failed = False
    try:
        from app import ip2xdb
        ip2xdb.reset()
    except Exception:
        pass


def resolve(ip: str) -> dict | None:
    """解析单个 IP → {"kind": "cn"|"world"|"local", "region": 省/国名}。

    私有/回环地址归 local（局域网/反代出口）；解析失败归 unknown。
    双库（IP2Region xdb + GeoLite2 mmdb）都缺失时返回 None（调用方据此降级）。
    """
    s = (ip or "").strip()
    if not s:
        return {"kind": "unknown", "region": "未知"}
    try:
        a = ipaddress.ip_address(s)
    except ValueError:
        return {"kind": "unknown", "region": "未知"}
    if a.is_private or a.is_loopback or a.is_unspecified or a.is_reserved:
        return {"kind": "local", "region": "局域网/代理出口"}
    # IP2Region xdb 优先：中国段精确到城市（免费 GeoLite2 很多国内家宽/运营商 IP 只到国家）；
    # xdb 未命中/境外段落回 GeoLite2
    if a.version == 4:
        try:
            from app import ip2xdb
            xres = _xdb_result(ip2xdb.query(s))
        except Exception:
            xres = None
        if xres is not None:
            return xres
    reader = _load()
    if reader is None:
        return None          # 双库都缺：端点层面 already 走 available() 降级
    try:
        rec = reader.get(s)
    except Exception:
        return {"kind": "unknown", "region": "未知"}
    if not rec:
        return {"kind": "unknown", "region": "未知"}
    country = (rec.get("country") or {}).get("iso_code") or ""
    ci = city_info(rec)
    if country == "CN":
        divs = rec.get("subdivisions") or []
        div = divs[0] if divs else {}
        iso = (div.get("iso_code") or "").replace("CN-", "")
        zh = CN_PROVINCES.get(iso)
        if not zh:
            zh = CN_PROVINCES_EN.get((div.get("names") or {}).get("en") or "")
        if zh:
            out = {"kind": "cn", "region": zh}
        else:
            # 查不到省级（移动/运营商/IPv6 段常见）：归「中国大陆」兜底桶，不当未知
            out = {"kind": "cn", "region": CN_FALLBACK}
        out.update(ci)
        return out
    zh = COUNTRY_ZH.get(country)
    if zh:
        out = {"kind": "world", "region": zh}
    else:
        en = (rec.get("country") or {}).get("names", {}).get("en") or "未知"
        out = {"kind": "world", "region": en}
    out.update(ci)
    return out
