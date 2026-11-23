"""模型费用配置与成本计算（v2.1.0 R153）。

配置面：租户配置（upstreams.json）里的 `{"_kind": "pricing", "pricing": {...}}` 标记行，
与聚合模型标记行同构——Registry 热加载，手改文件 5s 内生效；保存走 /api/pricing。

价格字段（单位：元）：
- request_fee       请求费：每次调用固定金额（与 token 数无关，覆盖生成类等非 token 场景）
- input_per_1k      输入费：元 / 1K 输入 token
- cache_hit_per_1k  缓存命中费：元 / 1K 缓存命中 token
- output_per_1k     输出费：元 / 1K 输出 token
- second_fee      单价（元/秒）：v2.1.1 R176 新增，视频生成（请求 seconds 字段）/ 音频转写
  （上游响应时长）按「秒」计价；0=关闭，其余 kind（图片/TTS/文本）不填。
  各平台非文本计费模式的覆盖：图片按次 → 请求费；TTS 按字数 → 输入费（输入文本已计
  prompt_tokens，元/1K 输入 token）；视频/转写按秒 → 本字段。

计费公式（记账时冻结，改价格只影响之后的请求）：
  billable_input = max(0, prompt_tokens - cached_tokens)
  —— prompt_tokens 在两种上游协议里都包含缓存命中部分（OpenAI 的
  prompt_tokens_details.cached_tokens 是子集；Anthropic 的 prompt = input + cache_read
  + cache_write），缓存命中按 cache_hit_per_1k 单算，不再并入输入费，避免重复计费。
  cost = request_fee + billable_input/1000*input_per_1k
       + cached_tokens/1000*cache_hit_per_1k + completion_tokens/1000*output_per_1k
       + seconds*second_fee      # v2.1.1 R176：视频（请求 seconds）/ 音频转写（响应时长）

口径：只对成功请求计费（status=ok / aborted——流被客户端掐断时已消耗的 token 照常计；
status=error 一律 0）。模型名查不到价格条目 = 免费（不计费），不做默认价兜底。
"""
from __future__ import annotations

# 价格字段：second_fee（元/秒）v2.1.1 R176 新增——视频（请求 seconds）/ 音频转写（响应时长）
# 按秒计价，计量秒数经 gateway record(seconds=...) 传入，日志 usage_detail 列留痕。
PRICE_FIELDS: tuple[str, ...] = (
    "request_fee", "input_per_1k", "cache_hit_per_1k", "output_per_1k", "second_fee",
)

#: 金额保留位数（元）。1K token 级单价通常 4-6 位小数就够，6 位足够且不丢精度。
COST_DECIMALS = 6


def parse_pricing(raw) -> dict:
    """{模型名: {价格字段}} 归一 + 校验。

    返回干净的 dict（零值条目与空名字被剥离，缺字段补 0.0）；raw 不是对象、
    价格不是数字或为负数 → ValueError（/api/pricing 转 400）。
    """
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("费用配置必须是对象：{模型名: {request_fee, input_per_1k, ...}}")
    out: dict[str, dict] = {}
    for name, row in raw.items():
        name = str(name).strip()
        if not name:
            continue
        if not isinstance(row, dict):
            raise ValueError(f"模型 {name!r} 的费用条目必须是对象（价格字段：request_fee/input_per_1k/cache_hit_per_1k/output_per_1k/second_fee）")
        vals: dict[str, float] = {}
        for f in PRICE_FIELDS:
            v = row.get(f, 0)
            try:
                v = float(v)
            except (TypeError, ValueError):
                raise ValueError(f"模型 {name!r} 的 {f} 必须是数字（元）")
            if v < 0:
                raise ValueError(f"模型 {name!r} 的 {f} 不能为负数")
            vals[f] = v
        if any(v > 0 for v in vals.values()):
            out[name] = vals
    return out


def compute_cost(pricing: dict | None, model_name: str,
                 prompt_tokens: int = 0, completion_tokens: int = 0,
                 cached_tokens: int = 0, billed: bool = True, seconds: float = 0.0):
    """按模型价格条目算一次调用的费用。

    返回 (合计, 请求费, 输入费, 缓存命中费, 输出费, 秒费) 六元组（元，已按 6 位小数取整）。
    seconds：v2.1.1 R176 可计费秒数（视频=请求 seconds；音频转写=上游响应时长；其余 0），
    秒费 = max(0, seconds)*second_fee。无条目 / 全 0 / billed=False / 模型名为空 → 全 0。
    """
    p = (pricing or {}).get(str(model_name or "").strip())
    if not p or not billed:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    prompt = max(0, int(prompt_tokens or 0))
    cached = max(0, int(cached_tokens or 0))
    completion = max(0, int(completion_tokens or 0))
    billable_input = max(0, prompt - cached)
    r = COST_DECIMALS
    # 分项各自取整后合计再取整：保证「分项合计 == 总费用」恒成立（图表堆叠口径一致），
    # 避免 round(合计) 与 round(分项) 之和差 1e-6 的浮点噪声。
    creq = round(float(p.get("request_fee", 0.0)), r)
    cin = round(billable_input / 1000.0 * float(p.get("input_per_1k", 0.0)), r)
    ccache = round(cached / 1000.0 * float(p.get("cache_hit_per_1k", 0.0)), r)
    cout = round(completion / 1000.0 * float(p.get("output_per_1k", 0.0)), r)
    csec = round(max(0.0, float(seconds or 0.0)) * float(p.get("second_fee", 0.0)), r)
    return (round(creq + cin + ccache + cout + csec, r), creq, cin, ccache, cout, csec)
