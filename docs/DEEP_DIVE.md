# LLM Relay · 深入参考

> README 的补充材料：思考控制详解、管理 API 完整清单、用量字段全表、实现要点 / 踩坑记录。
> 日常使用看 [README.md](../README.md) 就够，这份文档写给想改代码或排障的人。

## 思考控制详解

### 控制台交互

控制台「编辑平台」→「模型级思考设置」里，每个模型一行两个下拉：

```
deepseek-reasoner   [关闭思考 ▼]  [自动（按平台，同 Hermes）▼]
deepseek-chat       [跟随客户端 ▼]
```

第二个下拉是**具体传给上游的什么字段**（只在选「关闭思考」时出现）。
控制台对应两个控件：**思考 / 推理策略** + **上游关闭字段**，第二项下面会实时显示
「将写入上游请求：{...}」，所见即所得。

### 也可以用 JSON 配

```json
"extra": {
  "model_reasoning": {
    "deepseek-reasoner": {"policy": "off", "field": "enable_thinking"},
    "beam": "off"
  }
}
```

- `"off"` 这种字符串写法＝字段走自动；key 用**上游真实模型名**，支持 `"*"` 兜底
- 也可以手动指定字段（下拉里那几项），或者用 `extra.thinking_off_payload` 直接写一个 dict 盖掉
- 只关某个模型：`"model_reasoning": {"glm-4.6": "off", "*": "auto"}`
- 平台级 `extra.reasoning` 已废弃并忽略（老配置里写着也不生效），只认 `model_reasoning`

### 「关闭」到底改了什么

策略为 `off` 时，网关对**发给上游的请求体**做两步，让上游收到的请求本身就是关闭状态：

1. 清掉客户端传来的所有思考字段（`reasoning_effort` / `reasoning` / `thinking` / `enable_thinking` /
   `chat_template_kwargs.enable_thinking` / `thinking_budget` …）——客户端硬要思考也不行；
2. 按 `thinking_off` 模式**显式写入关闭字段**。

| `thinking_off` | 实际写进上游请求 | 适合 |
| --- | --- | --- |
| `auto`（默认） | 按平台名/域名自动挑下面某一个 | 懒人模式 |
| `reasoning_effort` | `{"reasoning_effort": "none"}` | OpenAI 新模型、agnes、ollama（实测安全） |
| `enable_thinking` | `{"enable_thinking": false}` | Qwen3 / DashScope / 硅基流动 |
| `thinking_type` | `{"thinking": {"type": "disabled"}}` | 智谱 GLM、Anthropic |
| `chat_template` | `{"chat_template_kwargs": {"enable_thinking": false}}` | vLLM / SGLang 自建 |
| `none` | 什么都不写（只做第 1 步） | 平台确实没有关闭开关时 |

`auto` 的判断依据是 `平台 id + 显示名 + base_url`：

- 命中 `zhipu/glm/bigmodel/z.ai` → `thinking_type`
- 命中 `qwen/dashscope/bailian/siliconflow` → `enable_thinking`
- `type=anthropic` → `thinking_type`
- Ollama（ollama.com、:11434、名字带 ollama）→ `reasoning_effort` + `think:false`
- OpenRouter / Nous Portal → `reasoning:{"enabled":false,"effort":"none"}`
- MiniMax → `thinking:{"type":"disabled"}`
- 其余 → `reasoning_effort`

### 实测数据（agnes-3.0-flash，同一句话各跑 3 次）

| 配置 | completion / reasoning tokens |
| --- | --- |
| auto（什么都不传） | 2/0 · 2/0 · 2/0 |
| off + `reasoning_effort:"none"` | 2/0 · 2/0 · 2/0 |
| off + `enable_thinking:false` | **25/22 · 45/42 · 46/43** |

`enable_thinking:false` 在 agnes 上不但没关掉，反而把思考打开了（3/3 复现）。
所以「关闭字段」不能一刀切，必须按平台选 —— 这就是 `thinking_off` 存在的原因。
ollama（`ollama.com/v1`）实测接受 `reasoning_effort:"none"`，正常返回 `Hi there!`。

### 各平台关闭能力一览

| 平台 | 能关吗 | 怎么做 |
| --- | --- | --- |
| 阿里百炼 / Qwen3 混合模型 | ✅ | `enable_thinking: false`（开思考时必须 `stream=true`） |
| 硅基流动 | ✅ | `enable_thinking: false` |
| 智谱 GLM-4.6 / 4.5 | ✅ | `thinking: {"type":"disabled"}`；GLM-5.3 系列强制思考，传了会报错 |
| OpenAI 推理模型 | ✅ | `reasoning_effort: "none"`（或 `"minimal"`） |
| Anthropic | ✅ | `thinking: {"type":"disabled"}`；旧模型不支持 thinking 时把模式改成 `none` |
| agnes | ✅ | `reasoning_effort: "none"` |
| DeepSeek | ❌ | `deepseek-reasoner` 永远思考，换 `deepseek-chat` |
| Kimi / MiniMax | ❌ | 思考是独立模型名，换模型 |

### 注意

- 「关闭思考」是**在发给上游的请求字段上**关闭，不是把思考内容藏起来——上游不思考了，自然也没有思考内容。
- 一个模型关不掉（比如 DeepSeek 的 `deepseek-reasoner`）时，正确的做法是换模型，而不是靠隐藏。

## 用量统计字段全表

`data/relay.db` 的 `requests` 表，每条请求一行：

| 字段 | 说明 |
| --- | --- |
| `ts` / `request_id` | 时间、请求 ID（响应头 `x-request-id` 同一个） |
| `platform` / `model` | **上游平台** / **上游真实模型名** |
| `model_alias` | 下游请求里写的原始 model |
| `in_format` | 下游协议：`openai` / `anthropic` |
| `stream` / `status` / `http_status` / `error` | 流式标记、成功/失败、上游状态码、错误摘要 |
| `prompt_tokens` / `completion_tokens` / `total_tokens` | 用量（优先取上游真实 usage） |
| `cached_tokens` / `cache_write_tokens` | 命中缓存 / 写入缓存的 token（Anthropic、OpenAI 缓存） |
| `reasoning_tokens` | 思维链 token |
| `estimated` | 1 = 上游没给 usage，本值是估算 |
| `latency_ms` / `ttft_ms` | 总耗时 / 首字延迟 |
| `client_ip` | 来源 IP（挂反代时是 `X-Forwarded-For` 还原出来的真实 IP） |
| `reasoning_policy` | 这次请求实际生效的思考策略：`off` / `auto` / `low` / `medium` / `high` |
| `key_hint` | 这次用的是平台里的哪个 API Key（打码后，如 `sk-3x3…mnD1`） |
| `key_index` | 对应的 key 序号（0 起，按平台配置里的顺序） |
| `user_id` | 属于哪个账号（多租户） |

## 管理员的账号维度

顶部「数据视角」选 **全部账号（汇总）** 或某个具体账号后，三个页面都会跟着变：

| 位置 | 表现 |
| --- | --- |
| 概览 | 多出「各账号用量」面板（请求 / 输入 / 输出 / 合计），点一行直接切到该账号 |
| 用量统计 | 下拉多一个 **按账号**，表格与曲线都按邮箱聚合 |
| 请求日志 | 多一列 **账号** |

普通账号看不到这些选项；即使手动调 `group_by=account`，也只会返回他自己一行。

## 管理 API 完整清单（前缀 `/api`）

控制台鉴权优先会话 Cookie，其次 Bearer（账号 API Key）：

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/overview` | 汇总 + 路径信息 |
| GET | `/api/providers` | 平台列表（key 打码）+ 预设模板 |
| POST | `/api/providers` | 新增/更新平台（提交掩码 key 会沿用它） |
| DELETE | `/api/providers/{platform}` | 删除平台 |
| POST | `/api/providers/{platform}/refresh-models` | 从上游拉模型列表并写回 |
| POST | `/api/providers/{platform}/test` | 真发一条 `ping` 验证 key / base_url |
| GET | `/api/models` | 全部 `平台/模型` |
| GET | `/api/stats/summary?days=30&group_by=platform` | 聚合统计；`group_by` 可选 `platform` / `model` / `model_only` / **`key`** / **`dkey`**（按下游 key 备注名） / `day` / `format` / **`account`**（管理员） |
| GET | `/api/stats/series?days=30&group_by=platform` | 每日 token 曲线；`group_by` 可选 `platform` / `model` / `key` / **`dkey`** / `total`，返回 `{days, keys, rows}`，前端靠它做「点选/多选 → 曲线联动」 |
| GET | `/api/stats/records?limit=200` | 原始请求日志；每行含 `dkey_label`（发起请求的下游客户端 key 备注名，无备注为 `(无备注)`、key 已删为 `(key 已删除)`、历史未记录为 `(未记录)`）；`from`/`to` 支持纯日期（`YYYY-MM-DD`）或到秒的日期时间（`YYYY-MM-DDTHH:MM:SS`），`range=15m\|30m\|1h\|6h\|today` 为锚定服务端当前时间的相对快捷区间（日志页默认近 1 小时） |
| GET | `/api/stats/regions?range=today\|7d\|30d` | 客户端 IP 地域分布（概览页「地域访问地图」）：按 IP 读时解析离线双库——中国段优先 IP2Region `IP2Region.xdb`（纯 Python 二进制查询 `app/ip2xdb.py`，精确到城市+运营商），境外与未命中落 GeoLite2（见下），私有段 local；返回 `db`（ok\|missing，双库都缺才 missing）+ 按 count 降序的 `data` + 城市热点 `cities`（含城市名/经纬度/次数，前端画到中国省地图上的橙色热点圆点，面积 ∝ 次数；中国城市坐标内置城市表、未收录落省会坐标）；库路径 `RELAY_IP2REGION_XDB` 默认 `data/IP2Region.xdb`、`RELAY_GEOIP_DB` 默认 `data/GeoLite2-City.mmdb`，见 `app/geoip.py` + `app/ip2xdb.py` |
| GET | `/api/stats/export.csv?days=30` | 导出 CSV |
| GET | `/api/settings` | 目录、版本、我的 Key 列表 |

鉴权头写法：`Authorization: Bearer *** → 以该 Key 所属账号身份操作。

## 下游 Key 备注、过期与统计

- **数据结构**：`api_keys` 表新增 `expires_at`（ISO 时间字符串，NULL = 永不过期）；`requests` 表新增 `dkey_id`（发起请求的客户端 key 外键，可空）。旧库启动时自动迁移（`ALTER TABLE` 补列）。
- **鉴权**：relay 入口 `authenticate()` 在 key 校验后检查禁用/过期状态：key 被吊销（`revoked=1`）返回 401「已禁用」；`now >= expires_at` 返回 401「已过期（过期时间 YYYY-MM-DD HH:MM）」——报错文案回显**实际过期时间**，控制台 Cookie 会话不受影响（仅客户端 Bearer 路径拦截）。过期/禁用 401 会作为一条 `status=error` 请求记录写进请求日志（带 `dkey_id` 归属，日志页「下游 Key」列可查；`error` 字段写明拦截原因），而 missing/invalid key 不落账（防攻击刷库）。服务端 `log.warning` 同步记录 key 打码值 + 过期时间。
- **Key 管理 API**：
  - `POST /api/keys` body `{label?, expires_at?}` — `expires_at` 接受 `YYYY-MM-DD` / `YYYY-MM-DD HH:MM` / ISO 带时区；留空 = 永不过期；坏格式 400。
  - `PATCH /api/keys/{id}` body `{label?, expires_at?}` — 改备注 / 过期时间（传 `""` 或 `null` 恢复永不过期）。
- **统计归并**：`summary` / `series` / `hourly`（`split=dkey`）的 `group_by="dkey"` 维度一律按**备注名**聚合：有备注用备注；无备注 `(无备注)`；key 已删 `(key 已删除)`；历史 `dkey_id` 为空的行 `(未记录)`。日志页「下游 Key」列、概览「近期活动」按下游 key 堆叠、用量页「按下游 key」下拉都走这一条线。
- 回归测试：`tests/test_relay.py` R68。

## 邮箱确认（magic link）

自助注册（第二个账号起）与邮箱确认的完整链路：

- `users.status`：`active` / `pending` / `disabled`。首个管理员注册时直接
  `active`（零配置引导，豁免确认）；之后的自助注册账号为 `pending`——
  登录返回 403 `email_unverified`，且**不发放 API Key**（数据面权限在
  确认后才生效）。
- 确认令牌：`t=<payload>.<sig>`，其中 `payload = base64url("v1:<email>:<unix_ts>")`，
  `sig` 为 `HMAC-SHA256(密钥, "payload.ts")` 的前 20 字节 hex（防链接被拼接篡改）。
  密钥存 `meta.email_verify_key`（首次使用自动生成，64 位 hex）。
- 服务端校验三件事：**签名有效、时间戳在 30 分钟窗口内、邮箱与账号一致**；
  全部通过 → 置 `active` 并发默认 API Key，返回 HTML 确认页
  （`GET /api/auth/verify-email?t=...`，公开端点，邮件里点进来无会话）；
  过期 / 篡改分别给出「链接已失效 / 链接无效」页。
- `POST /api/auth/resend-verify`（公开，登录页入口）：同一邮箱 60 秒冷却、
  每日 5 次上限（计数存 `meta`）；对不存在的邮箱返回统一话术，不泄露账号存在性。
- 邮件发送：`RELAY_SMTP_HOST` 配了走 SMTP（`smtplib`，默认 587/STARTTLS），
  否则**日志模式**把确认链接打出来（`[mail/log]`），内网 / 无邮箱环境也能引导。
  **发信配置可以在控制台改**（仅管理员）：「设置 → 邮箱发信设置（SMTP）」
  存进 `meta` 表（`mail_mode` / `mail_smtp_host` / `mail_smtp_port` /
  `mail_smtp_user` / `mail_smtp_pass` / `mail_smtp_from` / `mail_smtp_starttls`），
  优先级 **控制台设置 > 环境变量 > 默认**；密码只在页面上打码展示，
  保存时留空 = 沿用旧值（明文不落前端、API 响应也不带明文）。
  相关端点：`GET /api/settings/mail`（打码视图）、`POST /api/settings/mail`（保存）、
  `POST /api/settings/mail/test`（发测试邮件验证配置，失败返回 ok=false 不抛错）。
  确认邮件是 **multipart/alternative**：HTML 版带可点按钮链接（`<a>`，
  内联样式），纯文本版兜底完整 URL（老客户端直接显示链接）——
  见 `mailer.send_verification_email`（R65 回归）。
- 链接域名取自请求 `base_url`（反代下即公网域名），30 分钟窗口到期后
  用「重发确认邮件」取新链接。
- **pending 账号 30 分钟未确认自动清理**：`Store.purge_stale_pending()`
  删掉 `created_at` 超过 30 分钟的 pending 用户（连同其会话与 Key），
  在 `register` / `resend-verify` / `GET /api/users` 入口惰性调用——
  超时账号不存在了，重发走「重新注册」话术，邮箱立即可被重新注册。
- **管理员彻底删除账号**：`DELETE /api/users/{id}` → `Store.delete_user_full()`
  删 sessions / api_keys / users 行，`requests` 历史保留作审计；
  守卫：不能删自己、不能删「最后一个活跃管理员」。

## 实现要点 / 踩坑记录

- **流式透传要"旁路解析"**：一边把上游 SSE 帧转发给客户端，一边用 `SSEParser`
  增量解析出 `usage`。不能先读完再发，否则首字延迟被完全吃掉。
- **主动要 usage**：客户端没开 `stream_options.include_usage` 时，中转站自己开；
  只含 usage 的尾包会被**吞掉**（客户端看到的内容与它请求的完全一致），
  但用量已经记下。这是"精确统计"和"不改变客户端体验"的折中。
- **Anthropic 的 input_tokens 不含缓存**：`总输入 = input_tokens + cache_read + cache_creation`，
  折算成 OpenAI 口径的 `prompt_tokens`，另外单独记缓存命中。
- **Anthropic 要求 `max_tokens` 必填**、消息必须 user/assistant 交替 —— 转换时补默认值并合并同角色消息。
- **httpx 流式响应必须 `aclose()`**：在生成器的 `finally` 里收尾，客户端断开也不漏连接。
- **只有在上游还没开始回内容前才能重试**：一旦有字节发给客户端，换 key 重试会造成重复输出。
