# 更新日志

格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/)，版本号用 SemVer。

## [Unreleased]

## [1.0.0] - 2026-11-08

首个开源版本。

### 核心能力
- 双协议 × 双协议转发：下游 OpenAI / Anthropic × 上游 OpenAI 兼容 / Anthropic 原生，4 种组合自动翻译（含 SSE 真流式、工具调用）
- 精确 token 统计：主动向上游索取 usage（流式也精确），拿不到才估算并标 `est`；缓存命中 / 思维链 token 分列
- 多 key 轮换 + 冷却：401/429/5xx 自动换 key 重试，出错 key 冷却 60s
- **聚合模型**：一个聚合名（如 `fast` / `best`）映射多条「平台/模型」条目，RR 轮询 + 出错条目 60s 冷却自动切换；对下游暴露为裸名（owned_by=aggregation），条目 ≤20、去重、可 5s 热加载
- 按模型关思考：关闭字段按平台自动选（Ollama / 智谱 / 百炼 / Gemini / Kimi … 实测规则表）
  - **Google 修复**：Google 的 OpenAI 兼容端点不再接受顶层 `google.thinking_config`（发即 400 "Unknown name google"），改走 OpenAI 系 `reasoning_effort`（gemini-3.1 实测 none/minimal/high 全收）；Gemini 3/4 思考模型工具调用的 `thought_signature` 自动补官方哨兵值避免 400，可 `extra.thought_signature_bypass=false` 关闭
- 多账号隔离：邮箱 + 密码，首个注册 = 管理员；第二个起 magic link 邮件确认（HMAC + 30 分钟时效，pending 30 分钟自动清理）
- Web 控制台（单文件零 CDN）：平台 CRUD + 21 家预设模板、一键拉模型、连通性测试、
  概览（今日 KPI / 活跃日历 / 近期活动堆叠图 / 地域访问地图）、用量统计（P95 / TTFT / 缓存命中率 /
  星期×时段热力图 / CSV 导出）、请求日志（全量筛选 + 服务端时间边界 + 分页）
- 下游 key 管理：备注 + 过期（精确到日，当天仍可用）；过期 / 禁用 401 写审计记录；按备注名聚合统计
- 平台级出口代理（http/https/socks5/socks5h）
- 地域访问地图：双离线 GeoIP 库（IP2Region xdb 中国到城市 + GeoLite2 境外到国家），手绘 SVG 中国省地图 + 城市热点 + 省→城市下钻
- 邮箱发信设置：控制台配 SMTP + 发测试邮件；三级优先级（页面 > 环境变量 > 日志模式兜底）

### 部署
- 裸跑 / Docker / WSL / Nginx / Caddy / Cloudflare Tunnel / 子目录 / 自跑 HTTPS
- 上游配置 5s 热加载；SQLite 启动自动加列迁移（旧库直接升级）
- 实测容量基线与推荐规格（见 DEPLOY.md §四）
