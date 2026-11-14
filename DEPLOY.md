# LLM Relay · 部署

> 两条路线：**一、docker-compose 编排部署（服务器）** 和 **三、WSL 部署（Windows 本地）**。
> 反向代理怎么配统一放在**第二章**。容量怎么估、有哪些坑，见**第四章**。

## 一、docker-compose 编排部署（服务器）

目标机器：Linux + Docker（`docker compose` v2）。不需要装 Python。

### 1. 传上去

在**本机**执行（把 IP / 路径换成你的）：

```bash
scp llm-relay-*-src-*.tar.gz root@<服务器IP>:/opt/
```

### 2. 解包

```bash
ssh root@<服务器IP>
mkdir -p /opt/llm-relay && cd /opt
tar -xzf llm-relay-*-src-*.tar.gz
mv llm-relay-1.0.0/* /opt/llm-relay/
cd /opt/llm-relay
ls           # 应该看到 app/ deploy/ Dockerfile docker-compose.yaml README.md ...
```

### 3. 配置

```bash
# 3.1 上游平台（必填）
cp config/upstreams.example.json config/upstreams.json
vi config/upstreams.json         # 把 "sk-把你的-key-填这里" 换成真 key

# 3.2 环境变量（可选）
cp .env.example .env
vi .env                          # 改端口 / 子路径前缀

# 3.3 数据目录（首次）
mkdir -p data
```

`config/` 与 `data/` 是 bind mount：**删容器不丢配置、不丢用量库**；
备份只需 `tar -czf ... config data`。

### 4. 起服务

```bash
docker compose build
docker compose up -d
docker compose ps                # 等 STATUS 变成 healthy
docker compose logs --tail=50
```

首次启动会打印控制台地址。
想多挂个 Nginx 容器（不想用宿主机 Nginx 时）：`docker compose --profile with-nginx up -d`，
证书放 `deploy/certs/`（`fullchain.pem` / `privkey.pem`）。

### 5. 首次进入

浏览器打开 `http://<服务器IP>:8800/`（配好反代后是你的域名）：

1. 页面会自动切成**注册**（库里还没有账号）
2. **第一个注册的账号自动成为管理员**（豁免邮箱确认，注册即可登录），
   并自动发一把默认 API Key
3. 登录后可在「设置 → 我的 API Key」查看 / 再建 key 给客户端用

> 第二个及以后的账号注册后进入**待确认**状态：系统会往其邮箱发一封
> 带**确认链接（magic link，30 分钟有效）**的邮件，点链接才激活并能登录；
> 未确认前不能登录、没有 API Key。登录页支持「重发确认邮件」。

> 不配 SMTP 时邮件走**日志模式**：确认链接直接打进容器日志
> （`docker compose logs -f` 里搜 `[mail/log]`），浏览器打开该链接即完成确认——
> 内网 / 无邮箱环境也能完成第二个账号的引导。

邮件发送有两种配置方式，**页面优先**：

1. **控制台（推荐）**：管理员登录 →「设置 → 邮箱发信设置（SMTP）」填
   SMTP 服务器 / 端口 / 账号 / 密码（授权码）/ 发件人 / 模式（smtp 或 log），
   可直接在页面**发测试邮件**验证。配置存在数据卷 `data/relay.db` 里，随卷备份。
2. 环境变量（备选 / 兜底）：在 `.env` / `compose.yml` 里加：

```
RELAY_SMTP_HOST=smtp.example.com
RELAY_SMTP_PORT=587            # 465 时 STARTTLS 关掉或端口按服务商来
RELAY_SMTP_USER=noreply@example.com
RELAY_SMTP_PASS=***
RELAY_SMTP_FROM="LLM Relay <noreply@example.com>"   # 可选
RELAY_MAIL_MODE=smtp                                  # 可选：smtp / log（缺省自动）
```

> `config/upstreams.json` 里配的平台，在管理员账号下立即可见。
> 之后新建的账号各自独立，看不到别人的平台和用量。

### 6. 日常运维

```bash
docker compose logs -f --tail=100     # 看日志
docker compose restart                # 重启
docker compose down && docker compose up -d   # 重建容器（volume 保留）

# 备份（配置 + 用量库都在这里；主密钥 master.key 刻意排除——它必须与数据分开保管）
tar -czf /opt/backup-$(date +%F).tar.gz -C /opt/llm-relay --exclude='config/master.key' config data

# 升级：覆盖代码后重建
docker compose build && docker compose up -d
```

#### 主密钥（config/master.key，v1.1.0 起）

API Key 自 v1.1.0 起加密存储（AES-GCM）；各账号的上游平台 key 自 v1.1.1 起
加密存在 `config/tenants/u<id>/upstreams.json`（`enc_keys` 字段，每账号一把由 S
派生的 K_t）。解密密钥 = 主密钥 S，来源优先级：

1. 环境变量 `RELAY_KEY_SECRET`（hex/base64，64 位 hex 推荐——自建后写进 compose 的 `.env`，不进数据目录）
2. `config/master.key`（首次启动自动生成，0600；日志里会大声提醒一次）

铁律：

- **S 丢失 = 所有已加密的 API Key 与上游 key 永久不可恢复**（服务会拒绝启动，fail-fast 而不是悄悄换一把钥匙）
- S **不进数据备份、不进发布包**（上面的备份命令已 `--exclude`）；单独记到密码管理器/纸质
- 全新部署（没有任何加密材料）可以直接启动：S 自动生成 0600；只有「已有加密材料 + 找不到 S」才拒绝启动
- 恢复备份后先放好原 S 再起服务：`docker compose up -d`
- 文件权限：`chmod 600 /opt/llm-relay/config/master.key`（同目录的 `upstreams.json`、`relay.db`、`.env` 也建议 600）
- 升级回滚：v1.1.1 起租户配置是加密的（v1.1.0 代码读不了 `enc_keys`）——回滚要一并恢复升级前的 config/ 备份

### 7. 自检

```bash
curl -s http://127.0.0.1:8800/healthz        # {"ok":true,...}
curl -s http://127.0.0.1:8800/v1/models -H "Authorization: Bearer *** sk-relay-key>"
curl -N http://127.0.0.1:8800/v1/chat/completions \
  -H "Authorization: Bearer *** sk-relay-key>" \
  -H "Content-Type: application/json" \
  -d '{"model":"平台/模型","stream":true,"messages":[{"role":"user","content":"你好"}]}'
```

### 8. 地域访问地图（可选，双 GeoIP 库）

概览页「地域访问地图」按客户端 IP 定位请求来源：中国到**省**、境外到**国家**、
私有/回环地址归「局域网/代理出口」。地图为离线内嵌 SVG（无外部依赖），定位走两个可选的离线库：

- **IP2Region `IP2Region.xdb`**（中国段优先，开源免费、精确到城市+运营商）：
  免费 GeoLite2 对国内家宽/运营商 IP 常常只解析到国家级，放了这个库国内 IP 就能落到城市。
- **MaxMind `GeoLite2-City.mmdb`**（境外 + xdb 未命中兜底，需 [MaxMind 账号](https://www.maxmind.com/en/geolite2) 下载，建议每月更新一次）。

```bash
# 1a. IP2Region xdb（约 11MB，可选但强烈推荐；周更，从官方仓库拉最新版）
mkdir -p /opt/llm-relay/data
curl -sSL -o /opt/llm-relay/data/IP2Region.xdb \
  "https://raw.githubusercontent.com/lionsoul2014/ip2region/master/data/ip2region_v4.xdb"
#    国内服务器拉 GitHub 慢的话可换 gitee 镜像：https://gitee.com/lionsoul/ip2region
#    （注意：gitee 大文件 raw 链接需要登录，直接浏览器下载再 scp 上去）

# 1b. GeoLite2-City.mmdb（.mmdb，不是 .gz；账号下载页选 maxmind-databases）
mkdir -p /opt/llm-relay/geoip
#    把 GeoLite2-City.mmdb 放到 /opt/llm-relay/geoip/

# 2. compose 启用：取消注释两行
#    - environment 里 RELAY_GEOIP_DB: /app/geoip/GeoLite2-City.mmdb
#    - volumes 里    - ./geoip:/app/geoip
docker compose up -d   # 重建后生效
```

- 两个文件也可以都放进 `data/`（默认目录）：默认读 `data/IP2Region.xdb` 与
  `data/GeoLite2-City.mmdb`，不设环境变量；单独指路用 `RELAY_IP2REGION_XDB` / `RELAY_GEOIP_DB`。
- **不配置时功能自动关闭**：概览页该面板显示配置提示，其余页面与 API 不受影响
  （端点 `GET /api/stats/regions` 返回 `db="missing"`）；只配 xdb 时境外 IP 解析不到、只配 mmdb 时国内 IP 可能只到国家级。
- 定位口径说明：公司出口 / 代理 / CDN 后的请求会合并到同一地区，属 IP 定位固有特性。

### 附：非 Docker 运行（systemd 开机自启）

```ini
[Unit]
Description=LLM Relay
After=network.target

[Service]
WorkingDirectory=/opt/llm-relay
Environment=RELAY_HOST=127.0.0.1 RELAY_PORT=8800
ExecStart=/opt/llm-relay/.venv/bin/python run.py
Restart=always

[Install]
WantedBy=multi-user.target
```

## 二、反向代理配置

推荐形态：**网关只监听 `127.0.0.1`，Nginx / Caddy 终结 TLS**（compose 里端口默认就只绑 127.0.0.1，
防火墙只放 80/443，8800 不对外）。

### Nginx

参考 [`deploy/nginx.conf.example`](deploy/nginx.conf.example)（含 certbot 命令）：

```nginx
upstream llm_relay {
    server 127.0.0.1:8800;
    keepalive 32;
}

server {
    listen 443 ssl;
    http2 on;
    server_name relay.example.com;

    ssl_certificate     /etc/letsencrypt/live/relay.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/relay.example.com/privkey.pem;

    client_max_body_size 50m;

    location / {
        proxy_pass http://llm_relay;
        proxy_http_version 1.1;
        proxy_set_header Connection "";

        # ---- 流式转发必须的四项 ----
        proxy_buffering off;
        proxy_cache off;
        chunked_transfer_encoding on;
        proxy_read_timeout 600s;

        proxy_set_header Host              $host;
        proxy_set_header X-Real-IP         $remote_addr;
        proxy_set_header X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

证书：`certbot --nginx -d relay.example.com` 签好后改上面两个路径。
部署：`cp deploy/nginx.conf.example /etc/nginx/conf.d/llm-relay.conf && nginx -t && systemctl reload nginx`。

**关键点**：

- `proxy_buffering off` 必须开，否则流式会被攒成一坨，首字延迟暴涨
- `proxy_read_timeout` 要大于上游 `timeout_s`（默认 300s），长回答才不会中途被反代掐断
- 网关侧 `X-Forwarded-For` 会还原成请求日志里的「来源 IP」列，可直接验证反代生效

### Caddy

参考 [`deploy/caddy.Caddyfile`](deploy/caddy.Caddyfile)（Let's Encrypt 证书全自动续签）：

```caddyfile
relay.example.com {
    reverse_proxy 127.0.0.1:8800 {
        flush_interval -1          # 流式的关键：每个 chunk 立刻下发，不攒包
        transport http {
            read_timeout 600s
            write_timeout 600s
        }
        header_up X-Real-IP {remote_host}
    }
}
```

域名 A 记录指过来、80/443 放行，然后 `caddy run --config deploy/caddy.Caddyfile`
（或放 `/etc/caddy/Caddyfile` 后 `systemctl reload caddy`）。

### 子目录部署（/llmrelay）

对外是 `https://域名/llmrelay/v1` 时，网关要认识前缀，三选一：

- `.env` 里 `RELAY_BASE_PATH=/llmrelay`（注意：设了环境变量，控制台里该项会被锁定，以环境变量为准）
- 控制台「设置 → 访问路径」填 `/llmrelay`
- 直接改 `config/settings.json` 的 `base_path`

保存后**立即生效，不用重启**。`location /llmrelay/` 下 `proxy_pass` 结尾带不带 `/` 两种写法都支持：

```nginx
location /llmrelay/ {
    proxy_pass http://127.0.0.1:8800/;      # 去掉结尾的 / 也行：两者都支持
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_buffering off;                    # SSE 必须
    proxy_read_timeout 600s;
}
```

前端用 `location.pathname` 自动推出前缀，带/不带前缀的地址都能访问，本地直连不受影响。

### 其他方式

- **Cloudflare Tunnel / 内网穿透**：`cloudflared tunnel --url http://127.0.0.1:8800`，
  域名和证书交给 Cloudflare，本地什么都不用改。
- **网关自跑 HTTPS（不挂反代）**：`RELAY_SSL_CERTFILE=.../fullchain.pem RELAY_SSL_KEYFILE=.../privkey.pem
  RELAY_HOST=0.0.0.0 RELAY_PORT=443 python run.py`，启动日志会把入口打印成 `https://…`。

> 公网暴露时务必：① 设 `RELAY_CORS_ORIGINS=https://你的域名` 收紧跨域；② 各账号的 API Key 别提交进客户端代码仓库。

## 三、WSL 部署（Windows 本地）

在 Windows 本机的 WSL 里跑 docker compose，Windows 侧直接 `localhost` 访问，
适合本地自用 / 测试，不用碰服务器：

```powershell
# Windows 上进入跑 llm-relay 的 WSL 发行版
wsl -d <发行版名> bash
cd /mnt/e/<你的项目目录>/llm-relay
```

> 建议用**专门的发行版**。如果一个发行版里已经跑着别的 docker 栈，
> `down`/`up` 会互相影响，换一个。

```bash
cp config/upstreams.example.json config/upstreams.json   # 填真 key
cp .env.example .env                                     # 本地用 RELAY_BASE_PATH 留空
mkdir -p data
docker compose build
docker compose up -d
docker compose ps                                        # 等 STATUS 变 healthy
```

Windows 侧访问（WSL2 自动端口转发，Windows 上永远是 `localhost`）：

```
http://localhost:8800/                 # 控制台
http://localhost:8800/v1/models        # API
curl http://localhost:8800/healthz     # 自检
```

各客户端（Claude Code / Dify / LobeChat …）的 base_url 填 `http://127.0.0.1:8800` 即可。

注意事项：

- 项目放在 `/mnt/e`（Windows 盘挂载）上，中转站磁盘 IO 很轻，直接跑没问题；
  想要更顺畅可以把项目挪到 WSL 内 `$HOME` 下的 ext4 路径
- `config/`、`data/` 是 bind mount，**关掉 WSL 再开，数据都在**；容器 `restart: unless-stopped`，
  重新进入发行版后如 docker 没起，先 `sudo service docker start`（或在 `/etc/wsl.conf` 配 init 自动起）
- 端口默认只绑 `127.0.0.1`，只有本机 Windows 能用；要给局域网用，把 compose 里
  `ports` 改成 `0.0.0.0:8800`（并留意防火墙）
- 本地自用可以不挂反代；需要 https / 域名的话把本章当服务器部署用，反代照第二章配

## 四、容量规划与部署注意

### 容量怎么估

中转站**不跑推理**，只做 HTTP 转发 + 协议转换 + SSE 转码，资源需求跟「多少 token」几乎无关，
只跟 **并发连接数** 和 **请求条数（统计库大小）** 有关。瓶颈永远在上游，不在网关。

实测基线（Docker，`python:3.12-slim`，`-m 128m` 限额，单进程）：

| 场景 | 结果 |
| --- | --- |
| 启动 | healthy，cgroup 峰值 56.7 MiB |
| 空载 | cgroup 43~45 MiB / RSS 约 60 MiB |
| **200 条并发的流式请求** | **200/200 成功，RSS 峰值 76.3 MiB，无 OOM** |
| 8 并发 × 2 MB 请求体 | 8/8 成功，无 OOM |
| 10 并发 × 6 MB 请求体 | **OOM 被杀**（≈ 20 个满 1M 上下文同时进，人为构造的极端值） |

内存经验公式：**内存 ≈ 60 MiB + 在途请求体总字节 × 1.8**。请求体 ≈ `tokens × 3 字节`
（满 256K 上下文 ≈ 1 MB，满 1M ≈ 3 MB）。128 MB 限额下：

| 同时在途的都是这种请求 | 128 MB 能同时扛 |
| --- | --- |
| 典型 Agent 请求（45 KB） | **750+ 条** |
| 满 256K 上下文（1 MB） | **~35 条** |
| 满 1M 上下文（3 MB） | ~11 条（唯一会紧的情况） |

推荐规格：

| 场景 | 账号数 | 日请求量 | 峰值并发流 | CPU | 内存 | 磁盘 | 带宽 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 个人 / 试用 | 1~3 | < 1 千 | < 5 | 1 核 | 256 MB | 2 GB | 5 Mbps |
| 小团队 | 5~20 | 1~5 万 | ~20 | 1~2 核 | 512 MB | 10 GB | 20 Mbps |
| 部门 | 50~200 | 10~50 万 | ~100 | 2 核 | 2 GB | 50 GB | 50 Mbps |
| 大组织 | 500+ | 100 万+ | 300+ | 4 核 | 4 GB | 200 GB | 100 Mbps+ |

其它资源算法：

- **磁盘** ≈ `请求条数 × 600 B`。100 万请求约 600 MB；SQLite 删数据后要 `VACUUM` 才缩文件。
- **带宽**：出入口流量接近 1:1。日均流量可忽略，真正要看的是**峰值并发流的瞬时速率**：
  单条流 5~20 KB/s，按「峰值并发流 × 15 KB/s × 2（双向）」估。

### 部署注意

1. **必须单进程**（`workers=1`，compose 里已经是）。统计库是 SQLite 单写者，上游 key 的
   轮换位置和冷却状态存在进程内存里；开多 worker 会各算各的。要横向扩，得先换 Postgres + Redis。
   不要起多副本。
2. **反向代理必须关 buffering**（Nginx `proxy_buffering off`、Caddy `flush_interval -1`），
   否则流式会被攒成一坨，首字延迟暴涨。
3. **uvicorn 访问日志建议关掉**（`--no-access-log`）。量大了纯占磁盘，
   而中转站自己在 SQLite 里已经有更完整的记录。
4. **WAL 会长**：有长连接读者时 `relay.db-wal` 不 checkpoint，
   定期 `PRAGMA wal_checkpoint(TRUNCATE)` 或重启一次即可。
5. **超时**：上游默认 300 s，长回答的流式连接会一直占着；反代侧 `proxy_read_timeout` 要大于它。
6. **监控什么**：并发连接数、`/healthz` 延迟、`relay.db` 体积、上游 429 比例。
   token 统计本身就在库里，不用另接监控。
