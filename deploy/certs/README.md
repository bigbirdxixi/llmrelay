# 部署证书目录

把你要用的证书放在这里（文件名需与 `docker-compose.yaml` 中 nginx 配置一致）：

- `fullchain.pem` — 证书链（certbot 的 `/etc/letsencrypt/live/<域名>/fullchain.pem`）
- `privkey.pem` — 私钥（certbot 的 `/etc/letsencrypt/live/<域名>/privkey.pem`）

证书获取：

```bash
certbot --nginx -d relay.example.com
# 或
certbot certonly -d relay.example.com
```

> 本目录只保留 `README.md`，`*.pem / *.key / *.crt` 全部 gitignore——
> 每个部署环境的证书都不一样，不要提交任何环境的证书到仓库。
