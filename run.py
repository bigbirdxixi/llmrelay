"""启动入口：python run.py"""
from __future__ import annotations

import uvicorn

from app import config


def main() -> None:
    line = "=" * 66
    scheme = "https" if config.SSL_CERTFILE else "http"
    base = f"{scheme}://{config.HOST}:{config.PORT}"
    print(line)
    print(f"  LLM Relay v{config.VERSION}  ·  token 中转站")
    print(line)
    print(f"  控制台        {base}/")
    print(f"  OpenAI 入口   {base}/v1/chat/completions")
    print(f"  Claude 入口   {base}/v1/messages")
    print(f"  上游配置      {config.UPSTREAMS_FILE}")
    print(f"  用量库        {config.DB_FILE}")
    if config.SSL_CERTFILE:
        print(f"  TLS 证书      {config.SSL_CERTFILE}")
    if config.FORWARDED_ALLOW_IPS:
        print(f"  信任反代 IP   {config.FORWARDED_ALLOW_IPS}")
    print(line)
    uvicorn.run(
        "app.main:app",
        host=config.HOST,
        port=config.PORT,
        log_level="info",
        # 放在 Nginx / Caddy 后面时，正确还原客户端 IP 与协议
        proxy_headers=True,
        forwarded_allow_ips=config.FORWARDED_ALLOW_IPS,
        # 直接对外 HTTPS（不挂反代）时设置这两个环境变量
        ssl_certfile=config.SSL_CERTFILE or None,
        ssl_keyfile=config.SSL_KEYFILE or None,
    )


if __name__ == "__main__":
    main()
