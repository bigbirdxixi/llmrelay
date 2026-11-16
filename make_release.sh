#!/usr/bin/env bash
# 打生产部署包：代码包（无密钥）+ 可选的状态包（账号/配置/用量）
set -euo pipefail

cd "$(dirname "$0")"
# 版本号从 app/config.py 的 VERSION 自动读（避免打包时与代码版本错位，v1.2.3 起）
VER="$(sed -n 's/^VERSION = "\(.*\)"/\1/p' app/config.py | head -1)"
[ -n "$VER" ] || { echo "读不到 app/config.py 的 VERSION"; exit 1; }
STAMP="$(date +%Y%m%d-%H%M)"
NAME="llm-relay-${VER}"
STAGE=".pkgbuild/${NAME}"

rm -rf .pkgbuild && mkdir -p "$STAGE" dist

# ---------- 代码 ----------
cp -r app "$STAGE/"
mkdir -p "$STAGE/config" "$STAGE/deploy/certs"
cp run.py requirements.txt requirements-dev.txt pytest.ini README.md DEPLOY.md \
   Dockerfile docker-compose.yaml .dockerignore .env.example "$STAGE/"
cp -r tests "$STAGE/"
cp deploy/nginx.conf.example deploy/caddy.Caddyfile deploy/nginx-docker.conf "$STAGE/deploy/"
cp config/upstreams.example.json "$STAGE/config/"

# 清掉不该进包的东西
find "$STAGE" -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
find "$STAGE" -name '*.pyc' -delete 2>/dev/null || true
rm -rf "$STAGE/.pytest_cache" "$STAGE/tests/__pycache__"

# IP2Region xdb 不进包（2026-11-15 用户拍板：服务器 data/ 自备，包保持纯代码）

# 证书目录留空（自签测试证书不进去，线上要放自己的）
cat > "$STAGE/deploy/certs/README.txt" <<'EOF'
把真实证书放这里（走 docker compose --profile with-nginx 时需要）：
  fullchain.pem
  privkey.pem
用宿主机 Nginx/Caddy 的话不需要这个目录。
EOF

tar -czf "dist/${NAME}-src-${STAMP}.tar.gz" -C .pkgbuild "$NAME"
echo "代码包: dist/${NAME}-src-${STAMP}.tar.gz"

# ---------- 状态包（可选：迁移账号 / 上游配置 / 用量） ----------
if [ "${1:-}" = "--with-state" ]; then
  if [ -x .venv/Scripts/python.exe ]; then PY=.venv/Scripts/python.exe; else PY=python; fi
  # 先把 WAL 落盘，保证拷出来的是完整库
  "$PY" -c "import sqlite3;from pathlib import Path;c=sqlite3.connect(Path('data/relay.db'));c.execute('PRAGMA wal_checkpoint(TRUNCATE)');c.close();print('WAL 已 checkpoint')"

  S2=".pkgbuild/state/$NAME/state"
  mkdir -p "$S2/config/tenants" "$S2/data"
  cp config/upstreams.json config/settings.json "$S2/config/" 2>/dev/null || true
  cp -r config/tenants/. "$S2/config/tenants/" 2>/dev/null || true
  cp data/relay.db data/master_key.txt "$S2/data/" 2>/dev/null || true
  cp README-STATE.txt "$S2/" 2>/dev/null || true
  tar -czf "dist/${NAME}-state-${STAMP}.tar.gz" -C ".pkgbuild/state" "$NAME"
  echo "状态包: dist/${NAME}-state-${STAMP}.tar.gz"
fi

rm -rf .pkgbuild
ls -lh dist/
