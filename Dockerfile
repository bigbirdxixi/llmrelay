# syntax=docker/dockerfile:1
# LLM Relay —— 生产镜像（Python 3.12-slim，仅装运行期依赖）
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TZ=Asia/Shanghai

WORKDIR /app

# 先装依赖，代码改动不会让这一层失效
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && rm -rf /var/lib/apt/lists/*

COPY app ./app
COPY run.py ./

# config / data 通过 volume 挂进来（镜像里不带真实 key 和数据库）
VOLUME ["/app/config", "/app/data"]

EXPOSE 8800

# 容器内部探活。不带反代前缀也能用：网关同时接受 /healthz 与 <base>/healthz
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8800/healthz', timeout=5).status == 200 else 1)"

CMD ["python", "run.py"]
