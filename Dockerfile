# One image, two roles: Laya's own HTTP server, and the snake spectator app.
#
# The default PyPI `torch` for aarch64 is the CUDA build (+cuXXX), which assumes
# ARMv8.2-A server/Jetson parts and dies with SIGILL on a Cortex-A72. Pinning the
# CPU index here is what keeps that wheel from being selected on a Raspberry Pi.

FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models \
    HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /app

RUN pip install --index-url https://download.pytorch.org/whl/cpu torch

# `serve` pulls fastapi + uvicorn for the /v1/systemone wire protocol; onnxruntime
# backs the separate ONNX service (laya[onnx] would also drag in the onnx exporter).
RUN pip install "laya[serve]" onnxruntime

COPY snakeweb /app/snakeweb
COPY showcase /app/showcase
COPY laya_single.py /app/laya_single.py
COPY laya_capped.py /app/laya_capped.py
COPY laya_onnx_serve.py /app/laya_onnx_serve.py

EXPOSE 8000 8001 8080

# Overridden per service in docker-compose. Default: stock laya-serve.
CMD ["laya-serve"]
