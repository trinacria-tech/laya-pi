# One image, two roles: Laya's own HTTP server, and the snake spectator app.
#
# The default PyPI `torch` for aarch64 is the CUDA build (+cuXXX), which assumes
# ARMv8.2-A server/Jetson parts and dies with SIGILL on a Cortex-A72. Pinning the
# CPU index here is what keeps that wheel from being selected on a Raspberry Pi.

# llama.cpp's llama-server backs the GGUF decision models (d1-omni, Julia-1) through its
# own /v1/systemone. Built here because the release binaries need a newer glibc than
# bookworm. Baseline armv8-a kernels (what the Pi 4's A72 can run; bookworm's gcc 12
# can't build the newer ALL_VARIANTS set), so the image isn't tied to the build CPU.
FROM debian:bookworm-slim AS llama
ARG LLAMA_CPP_TAG=b11552
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 --branch ${LLAMA_CPP_TAG} https://github.com/ggml-org/llama.cpp /src \
    && cmake -S /src -B /build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF \
        -DGGML_CPU_ARM_ARCH=armv8-a -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF \
        -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DCMAKE_INSTALL_RPATH='$ORIGIN' \
    && cmake --build /build --target llama-server -j"$(nproc)" \
    && mkdir /out && cp /build/bin/llama-server /build/bin/*.so /out/

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

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=llama /out/*.so /usr/local/lib/
COPY --from=llama /out/llama-server /usr/local/bin/llama-server
RUN ldconfig

COPY snakeweb /app/snakeweb
COPY showcase /app/showcase
COPY laya_single.py /app/laya_single.py
COPY laya_capped.py /app/laya_capped.py
COPY laya_onnx_serve.py /app/laya_onnx_serve.py

EXPOSE 8000 8001 8080

# Overridden per service in docker-compose. Default: stock laya-serve.
CMD ["laya-serve"]
