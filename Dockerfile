# syntax=docker/dockerfile:1
# Medusa I2V - ltx-pipelines + LTX-2 19B
# Image legere : les modeles sont telecharges au runtime sur le network volume
# Supporte 2 modes : GPU Pod (interactif) et RunPod Serverless (API)
#
# Multi-stage build : devel (compile) -> runtime (execute)
# Build:  DOCKER_BUILDKIT=1 docker build --platform linux/amd64 -t medusa-i2v .
#
# GPU Pod:     docker run --gpus all -p 8888:8888 -v /workspace:/workspace medusa-i2v
# Serverless:  docker run --gpus all -e SERVERLESS=true -v /workspace:/workspace medusa-i2v

# ============================================================
# Stage 1 : builder (compile PyTorch, ltx-core, ltx-pipelines)
# ============================================================
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CMAKE_BUILD_PARALLEL_LEVEL=8 \
    UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1

# --- Build dependencies + Python 3.12 (natif Ubuntu 24.04) + uv ---
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3-venv python3-dev \
        build-essential gcc ninja-build git && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    python3 -m venv /opt/venv && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Installer uv (package manager ultra-rapide)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV PATH="/opt/venv/bin:$PATH" \
    VIRTUAL_ENV="/opt/venv"

# --- PyTorch stable (CUDA 12.8) ---
# Pin >=2.7.1,<3 : support CUDA 12.8, compatible ltx-core ~=2.7
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install "torch>=2.7.1,<3" torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu128

# --- ltx-core + ltx-pipelines depuis le repo Lightricks/LTX-2 ---
# Pin au commit 28c3c73 (2026-02-09) — inclut CPU fallback natif pour fuse_loras FP8
RUN --mount=type=cache,target=/root/.cache/uv \
    git clone --filter=blob:none --quiet https://github.com/Lightricks/LTX-2.git /tmp/LTX-2 && \
    cd /tmp/LTX-2 && git checkout 28c3c73fe557666c3de176e1e50a5220152ccfca && \
    cd /tmp/LTX-2/packages/ltx-core && uv pip install . && \
    cd /tmp/LTX-2/packages/ltx-pipelines && uv pip install . && \
    rm -rf /tmp/LTX-2

# --- Runtime Python dependencies (runpod, requests, etc.) ---
# Installe EN DERNIER pour que nos pins (transformers<5.0) aient le dernier mot
COPY requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install -r /tmp/requirements.txt

# ============================================================
# Stage 2 : runtime (pas de compilateur, pas de headers)
# ============================================================
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    RUNPOD_INIT_TIMEOUT=600

# --- Runtime dependencies only (Python 3.12 natif Ubuntu 24.04) ---
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3-dev \
        curl ffmpeg aria2 \
        libgl1 libglib2.0-0 \
        google-perftools tini \
        gcc libc6-dev && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# --- Copy venv depuis builder ---
COPY --from=builder /opt/venv /opt/venv

# --- Application ---
RUN mkdir -p /app
COPY src/start.sh /start.sh
RUN chmod +x /start.sh

COPY src/pipeline.py /app/pipeline.py
COPY src/handler.py /app/handler.py
COPY src/warmup_embeddings.py /app/warmup_embeddings.py
COPY src/audit_volume.py /app/audit_volume.py

WORKDIR /app

EXPOSE 8888

HEALTHCHECK NONE

ENTRYPOINT ["tini", "-s", "--"]
CMD ["/start.sh"]
