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
    PIP_PREFER_BINARY=1 \
    PYTHONUNBUFFERED=1 \
    CMAKE_BUILD_PARALLEL_LEVEL=8

# --- Build dependencies + Python 3.12 (natif Ubuntu 24.04) ---
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3-venv python3-dev python3-pip \
        build-essential gcc ninja-build git && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    python3 -m venv /opt/venv && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

ENV PATH="/opt/venv/bin:$PATH"

# --- PyTorch stable (CUDA 12.8) ---
# Pin >=2.7.1,<3 : support CUDA 12.8, compatible ltx-core ~=2.7
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install "torch>=2.7.1,<3" torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu128

# --- Core Python tooling ---
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install packaging setuptools wheel

# --- ltx-core + ltx-pipelines depuis le repo Lightricks/LTX-2 ---
# Pin au commit 28c3c73 (2026-02-09) — inclut CPU fallback natif pour fuse_loras FP8
RUN git clone --filter=blob:none --quiet https://github.com/Lightricks/LTX-2.git /tmp/LTX-2 && \
    cd /tmp/LTX-2 && git checkout 28c3c73fe557666c3de176e1e50a5220152ccfca

# Installer ltx-core d'abord (dependance de ltx-pipelines)
RUN --mount=type=cache,target=/root/.cache/pip \
    cd /tmp/LTX-2/packages/ltx-core && pip install .

# Installer ltx-pipelines
RUN --mount=type=cache,target=/root/.cache/pip \
    cd /tmp/LTX-2/packages/ltx-pipelines && pip install .

# Cleanup repo clone
RUN rm -rf /tmp/LTX-2

# --- Runtime Python dependencies (runpod, requests, etc.) ---
COPY requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /tmp/requirements.txt

# ============================================================
# Stage 2 : runtime (pas de compilateur, pas de headers)
# ============================================================
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    RUNPOD_INIT_TIMEOUT=600

# --- Runtime dependencies only (Python 3.12 natif Ubuntu 24.04) ---
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update && \
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
COPY src/bake_base_checkpoint.py /app/bake_base_checkpoint.py
COPY src/audit_volume.py /app/audit_volume.py

WORKDIR /app

EXPOSE 8888

HEALTHCHECK NONE

ENTRYPOINT ["tini", "-s", "--"]
CMD ["/start.sh"]
