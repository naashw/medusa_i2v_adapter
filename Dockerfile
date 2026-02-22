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
    UV_COMPILE_BYTECODE=1

# --- Build dependencies + Python 3.12 (natif Ubuntu 24.04) + uv ---
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3-venv python3-dev \
        build-essential gcc ninja-build git && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    python3 -m venv /opt/venv && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Installer uv — pin 0.9.30 : ltx-core requiert uv_build<0.10.0
COPY --from=ghcr.io/astral-sh/uv:0.9.30 /uv /usr/local/bin/uv

ENV PATH="/opt/venv/bin:$PATH" \
    VIRTUAL_ENV="/opt/venv"

# --- PyTorch stable (CUDA 12.8) ---
# Pin >=2.7.1,<3 : support CUDA 12.8, compatible ltx-core ~=2.7
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install "torch>=2.7.1,<3" torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu128 \
        --extra-index-url https://pypi.org/simple/

# --- ltx-core + ltx-pipelines + runtime deps en une seule resolution ---
# Pin au commit 28c3c73 (2026-02-09) — inclut CPU fallback natif pour fuse_loras FP8
# Resolution atomique : evite que requirements.txt ecrase ltx-core
COPY requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    git clone --filter=blob:none --quiet https://github.com/Lightricks/LTX-2.git /tmp/LTX-2 && \
    cd /tmp/LTX-2 && git checkout 28c3c73fe557666c3de176e1e50a5220152ccfca && \
    uv pip install \
        /tmp/LTX-2/packages/ltx-core \
        /tmp/LTX-2/packages/ltx-pipelines \
        -r /tmp/requirements.txt && \
    rm -rf /tmp/LTX-2

# Verification builder : ltx_core et ltx_pipelines importables
RUN /opt/venv/bin/python -c "import ltx_core; print('ltx_core OK:', ltx_core.__file__)" && \
    /opt/venv/bin/python -c "import ltx_pipelines; print('ltx_pipelines OK')" && \
    uv pip list | grep -i ltx

# ============================================================
# Stage 2 : runtime (pas de compilateur, pas de headers)
# ============================================================
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    RUNPOD_INIT_TIMEOUT=600 \
    HF_XET_HIGH_PERFORMANCE=1

# --- Runtime dependencies only (Python 3.12 natif Ubuntu 24.04) ---
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3-dev \
        curl ffmpeg \
        libgl1 libglib2.0-0 \
        google-perftools tini \
        gcc libc6-dev && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# --- Copy venv depuis builder ---
COPY --from=builder /opt/venv /opt/venv

# Verification runtime : ltx_core et ltx_pipelines importables apres COPY
RUN /opt/venv/bin/python -c "import ltx_core; print('ltx_core OK:', ltx_core.__file__)" && \
    /opt/venv/bin/python -c "import ltx_pipelines; print('ltx_pipelines OK')"

# --- Application ---
RUN mkdir -p /app
COPY src/start.sh /start.sh
RUN chmod +x /start.sh

COPY src/prompts.py /app/prompts.py
COPY src/pipeline.py /app/pipeline.py
COPY src/handler.py /app/handler.py
COPY src/warmup_embeddings.py /app/warmup_embeddings.py
COPY src/audit_volume.py /app/audit_volume.py

WORKDIR /app

EXPOSE 8888

HEALTHCHECK NONE

ENTRYPOINT ["tini", "-s", "--"]
CMD ["/start.sh"]
