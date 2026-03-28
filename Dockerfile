# syntax=docker/dockerfile:1
# Medusa I2V - ltx-pipelines + LTX-2.3 22B FP8
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
# --- Audit espace disque volume builder RunPod ---
FROM alpine:3.21 AS volume-audit
RUN echo "=== AUDIT VOLUME BUILDER ===" && \
    du -sh /runpod-volume/* 2>/dev/null | sort -rh | head -20 || echo "Volume non monte" && \
    echo "==========================="

FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CMAKE_BUILD_PARALLEL_LEVEL=8

# --- Build dependencies + Python 3.12 (natif Ubuntu 24.04) ---
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3-venv python3-dev python3-pip \
        build-essential gcc ninja-build git && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    python3 -m venv /opt/venv && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

ENV PATH="/opt/venv/bin:$PATH" \
    VIRTUAL_ENV="/opt/venv"

# --- PyTorch stable (CUDA 12.8) ---
# Pin >=2.9,<3 : support torch.compile dynamic
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir \
        "torch>=2.9,<3" torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu128 \
        --extra-index-url https://pypi.org/simple/

# --- ltx-core + ltx-pipelines depuis le repo Lightricks/LTX-2 ---
# Pin au commit 9e8a28e — LTX-2.3 support (FP8 cast, temporal upscaler, I2V natif)
RUN git clone --filter=blob:none --quiet https://github.com/Lightricks/LTX-2.git /tmp/LTX-2 && \
    cd /tmp/LTX-2 && git checkout 9e8a28e17ac4dd9e49695223d50753a1ebda36fe

# Installer ltx-core d'abord (dependance de ltx-pipelines)
RUN cd /tmp/LTX-2/packages/ltx-core && pip install --no-cache-dir .

# Installer ltx-pipelines
RUN cd /tmp/LTX-2/packages/ltx-pipelines && pip install --no-cache-dir .

# Cleanup repo clone
RUN rm -rf /tmp/LTX-2

# --- Depth Anything 3 (DA3) ---
RUN git clone --filter=blob:none --quiet https://github.com/ByteDance-Seed/depth-anything-3.git /tmp/depth-anything-3 && \
    cd /tmp/depth-anything-3 && pip install --no-cache-dir . && \
    rm -rf /tmp/depth-anything-3

# H100 sm_90
ENV TORCH_CUDA_ARCH_LIST="9.0"

# --- Runtime Python dependencies (runpod, requests, etc.) ---
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Verification builder : ltx_core et ltx_pipelines importables
RUN python -c "import ltx_core; print('ltx_core OK:', ltx_core.__file__)" && \
    python -c "import ltx_pipelines; print('ltx_pipelines OK')" && \
    python -c "import depth_anything_3; print('DA3 OK')" && \
    pip list | grep -i ltx

# ============================================================
# Stage 2 : runtime (pas de compilateur, pas de headers)
# ============================================================
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
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
        gcc g++ libc6-dev && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# --- Copy venv depuis builder ---
COPY --from=builder /opt/venv /opt/venv

# Verification runtime : ltx_core et ltx_pipelines importables apres COPY
RUN python -c "import ltx_core; print('ltx_core OK:', ltx_core.__file__)" && \
    python -c "import ltx_pipelines; print('ltx_pipelines OK')"

# --- Application ---
RUN mkdir -p /app
COPY src/start.sh /start.sh
RUN chmod +x /start.sh

COPY src/prompts.py /app/prompts.py
COPY src/video_encoder.py /app/video_encoder.py
COPY src/pipeline.py /app/pipeline.py
COPY src/handler.py /app/handler.py
COPY src/warmup_embeddings.py /app/warmup_embeddings.py
COPY src/audit_volume.py /app/audit_volume.py

# Hash du build (source + packages) pour versionner le cache Inductor
RUN pip freeze | md5sum | cut -c1-12 > /app/.build_hash

WORKDIR /app

EXPOSE 8888

HEALTHCHECK NONE

ENTRYPOINT ["tini", "-s", "--"]
CMD ["/start.sh"]
