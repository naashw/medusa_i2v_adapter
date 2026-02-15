# Medusa I2V - ComfyUI + LTX-2 19B
# Image legere : les modeles sont telecharges au runtime sur le network volume
# Supporte 2 modes : GPU Pod (interactif) et RunPod Serverless (API)
#
# Multi-stage build : devel (compile) -> runtime (execute)
# Build:  DOCKER_BUILDKIT=1 docker build --platform linux/amd64 -t medusa-i2v .
#
# GPU Pod:     docker run --gpus all -p 8188:8188 -p 8888:8888 -v /workspace:/workspace medusa-i2v
# Serverless:  docker run --gpus all -e SERVERLESS=true -v /workspace:/workspace medusa-i2v

# ============================================================
# Stage 1 : builder (compile PyTorch, Q8-Kernels, extensions)
# ============================================================
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_PREFER_BINARY=1 \
    PYTHONUNBUFFERED=1 \
    CMAKE_BUILD_PARALLEL_LEVEL=8

# --- Build dependencies + Python 3.11 ---
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-dev python3-pip \
        build-essential gcc ninja-build git && \
    ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip && \
    python3.11 -m venv /opt/venv && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

ENV PATH="/opt/venv/bin:$PATH"

# --- PyTorch stable (CUDA 12.8) ---
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch \
        torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu128

# --- Core Python tooling (requis avant Q8-Kernels avec --no-build-isolation) ---
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install packaging setuptools wheel

# --- LTX-Video Q8 Kernels (FP8 optimise, requiert CUDA 12.8+) ---
# Le setup.py appelle torch.cuda.get_device_capability() au build,
# ce qui echoue sans GPU. On patche pour utiliser un fallback env var.
# TORCH_CUDA_ARCH_LIST cible: RTX 4090 / L40S (8.9)
RUN --mount=type=cache,target=/root/.cache/pip \
    git clone --filter=blob:none --quiet https://github.com/Lightricks/LTX-Video-Q8-Kernels.git /tmp/q8-kernels && \
    cd /tmp/q8-kernels && git submodule update --init --recursive -q && \
    python -c "t=open('setup.py').read(); open('setup.py','w').write(t.replace('major, minor = torch.cuda.get_device_capability(0)','try:\n        major, minor = torch.cuda.get_device_capability(0)\n    except RuntimeError:\n        import os; return os.environ.get(\"Q8_DEVICE_ARCH\", \"ada\")'))" && \
    TORCH_CUDA_ARCH_LIST="8.9" Q8_DEVICE_ARCH=ada \
    pip install --no-build-isolation . && \
    rm -rf /tmp/q8-kernels && \
    python -c "\
p=__import__('importlib').import_module('q8_kernels.integration').__path__[0]+'/patch_transformer.py'; \
t=open(p).read(); \
open(p,'w').write(t.replace( \
    'transformer, use_fp8_attention=False, transform_weights=True', \
    'transformer, use_fp8_attention=False, transform_weights=True, quantize_self_attn=True, quantize_cross_attn=True, quantize_ffn=True'))"

# --- ComfyUI + Python dependencies ---
COPY requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /tmp/requirements.txt && \
    yes | comfy --workspace /ComfyUI install

# --- Custom nodes for LTX-2 I2V (pinned commits, 2026-02-15) ---
RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/Lightricks/ComfyUI-LTXVideo.git && \
    cd ComfyUI-LTXVideo && git checkout 82bd963cdeb66d023bed8c99324a307020907ef8 && cd .. && \
    pip install -r ComfyUI-LTXVideo/requirements.txt && \
    rm -rf ComfyUI-LTXVideo/.git

RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git && \
    cd ComfyUI-VideoHelperSuite && git checkout 993082e4f2473bf4acaf06f51e33877a7eb38960 && cd .. && \
    pip install -r ComfyUI-VideoHelperSuite/requirements.txt && \
    rm -rf ComfyUI-VideoHelperSuite/.git

RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/cubiq/ComfyUI_essentials.git && \
    cd ComfyUI_essentials && git checkout 9d9f4bedfc9f0321c19faf71855e228c93bd0dc9 && cd .. && \
    pip install -r ComfyUI_essentials/requirements.txt && \
    rm -rf ComfyUI_essentials/.git

# --- RunPod Serverless handler (pinned commit) ---
# Le Dockerfile upstream aplatit src/ a la racine (ADD src/network_volume.py ./),
# on reproduit ce comportement pour que handler.py trouve ses imports
RUN git clone https://github.com/runpod-workers/worker-comfyui.git /worker-comfyui && \
    cd /worker-comfyui && git checkout 0e2bf226f9ee3d7b6725f61ffbee652b67b6d172 && \
    cp src/network_volume.py . && \
    rm -rf .git

# ============================================================
# Stage 2 : runtime (pas de compilateur, pas de headers)
# ============================================================
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    RUNPOD_INIT_TIMEOUT=600

# --- Runtime dependencies only ---
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.11 \
        curl ffmpeg aria2 \
        libgl1 libglib2.0-0 \
        google-perftools tini && \
    ln -sf /usr/bin/python3.11 /usr/bin/python && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# --- Copy depuis builder ---
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /ComfyUI /ComfyUI
COPY --from=builder /worker-comfyui /worker-comfyui

# --- Extra model paths template ---
COPY src/extra_model_paths.yaml /extra_model_paths.yaml

# --- Startup script ---
COPY src/start.sh /start.sh
RUN chmod +x /start.sh

# --- Handler wrapper (cleanup post-job) ---
COPY src/handler_wrapper.py /handler_wrapper.py

EXPOSE 8188 8888

HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD curl -sf http://localhost:8188/system_stats || exit 1

ENTRYPOINT ["tini", "-s", "--"]
CMD ["/start.sh"]
