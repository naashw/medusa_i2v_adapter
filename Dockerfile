# Medusa I2V - ComfyUI + LTX-2 19B
# Image legere : les modeles sont telecharges au runtime sur le network volume
# Supporte 2 modes : GPU Pod (interactif) et RunPod Serverless (API)
#
# Build:  DOCKER_BUILDKIT=1 docker build -t medusa-i2v .
#
# GPU Pod:     docker run --gpus all -p 8188:8188 -p 8888:8888 -v /workspace:/workspace medusa-i2v
# Serverless:  docker run --gpus all -e SERVERLESS=true -v /workspace:/workspace medusa-i2v

FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_PREFER_BINARY=1 \
    PYTHONUNBUFFERED=1 \
    CMAKE_BUILD_PARALLEL_LEVEL=8

# --- System dependencies ---
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-dev python3-pip \
        curl ffmpeg ninja-build git aria2 git-lfs wget \
        libgl1 libglib2.0-0 build-essential gcc \
        google-perftools tini && \
    ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip && \
    python3.11 -m venv /opt/venv && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

ENV PATH="/opt/venv/bin:$PATH"

# --- PyTorch stable (CUDA 12.8) ---
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu128

# --- Core Python tooling (requis avant Q8-Kernels avec --no-build-isolation) ---
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install packaging setuptools wheel

# --- LTX-Video Q8 Kernels (FP8 optimise, requiert CUDA 12.8+) ---
# Le setup.py appelle torch.cuda.get_device_capability() au build,
# ce qui echoue sans GPU. On patche pour utiliser un fallback env var.
# TORCH_CUDA_ARCH_LIST cible: A100(8.0), A40(8.6), L40S(8.9), H100(9.0)
RUN --mount=type=cache,target=/root/.cache/pip \
    git clone --filter=blob:none --quiet https://github.com/Lightricks/LTX-Video-Q8-Kernels.git /tmp/q8-kernels && \
    cd /tmp/q8-kernels && git submodule update --init --recursive -q && \
    python -c "
p = 'setup.py'
t = open(p).read()
old = 'major, minor = torch.cuda.get_device_capability(0)'
new = '''try:\n        major, minor = torch.cuda.get_device_capability(0)\n    except RuntimeError:\n        import os; return os.environ.get('Q8_DEVICE_ARCH', 'ada')'''
open(p, 'w').write(t.replace(old, new))
" && \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0" Q8_DEVICE_ARCH=ada \
    pip install --no-build-isolation . && \
    rm -rf /tmp/q8-kernels

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
RUN git clone https://github.com/runpod-workers/worker-comfyui.git /worker-comfyui && \
    cd /worker-comfyui && git checkout 0e2bf226f9ee3d7b6725f61ffbee652b67b6d172 && \
    rm -rf .git

# --- Workflows API (seuls les workflows API sont copies dans l'image) ---
RUN mkdir -p /workflows
COPY workflows/medusa_i2v_v5_fast_api.json /workflows/
COPY workflows/medusa_i2v_1pass_upscale_api.json /workflows/
COPY workflows/medusa_i2v_v2_spatial_api.json /workflows/
COPY workflows/medusa_i2v_v3_native_api.json /workflows/

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
