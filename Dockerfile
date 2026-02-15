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
        google-perftools && \
    ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip && \
    python3.11 -m venv /opt/venv && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

ENV PATH="/opt/venv/bin:$PATH"

# --- PyTorch nightly (CUDA 12.8) ---
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --pre torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/nightly/cu128

# --- Core Python tooling ---
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install packaging setuptools wheel

# --- ComfyUI + utilities ---
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install comfy-cli jupyterlab opencv-python && \
    yes | comfy --workspace /ComfyUI install

# --- Custom nodes for LTX-2 I2V ---
RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/Lightricks/ComfyUI-LTXVideo.git && \
    (pip install -r ComfyUI-LTXVideo/requirements.txt 2>/dev/null || true)

RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git && \
    (pip install -r ComfyUI-VideoHelperSuite/requirements.txt 2>/dev/null || true)

RUN cd /ComfyUI/custom_nodes && \
    git clone https://github.com/cubiq/ComfyUI_essentials.git && \
    (pip install -r ComfyUI_essentials/requirements.txt 2>/dev/null || true)

# --- RunPod Serverless handler ---
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install runpod websocket-client boto3

RUN git clone --depth 1 https://github.com/runpod-workers/worker-comfyui.git /worker-comfyui

# --- Workflows (copies dans l'image, deployes au runtime) ---
RUN mkdir -p /workflows
COPY workflows/*.json /workflows/

# --- Extra model paths template ---
COPY src/extra_model_paths.yaml /extra_model_paths.yaml

# --- Startup script ---
COPY src/start.sh /start.sh
RUN chmod +x /start.sh

EXPOSE 8188 8888

CMD ["/start.sh"]
