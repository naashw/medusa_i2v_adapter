# RunPod worker-comfyui base with LTX-2 support (rebuilt 2026-02-15)
FROM runpod/worker-comfyui:5.7.1-base

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl wget ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Force upgrade ComfyUI to latest version with LTX-2 core support
RUN pip install --no-cache-dir --upgrade comfy-org>=0.6.0 comfy-cli

# Wait for ComfyUI core to be ready, then reinstall custom nodes with LTX-2 support
RUN sleep 5 && \
    comfy-node-install comfyui-videohelpersuite@1.7.9 && \
    comfy-node-install ComfyUI_essentials && \
    comfy-node-install https://github.com/Lightricks/ComfyUI-LTXVideo

# Download LTX-2 models
RUN comfy model download \
    --url https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-19b-dev-fp8.safetensors \
    --relative-path models/checkpoints \
    --filename ltx-2-19b-dev-fp8.safetensors

RUN comfy model download \
    --url https://huggingface.co/Comfy-Org/ltx-2/resolve/main/split_files/text_encoders/gemma_3_12B_it_fp8_scaled.safetensors \
    --relative-path models/clip \
    --filename gemma_3_12B_it_fp8_scaled.safetensors

RUN comfy model download \
    --url https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-19b-distilled-lora-384.safetensors \
    --relative-path models/loras \
    --filename ltx-2-19b-distilled-lora-384.safetensors

RUN comfy model download \
    --url https://huggingface.co/MachineDelusions/LTX-2_Image2Video_Adapter_LoRa/resolve/main/LTX-2-Image2Vid-Adapter.safetensors \
    --relative-path models/loras \
    --filename LTX-2-Image2Vid-Adapter.safetensors

RUN comfy model download \
    --url https://huggingface.co/Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-In/resolve/main/ltx-2-19b-lora-camera-control-dolly-in.safetensors \
    --relative-path models/loras \
    --filename ltx-2-19b-lora-camera-control-dolly-in.safetensors

RUN comfy model download \
    --url https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-spatial-upscaler-x2-1.0.safetensors \
    --relative-path models/latent_upscale_models \
    --filename ltx-2-spatial-upscaler-x2-1.0.safetensors
