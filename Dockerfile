# RunPod worker-comfyui with LTX-2 custom nodes (rebuilt 2026-02-15)
FROM runpod/worker-comfyui:5.7.1-base

# ComfyUI is already installed + configured in base image
# Just install the custom nodes for LTX-2 support

# Install custom nodes (ComfyUI already has LTX-2 in core since v0.6.0)
RUN comfy node install comfyui-videohelpersuite@1.7.9 && \
    comfy node install ComfyUI_essentials && \
    comfy node install https://github.com/Lightricks/ComfyUI-LTXVideo

# Download LTX-2 models in parallel for faster build
RUN mkdir -p /comfyui/models/{checkpoints,clip,loras,latent_upscale_models} && \
    cd /comfyui && \
    wget -q -O models/checkpoints/ltx-2-19b-dev-fp8.safetensors \
    https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-19b-dev-fp8.safetensors & \
    wget -q -O models/clip/gemma_3_12B_it_fp8_scaled.safetensors \
    https://huggingface.co/Comfy-Org/ltx-2/resolve/main/split_files/text_encoders/gemma_3_12B_it_fp8_scaled.safetensors & \
    wget -q -O models/loras/ltx-2-19b-distilled-lora-384.safetensors \
    https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-19b-distilled-lora-384.safetensors & \
    wget -q -O models/loras/LTX-2-Image2Vid-Adapter.safetensors \
    https://huggingface.co/MachineDelusions/LTX-2_Image2Video_Adapter_LoRa/resolve/main/LTX-2-Image2Vid-Adapter.safetensors & \
    wget -q -O models/loras/ltx-2-19b-lora-camera-control-dolly-in.safetensors \
    https://huggingface.co/Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-In/resolve/main/ltx-2-19b-lora-camera-control-dolly-in.safetensors & \
    wget -q -O models/latent_upscale_models/ltx-2-spatial-upscaler-x2-1.0.safetensors \
    https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-spatial-upscaler-x2-1.0.safetensors & \
    wait
