# RunPod worker-comfyui with LTX-2 custom nodes (rebuilt 2026-02-15)
# dev tag = ComfyUI latest (with LTX-2 core support)
FROM runpod/worker-comfyui:dev

# Install custom nodes
RUN comfy node install comfyui-videohelpersuite@1.7.9 && \
    comfy node install ComfyUI_essentials && \
    comfy node install https://github.com/Lightricks/ComfyUI-LTXVideo

# Install wget for parallel downloads
RUN apt-get update && apt-get install -y --no-install-recommends wget && \
    rm -rf /var/lib/apt/lists/*

# Create model directories
RUN mkdir -p /comfyui/models/{checkpoints,clip,loras,latent_upscale_models}

# Download checkpoint + text encoder (biggest files first)
RUN cd /comfyui/models && \
    wget -q --show-progress -O checkpoints/ltx-2-19b-dev-fp8.safetensors \
      https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-19b-dev-fp8.safetensors & \
    wget -q --show-progress -O clip/gemma_3_12B_it_fp8_scaled.safetensors \
      https://huggingface.co/Comfy-Org/ltx-2/resolve/main/split_files/text_encoders/gemma_3_12B_it_fp8_scaled.safetensors & \
    wget -q --show-progress -O loras/ltx-2-19b-distilled-lora-384.safetensors \
      https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-19b-distilled-lora-384.safetensors & \
    wget -q --show-progress -O loras/LTX-2-Image2Vid-Adapter.safetensors \
      https://huggingface.co/MachineDelusions/LTX-2_Image2Video_Adapter_LoRa/resolve/main/LTX-2-Image2Vid-Adapter.safetensors & \
    wait

# Download upscalers + IC LoRAs (parallel)
RUN cd /comfyui/models && \
    wget -q -O latent_upscale_models/ltx-2-spatial-upscaler-x2-1.0.safetensors \
      https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-spatial-upscaler-x2-1.0.safetensors & \
    wget -q -O latent_upscale_models/ltx-2-temporal-upscaler-x2-1.0.safetensors \
      https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-temporal-upscaler-x2-1.0.safetensors & \
    wget -q -O loras/ltx-2-19b-ic-lora-union-ref0.5.safetensors \
      https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-19b-ic-lora-union-ref0.5.safetensors & \
    wget -q -O loras/ltx-2-19b-ic-lora-canny-control.safetensors \
      https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-19b-ic-lora-canny-control.safetensors & \
    wget -q -O loras/ltx-2-19b-ic-lora-depth-control.safetensors \
      https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-19b-ic-lora-depth-control.safetensors & \
    wget -q -O loras/ltx-2-19b-ic-lora-detailer.safetensors \
      https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-19b-ic-lora-detailer.safetensors & \
    wget -q -O loras/ltx-2-19b-ic-lora-pose-control.safetensors \
      https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-19b-ic-lora-pose-control.safetensors & \
    wait

# Download camera LoRAs (parallel)
RUN cd /comfyui/models/loras && \
    wget -q -O ltx-2-19b-lora-camera-control-dolly-in.safetensors \
      https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-19b-lora-camera-control-dolly-in.safetensors & \
    wget -q -O ltx-2-19b-lora-camera-control-dolly-out.safetensors \
      https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-19b-lora-camera-control-dolly-out.safetensors & \
    wget -q -O ltx-2-19b-lora-camera-control-dolly-left.safetensors \
      https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-19b-lora-camera-control-dolly-left.safetensors & \
    wget -q -O ltx-2-19b-lora-camera-control-dolly-right.safetensors \
      https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-19b-lora-camera-control-dolly-right.safetensors & \
    wget -q -O ltx-2-19b-lora-camera-control-jib-down.safetensors \
      https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-19b-lora-camera-control-jib-down.safetensors & \
    wget -q -O ltx-2-19b-lora-camera-control-jib-up.safetensors \
      https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-19b-lora-camera-control-jib-up.safetensors & \
    wget -q -O ltx-2-19b-lora-camera-control-static.safetensors \
      https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-19b-lora-camera-control-static.safetensors & \
    wait
