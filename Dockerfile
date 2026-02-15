# RunPod worker-comfyui with LTX-2 custom nodes (rebuilt 2026-02-15)
# dev tag = ComfyUI latest (with LTX-2 core support)
FROM runpod/worker-comfyui:dev

# Install custom nodes
RUN comfy node install comfyui-videohelpersuite@1.7.9 && \
    comfy node install ComfyUI_essentials && \
    comfy node install https://github.com/Lightricks/ComfyUI-LTXVideo

# Checkpoint
RUN comfy model download \
    --url https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-19b-dev-fp8.safetensors \
    --relative-path models/checkpoints --filename ltx-2-19b-dev-fp8.safetensors

# Text encoder
RUN comfy model download \
    --url https://huggingface.co/Comfy-Org/ltx-2/resolve/main/split_files/text_encoders/gemma_3_12B_it_fp8_scaled.safetensors \
    --relative-path models/clip --filename gemma_3_12B_it_fp8_scaled.safetensors

# Upscalers
RUN comfy model download \
    --url https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-spatial-upscaler-x2-1.0.safetensors \
    --relative-path models/latent_upscale_models --filename ltx-2-spatial-upscaler-x2-1.0.safetensors
RUN comfy model download \
    --url https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-temporal-upscaler-x2-1.0.safetensors \
    --relative-path models/latent_upscale_models --filename ltx-2-temporal-upscaler-x2-1.0.safetensors

# Core LoRAs
RUN comfy model download \
    --url https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-19b-distilled-lora-384.safetensors \
    --relative-path models/loras --filename ltx-2-19b-distilled-lora-384.safetensors
RUN comfy model download \
    --url https://huggingface.co/MachineDelusions/LTX-2_Image2Video_Adapter_LoRa/resolve/main/LTX-2-Image2Vid-Adapter.safetensors \
    --relative-path models/loras --filename LTX-2-Image2Vid-Adapter.safetensors

# IC LoRAs
RUN comfy model download \
    --url https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-19b-ic-lora-union-ref0.5.safetensors \
    --relative-path models/loras --filename ltx-2-19b-ic-lora-union-ref0.5.safetensors
RUN comfy model download \
    --url https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-19b-ic-lora-canny-control.safetensors \
    --relative-path models/loras --filename ltx-2-19b-ic-lora-canny-control.safetensors
RUN comfy model download \
    --url https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-19b-ic-lora-depth-control.safetensors \
    --relative-path models/loras --filename ltx-2-19b-ic-lora-depth-control.safetensors
RUN comfy model download \
    --url https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-19b-ic-lora-detailer.safetensors \
    --relative-path models/loras --filename ltx-2-19b-ic-lora-detailer.safetensors
RUN comfy model download \
    --url https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-19b-ic-lora-pose-control.safetensors \
    --relative-path models/loras --filename ltx-2-19b-ic-lora-pose-control.safetensors

# Camera LoRAs
RUN comfy model download \
    --url https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-19b-lora-camera-control-dolly-in.safetensors \
    --relative-path models/loras --filename ltx-2-19b-lora-camera-control-dolly-in.safetensors
RUN comfy model download \
    --url https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-19b-lora-camera-control-dolly-out.safetensors \
    --relative-path models/loras --filename ltx-2-19b-lora-camera-control-dolly-out.safetensors
RUN comfy model download \
    --url https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-19b-lora-camera-control-dolly-left.safetensors \
    --relative-path models/loras --filename ltx-2-19b-lora-camera-control-dolly-left.safetensors
RUN comfy model download \
    --url https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-19b-lora-camera-control-dolly-right.safetensors \
    --relative-path models/loras --filename ltx-2-19b-lora-camera-control-dolly-right.safetensors
RUN comfy model download \
    --url https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-19b-lora-camera-control-jib-down.safetensors \
    --relative-path models/loras --filename ltx-2-19b-lora-camera-control-jib-down.safetensors
RUN comfy model download \
    --url https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-19b-lora-camera-control-jib-up.safetensors \
    --relative-path models/loras --filename ltx-2-19b-lora-camera-control-jib-up.safetensors
RUN comfy model download \
    --url https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-19b-lora-camera-control-static.safetensors \
    --relative-path models/loras --filename ltx-2-19b-lora-camera-control-static.safetensors
