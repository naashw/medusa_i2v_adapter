# clean base image containing only comfyui, comfy-cli and comfyui-manager
FROM runpod/worker-comfyui:5.5.1-base

# install custom nodes into comfyui (rebuild 2026-02-14)
RUN comfy node install --exit-on-fail comfyui-videohelpersuite@1.7.9 --mode remote
RUN comfy node install --exit-on-fail ComfyUI_essentials --mode remote

# download models into comfyui
RUN comfy model download --url https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-19b-dev-fp8.safetensors --relative-path models/checkpoints --filename ltx-2-19b-dev-fp8.safetensors

RUN comfy model download --url https://huggingface.co/Comfy-Org/ltx-2/resolve/main/split_files/text_encoders/gemma_3_12B_it_fp8_scaled.safetensors --relative-path models/clip --filename gemma_3_12B_it_fp8_scaled.safetensors

RUN comfy model download --url https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-19b-distilled-lora-384.safetensors --relative-path models/loras --filename ltx-2-19b-distilled-lora-384.safetensors

RUN comfy model download --url https://huggingface.co/MachineDelusions/LTX-2_Image2Video_Adapter_LoRa/resolve/main/LTX-2-Image2Vid-Adapter.safetensors --relative-path models/loras --filename LTX-2-Image2Vid-Adapter.safetensors

RUN comfy model download --url https://huggingface.co/Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-In/resolve/main/ltx-2-19b-lora-camera-control-dolly-in.safetensors --relative-path models/loras --filename ltx-2-19b-lora-camera-control-dolly-in.safetensors

RUN comfy model download --url https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-spatial-upscaler-x2-1.0.safetensors --relative-path models/latent_upscale_models --filename ltx-2-spatial-upscaler-x2-1.0.safetensors

# Copy workflows API into image (pour simplifier les requêtes)
COPY medusa_i2v_adapter_v5_very_fast_upscale_cleaned_api.json /comfyui/workflows/v1-resize.json
COPY medusa_i2v_adapter_v2_spatial_upscaler_api.json /comfyui/workflows/v2-spatial-upscaler.json
