#!/bin/bash
set -e

echo "============================================"
echo "  Medusa I2V - ComfyUI + LTX-2 19B"
echo "============================================"

# -----------------------------------------------
# 1. tcmalloc (optimisation memoire)
# -----------------------------------------------
TCMALLOC="$(ldconfig -p | grep -Po "libtcmalloc.so.\d" | head -n 1)"
if [ -n "$TCMALLOC" ]; then
    export LD_PRELOAD="$TCMALLOC"
    echo "[medusa] tcmalloc charge: $TCMALLOC"
else
    echo "[medusa] tcmalloc non trouve, pas d'optimisation memoire"
fi

# -----------------------------------------------
# 2. Workspace / Network volume
# -----------------------------------------------
WORKSPACE="${WORKSPACE:-/workspace}"
COMFYUI_DIR="/ComfyUI"
MODELS_DIR="${WORKSPACE}/models"

echo "[medusa] Workspace: $WORKSPACE"
echo "[medusa] Models dir: $MODELS_DIR"

mkdir -p "${MODELS_DIR}/checkpoints"
mkdir -p "${MODELS_DIR}/text_encoders"
mkdir -p "${MODELS_DIR}/loras"
mkdir -p "${MODELS_DIR}/latent_upscale_models"

# -----------------------------------------------
# 3. extra_model_paths.yaml (ComfyUI -> workspace)
# -----------------------------------------------
if [ -f /extra_model_paths.yaml ]; then
    # Utiliser le template embarque, substituer le path workspace
    sed "s|/workspace/models|${MODELS_DIR}|g" /extra_model_paths.yaml > "${COMFYUI_DIR}/extra_model_paths.yaml"
    echo "[medusa] extra_model_paths.yaml configure (depuis template)"
else
    # Fallback : generation dynamique
    cat > "${COMFYUI_DIR}/extra_model_paths.yaml" << EOF
medusa:
    base_path: ${MODELS_DIR}
    checkpoints: checkpoints
    loras: loras
    text_encoders: text_encoders
    latent_upscale_models: latent_upscale_models
EOF
    echo "[medusa] extra_model_paths.yaml configure (genere dynamiquement)"
fi

# -----------------------------------------------
# 4. Copie des workflows
# -----------------------------------------------
WORKFLOW_DEST="${COMFYUI_DIR}/user/default/workflows"
mkdir -p "$WORKFLOW_DEST"
if [ -d /workflows ] && ls /workflows/*.json 1>/dev/null 2>&1; then
    cp -n /workflows/*.json "$WORKFLOW_DEST/" 2>/dev/null || true
    echo "[medusa] Workflows copies dans ComfyUI"
fi

# -----------------------------------------------
# 5. Fonction de telechargement
# -----------------------------------------------
download_model() {
    local url="$1"
    local dest_dir="$2"
    local filename
    filename=$(basename "$url")
    local filepath="${dest_dir}/${filename}"

    # Skip si fichier existe et > 1MB (pas corrompu)
    if [ -f "$filepath" ]; then
        local size
        size=$(stat -c%s "$filepath" 2>/dev/null || echo 0)
        if [ "$size" -gt 1000000 ]; then
            echo "[medusa] Deja present: $filename ($(numfmt --to=iec "$size"))"
            return 0
        fi
        echo "[medusa] Corrompu (${size}B), re-telechargement: $filename"
    fi

    echo "[medusa] Telechargement: $filename"
    aria2c -x 16 -s 16 -k 1M \
        -d "$dest_dir" -o "$filename" \
        "$url" \
        --console-log-level=error \
        --summary-interval=0 \
        --check-certificate=false
}

# -----------------------------------------------
# 6. Telechargement des modeles (en parallele)
# -----------------------------------------------
echo "[medusa] Demarrage des telechargements..."

# --- Checkpoint ---
download_model \
    "https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-19b-dev-fp8.safetensors" \
    "${MODELS_DIR}/checkpoints" &

# --- Text encoder ---
download_model \
    "https://huggingface.co/Comfy-Org/ltx-2/resolve/main/split_files/text_encoders/gemma_3_12B_it_fp8_scaled.safetensors" \
    "${MODELS_DIR}/text_encoders" &

# --- Distilled LoRA ---
download_model \
    "https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-19b-distilled-lora-384.safetensors" \
    "${MODELS_DIR}/loras" &

# --- I2V Adapter ---
download_model \
    "https://huggingface.co/MachineDelusions/LTX-2_Image2Video_Adapter_LoRa/resolve/main/LTX-2-Image2Vid-Adapter.safetensors" \
    "${MODELS_DIR}/loras" &

# --- Spatial upscaler ---
download_model \
    "https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-spatial-upscaler-x2-1.0.safetensors" \
    "${MODELS_DIR}/latent_upscale_models" &

# --- Temporal upscaler ---
download_model \
    "https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-temporal-upscaler-x2-1.0.safetensors" \
    "${MODELS_DIR}/latent_upscale_models" &

# --- Camera LoRAs ---
CAMERA_LORAS=(
    "https://huggingface.co/Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-In/resolve/main/ltx-2-19b-lora-camera-control-dolly-in.safetensors"
    "https://huggingface.co/Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Out/resolve/main/ltx-2-19b-lora-camera-control-dolly-out.safetensors"
    "https://huggingface.co/Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Left/resolve/main/ltx-2-19b-lora-camera-control-dolly-left.safetensors"
    "https://huggingface.co/Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Right/resolve/main/ltx-2-19b-lora-camera-control-dolly-right.safetensors"
    "https://huggingface.co/Lightricks/LTX-2-19b-LoRA-Camera-Control-Jib-Down/resolve/main/ltx-2-19b-lora-camera-control-jib-down.safetensors"
    "https://huggingface.co/Lightricks/LTX-2-19b-LoRA-Camera-Control-Jib-Up/resolve/main/ltx-2-19b-lora-camera-control-jib-up.safetensors"
    "https://huggingface.co/Lightricks/LTX-2-19b-LoRA-Camera-Control-Static/resolve/main/ltx-2-19b-lora-camera-control-static.safetensors"
)

for lora_url in "${CAMERA_LORAS[@]}"; do
    download_model "$lora_url" "${MODELS_DIR}/loras" &
done

# Attendre tous les telechargements
echo "[medusa] Attente fin des telechargements..."
wait
echo "[medusa] Tous les modeles sont prets."

# -----------------------------------------------
# 7. Demarrage (GPU Pod ou Serverless)
# -----------------------------------------------
# Mode detecte par variable SERVERLESS=true
# ou automatiquement si RUNPOD_ENDPOINT_ID est present

if [ "${SERVERLESS}" = "true" ] || [ -n "${RUNPOD_ENDPOINT_ID}" ]; then
    # ===== MODE SERVERLESS =====
    echo "[medusa] Mode: SERVERLESS (RunPod API)"

    # Demarrer ComfyUI en background
    cd "${COMFYUI_DIR}"
    python main.py \
        --listen 127.0.0.1 \
        --port 8188 \
        --extra-model-paths-config "${COMFYUI_DIR}/extra_model_paths.yaml" &
    COMFYUI_PID=$!

    # Attendre que ComfyUI soit pret
    echo "[medusa] Attente demarrage ComfyUI..."
    MAX_RETRIES=60
    RETRY=0
    while [ $RETRY -lt $MAX_RETRIES ]; do
        if curl -s http://127.0.0.1:8188/system_stats > /dev/null 2>&1; then
            echo "[medusa] ComfyUI pret (apres ${RETRY}s)"
            break
        fi
        RETRY=$((RETRY + 1))
        sleep 1
    done

    if [ $RETRY -eq $MAX_RETRIES ]; then
        echo "[medusa] ERREUR: ComfyUI n'a pas demarre apres ${MAX_RETRIES}s"
        exit 1
    fi

    # Lancer le handler RunPod
    echo "[medusa] Demarrage du handler RunPod..."
    cd /worker-comfyui
    exec python handler.py

else
    # ===== MODE GPU POD =====
    echo "[medusa] Mode: GPU POD (interactif)"

    # JupyterLab en background
    jupyter lab \
        --ip=0.0.0.0 \
        --port=8888 \
        --no-browser \
        --allow-root \
        --ServerApp.token='' \
        --ServerApp.allow_origin='*' \
        --notebook-dir="${WORKSPACE}" &
    echo "[medusa] JupyterLab demarre sur port 8888"

    # ComfyUI en foreground
    echo "[medusa] Demarrage ComfyUI sur port 8188..."
    cd "${COMFYUI_DIR}"
    exec python main.py \
        --listen 0.0.0.0 \
        --port 8188 \
        --extra-model-paths-config "${COMFYUI_DIR}/extra_model_paths.yaml"
fi
