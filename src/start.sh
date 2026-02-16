#!/bin/bash
set -euo pipefail

# -----------------------------------------------
# Signal handling
# -----------------------------------------------
CHILD_PIDS=()

cleanup() {
    echo "[medusa] Arret en cours..."
    for pid in "${CHILD_PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait
    exit 0
}

trap cleanup SIGTERM SIGINT SIGQUIT

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
# RunPod Serverless monte le network volume sur /runpod-volume
# GPU Pods utilisent /workspace
if [ -d "/runpod-volume" ]; then
    WORKSPACE="${WORKSPACE:-/runpod-volume}"
else
    WORKSPACE="${WORKSPACE:-/workspace}"
fi
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
    sed "s|/workspace/models|${MODELS_DIR}|g" /extra_model_paths.yaml > "${COMFYUI_DIR}/extra_model_paths.yaml"
    echo "[medusa] extra_model_paths.yaml configure (depuis template)"
else
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
# 5. Fonction de telechargement
# -----------------------------------------------
download_model() {
    local url="$1"
    local dest_dir="$2"
    local min_size="${3:-1000000}"
    local filename
    filename=$(basename "$url")
    local filepath="${dest_dir}/${filename}"

    if [ -f "$filepath" ]; then
        local size
        size=$(stat -c%s "$filepath" 2>/dev/null || echo 0)
        if [ "$size" -gt "$min_size" ]; then
            echo "[medusa] Deja present: $filename ($(numfmt --to=iec "$size"))"
            return 0
        fi
        echo "[medusa] Corrompu (${size}B < min ${min_size}B), re-telechargement: $filename"
    fi

    echo "[medusa] Telechargement: $filename"
    aria2c -x 16 -s 16 -k 1M \
        -d "$dest_dir" -o "$filename" \
        "$url" \
        --console-log-level=error \
        --summary-interval=0 \
        --check-certificate=true \
        --file-allocation=none \
        --max-tries=3 \
        --retry-wait=5 \
        --timeout=600

    local final_size
    final_size=$(stat -c%s "$filepath" 2>/dev/null || echo 0)
    if [ "$final_size" -lt "$min_size" ]; then
        echo "[medusa] ERREUR: $filename trop petit (${final_size}B < min ${min_size}B)"
        return 1
    fi
}

# -----------------------------------------------
# 6. Telechargement des modeles (en parallele)
# -----------------------------------------------
echo "[medusa] Demarrage des telechargements..."
DOWNLOAD_PIDS=()

# --- Checkpoint (>10GB) ---
download_model \
    "https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-19b-dev-fp8.safetensors" \
    "${MODELS_DIR}/checkpoints" 10000000000 &
DOWNLOAD_PIDS+=($!)

# --- Text encoder (>6GB) ---
download_model \
    "https://huggingface.co/Comfy-Org/ltx-2/resolve/main/split_files/text_encoders/gemma_3_12B_it_fp8_scaled.safetensors" \
    "${MODELS_DIR}/text_encoders" 6000000000 &
DOWNLOAD_PIDS+=($!)

# --- Distilled LoRA (>100MB) ---
download_model \
    "https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-19b-distilled-lora-384.safetensors" \
    "${MODELS_DIR}/loras" 100000000 &
DOWNLOAD_PIDS+=($!)

# --- I2V Adapter (>100MB) ---
download_model \
    "https://huggingface.co/MachineDelusions/LTX-2_Image2Video_Adapter_LoRa/resolve/main/LTX-2-Image2Vid-Adapter.safetensors" \
    "${MODELS_DIR}/loras" 100000000 &
DOWNLOAD_PIDS+=($!)

# --- Spatial upscaler (>50MB) ---
download_model \
    "https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-spatial-upscaler-x2-1.0.safetensors" \
    "${MODELS_DIR}/latent_upscale_models" 50000000 &
DOWNLOAD_PIDS+=($!)

# --- Temporal upscaler (>50MB) ---
download_model \
    "https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-temporal-upscaler-x2-1.0.safetensors" \
    "${MODELS_DIR}/latent_upscale_models" 50000000 &
DOWNLOAD_PIDS+=($!)

# --- Camera LoRAs (>100MB each) ---
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
    download_model "$lora_url" "${MODELS_DIR}/loras" 100000000 &
    DOWNLOAD_PIDS+=($!)
done

# Attendre tous les telechargements et verifier les codes retour
echo "[medusa] Attente fin des telechargements..."
DOWNLOAD_FAILED=0
for pid in "${DOWNLOAD_PIDS[@]}"; do
    if ! wait "$pid"; then
        DOWNLOAD_FAILED=1
    fi
done

if [ "$DOWNLOAD_FAILED" -eq 1 ]; then
    echo "[medusa] ERREUR: Un ou plusieurs telechargements ont echoue"
    exit 1
fi
echo "[medusa] Tous les modeles sont prets."

# -----------------------------------------------
# 7. Demarrage (GPU Pod ou Serverless)
# -----------------------------------------------
if [ "${SERVERLESS:-}" = "true" ] || [ -n "${RUNPOD_ENDPOINT_ID:-}" ]; then
    # ===== MODE SERVERLESS =====
    echo "[medusa] Mode: SERVERLESS (RunPod API)"

    # Dossiers persistants sur le network volume
    OUTPUT_DIR="${WORKSPACE}/output"
    CACHE_DIR="${WORKSPACE}/cache"
    mkdir -p "$OUTPUT_DIR" "$CACHE_DIR"
    export OUTPUT_VOLUME_DIR="$OUTPUT_DIR"
    export CACHE_DIR="$CACHE_DIR"
    echo "[medusa] Output dir: $OUTPUT_DIR"
    echo "[medusa] Cache dir: $CACHE_DIR"

    # Sync embedding cache depuis le volume (persistant entre cold starts)
    EMBEDDING_CACHE_VOL="${WORKSPACE}/cache/embeddings"
    EMBEDDING_CACHE_LOCAL="/ComfyUI/cache/embeddings"
    if [ -d "$EMBEDDING_CACHE_VOL" ]; then
        PT_COUNT=$(find "$EMBEDDING_CACHE_VOL" -name "*.pt" 2>/dev/null | wc -l)
        if [ "$PT_COUNT" -gt 0 ]; then
            mkdir -p "$EMBEDDING_CACHE_LOCAL"
            cp "$EMBEDDING_CACHE_VOL"/*.pt "$EMBEDDING_CACHE_LOCAL/" 2>/dev/null
            echo "[medusa] Embeddings cache: $PT_COUNT fichier(s) copies depuis volume"
        fi
    fi

    # Desactiver ComfyUI-Manager network checks (economise ~2min au cold start)
    MANAGER_DIR="${COMFYUI_DIR}/user/__manager"
    mkdir -p "$MANAGER_DIR"
    cat > "${MANAGER_DIR}/config.ini" << EOF
[default]
network_mode = offline
EOF
    echo "[medusa] ComfyUI-Manager: mode offline (serverless)"

    cd "${COMFYUI_DIR}"
    python main.py \
        --listen 127.0.0.1 \
        --port 8188 \
        --disable-auto-launch \
        --disable-metadata \
        --disable-smart-memory \
        --reserve-vram 0 \
        --extra-model-paths-config "${COMFYUI_DIR}/extra_model_paths.yaml" &
    COMFYUI_PID=$!
    CHILD_PIDS+=($COMFYUI_PID)

    echo "[medusa] Demarrage du handler RunPod..."
    exec python /handler_wrapper.py

else
    # ===== MODE GPU POD =====
    echo "[medusa] Mode: GPU POD (interactif)"

    # JupyterLab avec token securise
    if [ -z "${JUPYTER_TOKEN:-}" ]; then
        JUPYTER_TOKEN=$(python -c "import secrets; print(secrets.token_hex(32))")
        echo "[medusa] JupyterLab token genere: ${JUPYTER_TOKEN}"
    fi

    jupyter lab \
        --ip=0.0.0.0 \
        --port=8888 \
        --no-browser \
        --allow-root \
        --ServerApp.token="${JUPYTER_TOKEN}" \
        --ServerApp.allow_origin='*' \
        --notebook-dir="${WORKSPACE}" &
    CHILD_PIDS+=($!)
    echo "[medusa] JupyterLab demarre sur port 8888"

    echo "[medusa] Demarrage ComfyUI sur port 8188..."
    cd "${COMFYUI_DIR}"
    exec python main.py \
        --listen 0.0.0.0 \
        --port 8188 \
        --extra-model-paths-config "${COMFYUI_DIR}/extra_model_paths.yaml"
fi
