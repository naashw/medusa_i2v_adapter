#!/bin/bash
set -euo pipefail

# -----------------------------------------------
# Signal handling
# -----------------------------------------------
CHILD_PIDS=()

cleanup() {
    local sig="$1"
    local mem_info
    mem_info=$(awk '/MemTotal|MemAvailable/{printf "%s: %.0fM ", $1, $2/1024}' /proc/meminfo 2>/dev/null || echo "")
    echo "[medusa] Signal recu: $sig ($mem_info)"
    echo "[medusa] Arret en cours..."
    for pid in "${CHILD_PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait
    exit 0
}

trap 'cleanup SIGTERM' SIGTERM
trap 'cleanup SIGINT' SIGINT
trap 'cleanup SIGQUIT' SIGQUIT

echo "============================================"
echo "  Medusa I2V - ltx-pipelines + LTX-2 19B"
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
if [ -d "/runpod-volume" ]; then
    WORKSPACE="${WORKSPACE:-/runpod-volume}"
else
    WORKSPACE="${WORKSPACE:-/workspace}"
fi
MODELS_DIR="${WORKSPACE}/models"

echo "[medusa] Workspace: $WORKSPACE"
echo "[medusa] Models dir: $MODELS_DIR"

mkdir -p "${MODELS_DIR}/checkpoints"
mkdir -p "${MODELS_DIR}/text_encoders"
mkdir -p "${MODELS_DIR}/loras"

# Exporter pour handler.py
export MODELS_DIR="$MODELS_DIR"
export VOLUME_ROOT="$WORKSPACE"

# -----------------------------------------------
# 3. Fonction de telechargement
# -----------------------------------------------
download_model() {
    local url="$1"
    local dest_dir="$2"
    local filename
    filename=$(basename "$url")
    local filepath="${dest_dir}/${filename}"

    if [ -f "$filepath" ] && [ ! -f "${filepath}.aria2" ]; then
        echo "[medusa] Deja present: $filename"
        return 0
    elif [ -f "${filepath}.aria2" ]; then
        echo "[medusa] Reprise telechargement interrompu: $filename"
    fi

    # Log etat systeme avant telechargement
    local disk_avail mem_avail
    disk_avail=$(df -h "$dest_dir" | awk 'NR==2{print $4}')
    mem_avail=$(awk '/MemAvailable/{printf "%.0fM", $2/1024}' /proc/meminfo 2>/dev/null || echo "?")
    echo "[medusa] Telechargement: $filename (disque libre: ${disk_avail}, RAM libre: ${mem_avail})"

    local dl_exit
    aria2c -x 16 -s 16 -k 1M \
        -d "$dest_dir" -o "$filename" \
        "$url" \
        --continue=true \
        --console-log-level=warn \
        --summary-interval=0 \
        --check-certificate=true \
        --file-allocation=none \
        --max-tries=3 \
        --retry-wait=5 \
        --timeout=600
    dl_exit=$?

    if [ "$dl_exit" -ne 0 ]; then
        local disk_after mem_after
        disk_after=$(df -h "$dest_dir" | awk 'NR==2{print $4}')
        mem_after=$(awk '/MemAvailable/{printf "%.0fM", $2/1024}' /proc/meminfo 2>/dev/null || echo "?")
        echo "[medusa] ERREUR: $filename echoue (exit=$dl_exit, disque libre: ${disk_after}, RAM libre: ${mem_after})"
        return 1
    fi

    echo "[medusa] OK: $filename"
}

# -----------------------------------------------
# 4. Telechargement des modeles (sequentiel)
# -----------------------------------------------
echo "[medusa] Demarrage des telechargements (sequentiel)..."

# --- Checkpoint (>10GB) ---
download_model \
    "https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-19b-dev.safetensors" \
    "${MODELS_DIR}/checkpoints"

# --- Distilled LoRA (>100MB) ---
download_model \
    "https://huggingface.co/Lightricks/LTX-2/resolve/main/ltx-2-19b-distilled-lora-384.safetensors" \
    "${MODELS_DIR}/loras"

# --- I2V Adapter (>100MB) ---
download_model \
    "https://huggingface.co/MachineDelusions/LTX-2_Image2Video_Adapter_LoRa/resolve/main/LTX-2-Image2Vid-Adapter.safetensors" \
    "${MODELS_DIR}/loras"

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
    download_model "$lora_url" "${MODELS_DIR}/loras"
done

# --- Gemma 3 12B (format HuggingFace, ~24GB BF16) ---
GEMMA_DIR="${MODELS_DIR}/text_encoders/gemma-3-12b-it"
if [ -d "$GEMMA_DIR" ] && [ -f "$GEMMA_DIR/config.json" ]; then
    echo "[medusa] Deja present: gemma-3-12b-it/"
else
    echo "[medusa] Telechargement: gemma-3-12b-it (HuggingFace format, ~24GB)..."
    python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'google/gemma-3-12b-it',
    local_dir='$GEMMA_DIR',
    ignore_patterns=['*.gguf', '*.bin'],
)
"
fi

echo "[medusa] Tous les modeles sont prets."

# -----------------------------------------------
# 4b. Audit volume (dry-run, log fichiers inutilises)
# -----------------------------------------------
echo "[medusa] Audit volume (dry-run)..."
python /app/audit_volume.py --volume "$WORKSPACE" || echo "[medusa] Audit volume echoue (non bloquant)"

# -----------------------------------------------
# 5. Demarrage
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

    # Warmup embeddings dans un process isole (l'OS recupere 100% RAM a la fin)
    echo "[medusa] Warmup embeddings (process isole)..."
    CUDA_VISIBLE_DEVICES="" LD_PRELOAD="" python /app/warmup_embeddings.py
    echo "[medusa] Warmup termine, lancement handler..."

    exec env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python /app/handler.py

else
    # ===== MODE GPU POD =====
    echo "[medusa] Mode: GPU POD (interactif)"

    # JupyterLab avec token securise
    if command -v jupyter &>/dev/null; then
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
    fi

    echo "[medusa] GPU Pod pret. Pipeline disponible via Python."
    echo "[medusa] Pour lancer le handler manuellement : python /app/handler.py"

    # Garder le container en vie
    wait
fi
