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

# Audit espace disque au demarrage
echo "[medusa] === Espace disque volume ==="
du -sh "$WORKSPACE"/* 2>/dev/null | sort -rh | head -20 || echo "[medusa] Volume vide ou inaccessible"
echo "[medusa] ==========================="

mkdir -p "${MODELS_DIR}/checkpoints"
mkdir -p "${MODELS_DIR}/text_encoders"
mkdir -p "${MODELS_DIR}/loras"
mkdir -p "${MODELS_DIR}/upscalers"
mkdir -p "${WORKSPACE}/cache/transformer"

# Exporter pour handler.py
export MODELS_DIR="$MODELS_DIR"
export VOLUME_ROOT="$WORKSPACE"

# -----------------------------------------------
# 3. Fonction de telechargement
# -----------------------------------------------
download_model() {
    local repo_id="$1"
    local filename="$2"
    local dest_dir="$3"
    local filepath="${dest_dir}/${filename}"

    if [ -f "$filepath" ]; then
        # Valider les fichiers safetensors (detecte les telechargements partiels)
        if [[ "$filename" == *.safetensors ]]; then
            if ! python -c "from safetensors import safe_open; f = safe_open('${filepath}', framework='pt'); del f" 2>/dev/null; then
                local file_size
                file_size=$(stat -c%s "$filepath" 2>/dev/null || echo "?")
                echo "[medusa] CORROMPU: $filename (${file_size} bytes) — suppression et re-telechargement"
                rm -f "$filepath"
            else
                echo "[medusa] Deja present (valide): $filename"
                return 0
            fi
        else
            echo "[medusa] Deja present: $filename"
            return 0
        fi
    fi

    local disk_avail mem_avail
    disk_avail=$(df -h "$dest_dir" | awk 'NR==2{print $4}')
    mem_avail=$(awk '/MemAvailable/{printf "%.0fM", $2/1024}' /proc/meminfo 2>/dev/null || echo "?")
    echo "[medusa] Telechargement: $filename (disque libre: ${disk_avail}, RAM libre: ${mem_avail})"

    python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='${repo_id}',
    filename='${filename}',
    local_dir='${dest_dir}',
)
"
    local dl_exit=$?

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
echo "[medusa] Demarrage des telechargements (sequentiel, hf_xet)..."

# --- Checkpoint (~38GB) ---
download_model "Lightricks/LTX-2" "ltx-2-19b-dev.safetensors" "${MODELS_DIR}/checkpoints"

# --- Distilled LoRA ---
download_model "Lightricks/LTX-2" "ltx-2-19b-distilled-lora-384.safetensors" "${MODELS_DIR}/loras"

# --- I2V Adapter ---
download_model "MachineDelusions/LTX-2_Image2Video_Adapter_LoRa" "LTX-2-Image2Vid-Adapter.safetensors" "${MODELS_DIR}/loras"

# --- Camera LoRAs ---
CAMERA_LORAS=(
    "Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-In|ltx-2-19b-lora-camera-control-dolly-in.safetensors"
    "Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Out|ltx-2-19b-lora-camera-control-dolly-out.safetensors"
    "Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Left|ltx-2-19b-lora-camera-control-dolly-left.safetensors"
    "Lightricks/LTX-2-19b-LoRA-Camera-Control-Dolly-Right|ltx-2-19b-lora-camera-control-dolly-right.safetensors"
    "Lightricks/LTX-2-19b-LoRA-Camera-Control-Jib-Down|ltx-2-19b-lora-camera-control-jib-down.safetensors"
    "Lightricks/LTX-2-19b-LoRA-Camera-Control-Jib-Up|ltx-2-19b-lora-camera-control-jib-up.safetensors"
    "Lightricks/LTX-2-19b-LoRA-Camera-Control-Static|ltx-2-19b-lora-camera-control-static.safetensors"
)

for entry in "${CAMERA_LORAS[@]}"; do
    repo_id="${entry%%|*}"
    filename="${entry##*|}"
    download_model "$repo_id" "$filename" "${MODELS_DIR}/loras"
done

# --- Spatial upscaler x2 (~1GB) ---
download_model "Lightricks/LTX-2" "ltx-2-spatial-upscaler-x2-1.0.safetensors" "${MODELS_DIR}/upscalers"

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
    LD_PRELOAD="" python /app/warmup_embeddings.py
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
