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
echo "  Medusa I2V - ltx-pipelines + LTX-2.3 22B Distilled"
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


# -----------------------------------------------
# 2b. Migration volume LTX-2 → LTX-2.3 (idempotent)
# -----------------------------------------------
migrate_volume() {
    echo "[medusa] === Migration volume LTX-2 → LTX-2.3 ==="

    # Anciens fichiers a supprimer
    local old_files=(
        "models/checkpoints/ltx-2-19b-dev.safetensors"
        "models/checkpoints/ltx-2.3-22b-dev-fp8.safetensors"
        "models/loras/ltx-2-19b-distilled-lora-384.safetensors"
        "models/loras/LTX-2-Image2Vid-Adapter.safetensors"
        "models/loras/ltx-2.3-22b-distilled-lora-384.safetensors"
        "models/upscalers/ltx-2-spatial-upscaler-x2-1.0.safetensors"
    )

    # Detecter si une migration est necessaire (au moins un ancien fichier present)
    local needs_migration=false
    for f in "${old_files[@]}"; do
        if [[ -f "${WORKSPACE}/${f}" ]]; then
            needs_migration=true
            break
        fi
    done

    if [[ "$needs_migration" == "false" ]]; then
        echo "[medusa] Pas de migration necessaire (aucun ancien fichier detecte)"
        return
    fi

    echo "[medusa] Anciens fichiers detectes — migration en cours..."

    # Supprimer anciens caches (invalides avec nouveau modele)
    local cache_dirs=("cache/transformer" "cache/embeddings" "cache/dedup" "output")
    for d in "${cache_dirs[@]}"; do
        local target="${WORKSPACE}/${d}"
        if [[ -d "$target" ]]; then
            echo "[medusa] Suppression cache: $target"
            rm -rf "$target"
        fi
    done

    # Supprimer anciens fichiers
    for f in "${old_files[@]}"; do
        local target="${WORKSPACE}/${f}"
        if [[ -f "$target" ]]; then
            echo "[medusa] Suppression ancien: $target"
            rm -f "$target"
        fi
    done

    # Supprimer les 7 camera LoRAs
    local camera_loras=(
        "ltx-2-19b-lora-camera-control-dolly-in.safetensors"
        "ltx-2-19b-lora-camera-control-dolly-out.safetensors"
        "ltx-2-19b-lora-camera-control-dolly-left.safetensors"
        "ltx-2-19b-lora-camera-control-dolly-right.safetensors"
        "ltx-2-19b-lora-camera-control-jib-down.safetensors"
        "ltx-2-19b-lora-camera-control-jib-up.safetensors"
        "ltx-2-19b-lora-camera-control-static.safetensors"
    )
    for f in "${camera_loras[@]}"; do
        local target="${MODELS_DIR}/loras/${f}"
        if [[ -f "$target" ]]; then
            echo "[medusa] Suppression camera LoRA: $target"
            rm -f "$target"
        fi
    done

    # Supprimer cache HuggingFace residuel
    if [[ -d "${WORKSPACE}/.cache/huggingface" ]]; then
        echo "[medusa] Suppression cache HuggingFace residuel"
        rm -rf "${WORKSPACE}/.cache/huggingface"
    fi

    # Supprimer corbeille trash-put residuelle
    if [[ -d "${WORKSPACE}/.Trash" ]]; then
        echo "[medusa] Suppression corbeille residuelle"
        rm -rf "${WORKSPACE}/.Trash"
    fi

    # Log espace disque apres nettoyage
    local volume_bytes
    volume_bytes=$(du -sb "$WORKSPACE" 2>/dev/null | awk '{print $1}')
    local volume_gb=$(( volume_bytes / 1073741824 ))
    echo "[medusa] Volume apres nettoyage: ${volume_gb}GB"

    echo "[medusa] === Migration terminee ==="
}

migrate_volume

mkdir -p "${MODELS_DIR}/checkpoints"
mkdir -p "${MODELS_DIR}/text_encoders"
mkdir -p "${MODELS_DIR}/loras"
mkdir -p "${MODELS_DIR}/upscalers"
mkdir -p "${WORKSPACE}/cache/transformer"
mkdir -p "${WORKSPACE}/cache/triton"
mkdir -p "${WORKSPACE}/cache/inductor"
mkdir -p "${WORKSPACE}/output"

export MODELS_DIR="$MODELS_DIR"
export VOLUME_ROOT="$WORKSPACE"
export CACHE_DIR="${WORKSPACE}/cache"
export OUTPUT_VOLUME_DIR="${WORKSPACE}/output"
export TRITON_CACHE_DIR="${WORKSPACE}/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="${WORKSPACE}/cache/inductor"
export TORCHINDUCTOR_FX_GRAPH_CACHE=1

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
                trash-put "$filepath"
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

# --- Checkpoint distilled BF16 (~46GB) — le plus gros, echouer tot ---
download_model "Lightricks/LTX-2.3" "ltx-2.3-22b-distilled.safetensors" "${MODELS_DIR}/checkpoints"

# --- Spatial upscaler x2 (~1GB) ---
download_model "Lightricks/LTX-2.3" "ltx-2.3-spatial-upscaler-x2-1.0.safetensors" "${MODELS_DIR}/upscalers"

# --- Temporal upscaler x2 (~262MB) ---
download_model "Lightricks/LTX-2.3" "ltx-2.3-temporal-upscaler-x2-1.0.safetensors" "${MODELS_DIR}/upscalers"

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
# 4b. Audit volume (dry-run, log fichiers inutilises) — desactive (lent sur NFS)
# -----------------------------------------------
# echo "[medusa] Audit volume (dry-run)..."
# python /app/audit_volume.py --volume "$WORKSPACE" || echo "[medusa] Audit volume echoue (non bloquant)"

# -----------------------------------------------
# 5. Demarrage
# -----------------------------------------------
if [ "${SERVERLESS:-}" = "true" ] || [ -n "${RUNPOD_ENDPOINT_ID:-}" ]; then
    # ===== MODE SERVERLESS =====
    echo "[medusa] Mode: SERVERLESS (RunPod API)"

    echo "[medusa] Output dir: $OUTPUT_VOLUME_DIR"
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
