#!/usr/bin/env bash
# Setup script for RunPod GPU Pod (H100) — test environment
# Usage: ssh root@IP -p PORT -i KEY < scripts/test_pod.sh
#    or: ssh root@IP -p PORT -i KEY "bash -s" < scripts/test_pod.sh
set -euo pipefail

# RunPod: nproc retourne les cores de l'hote, pas du pod
# Utiliser RUNPOD_CPU_COUNT si dispo, sinon nproc, cap a 32
CPU_COUNT="${RUNPOD_CPU_COUNT:-$(nproc)}"
if [[ "$CPU_COUNT" -gt 32 ]]; then CPU_COUNT=32; fi
export MAX_JOBS="$CPU_COUNT"
export NINJA_MAX_JOBS="$CPU_COUNT"
export CMAKE_BUILD_PARALLEL_LEVEL="$CPU_COUNT"
echo "Build parallelism: $CPU_COUNT jobs"
export TORCH_CUDA_ARCH_LIST="9.0"

PIP="pip install --break-system-packages --no-cache-dir"
# Identifiants de clone GitLab injectes via variables d'env (a passer dans l'env SSH du pod).
# Ne JAMAIS hardcoder le token. Cf. scripts/.env.example
GITLAB_DEPLOY_USER="${GITLAB_DEPLOY_USER:?GITLAB_DEPLOY_USER non definie}"
GITLAB_REPO="${GITLAB_REPO:-medusa7293008/medusa_i2v_adapter}"
REPO_URL="https://${GITLAB_DEPLOY_USER}:${GITLAB_DEPLOY_TOKEN:?GITLAB_DEPLOY_TOKEN non definie}@gitlab.com/${GITLAB_REPO}.git"
LTX_COMMIT="9e8a28e17ac4dd9e49695223d50753a1ebda36fe"
APP_DIR="/root/medusa_i2v_adapter"
LTX_DIR="/tmp/LTX-2"

echo "=== [1/6] Symlink /runpod-volume -> /workspace ==="
if [ ! -e /runpod-volume ]; then
    ln -s /workspace /runpod-volume
    echo "OK: symlink created"
else
    echo "SKIP: /runpod-volume already exists"
fi

echo "=== [2/6] PyTorch >=2.9 (CUDA 12.8) ==="
$PIP "torch>=2.9" --index-url https://download.pytorch.org/whl/cu128
python3 -c "import torch; print(f'PyTorch {torch.__version__} CUDA {torch.version.cuda}')"

echo "=== [3/6] Clone LTX-2 (commit $LTX_COMMIT) ==="
if [ ! -d "$LTX_DIR" ]; then
    git clone --filter=blob:none --quiet https://github.com/Lightricks/LTX-2.git "$LTX_DIR"
fi
cd "$LTX_DIR" && git checkout "$LTX_COMMIT" 2>/dev/null

echo "=== [4/6] Install ltx-core + ltx-pipelines (--no-deps to keep PyTorch) ==="
cd "$LTX_DIR/packages/ltx-core" && $PIP --no-deps .
cd "$LTX_DIR/packages/ltx-pipelines" && $PIP --no-deps .
# Install their deps without torch
$PIP einops scipy

echo "=== [5/6] Clone medusa_i2v_adapter ==="
if [ ! -d "$APP_DIR" ]; then
    git clone "$REPO_URL" "$APP_DIR"
else
    cd "$APP_DIR" && git pull
fi
$PIP "huggingface-hub[hf-xet]>=0.28,<1.0" safetensors sentencepiece accelerate \
    "runpod>=1.8,<2.0" "requests>=2.31,<3.0" "boto3>=1.34,<2.0" \
    "transformers>=4.52,<5.0" "Pillow>=10.0"

echo "=== [6/6] Verification ==="
python3 -c "
import torch; print(f'torch {torch.__version__} CUDA {torch.version.cuda}')
import ltx_core; print('ltx_core OK')
import ltx_pipelines; print('ltx_pipelines OK')
import transformers; print(f'transformers {transformers.__version__}')
print(f'SDPA backends: cuDNN attention natif (H100 sm_90)')
"
echo ""
echo "=== READY ==="
echo "Volume: $(ls /runpod-volume/)"
echo "Code:   $APP_DIR/src/"
echo ""
echo "Pour lancer le pipeline:"
echo "  cd $APP_DIR && python3 src/handler.py"
