#!/bin/bash

# ==========================================
# ENVOI WORKFLOW + IMAGE vers RunPod
# Usage: ./send-workflow.sh <image> [workflow]
# ==========================================

API_KEY="${RUNPOD_API_KEY:?RUNPOD_API_KEY non definie}"
ENDPOINT_ID="${RUNPOD_ENDPOINT_ID:?RUNPOD_ENDPOINT_ID non definie}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

IMAGE_FILE="${1:?Usage: $0 <image.png> [workflow.json]}"
WORKFLOW_FILE="${2:-${SCRIPT_DIR}/../workflows/medusa_i2v_v5_fast_api.json}"

if [ ! -f "$IMAGE_FILE" ]; then
  echo "Erreur: image introuvable: $IMAGE_FILE"
  exit 1
fi

echo "Envoi du workflow : $WORKFLOW_FILE"
echo "Image input      : $IMAGE_FILE"
echo ""

WORKFLOW_JSON=$(cat "$WORKFLOW_FILE")
IMAGE_BASE64=$(base64 -w 0 "$IMAGE_FILE")

# Le handler worker-comfyui upload les images via /upload/image
# avant d'executer le workflow. "name" doit correspondre au noeud LoadImage.
PAYLOAD=$(jq -n \
  --argjson workflow "$WORKFLOW_JSON" \
  --arg img_b64 "$IMAGE_BASE64" \
  '{
    input: {
      workflow: $workflow,
      images: [{
        name: "input.png",
        image: $img_b64
      }]
    }
  }')

curl -s -X POST "https://api.runpod.ai/v2/${ENDPOINT_ID}/runsync" \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer ${API_KEY}" \
  -d "$PAYLOAD" | jq .
