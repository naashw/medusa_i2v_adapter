#!/bin/bash

# ==========================================
# ENVOI DIRECT DU WORKFLOW (sans Dockerfile)
# ==========================================

API_KEY="${RUNPOD_API_KEY:?RUNPOD_API_KEY non definie}"
ENDPOINT_ID="${RUNPOD_ENDPOINT_ID:?RUNPOD_ENDPOINT_ID non definie}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKFLOW_FILE="${1:-${SCRIPT_DIR}/../workflows/medusa_i2v_v5_fast_api.json}"

echo "🎬 Envoi du workflow : $WORKFLOW_FILE"
echo ""

# Lit le workflow et l'envoie
WORKFLOW_JSON=$(cat "$WORKFLOW_FILE")

curl -X POST "https://api.runpod.ai/v2/${ENDPOINT_ID}/run" \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer ${API_KEY}" \
  -d "{
    \"input\": {
      \"workflow\": $WORKFLOW_JSON
    }
  }"
