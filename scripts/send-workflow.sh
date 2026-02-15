#!/bin/bash

# ==========================================
# ENVOI DIRECT DU WORKFLOW (sans Dockerfile)
# ==========================================

API_KEY="***REMOVED_RUNPOD_API_KEY***"
ENDPOINT_ID="***REMOVED_ENDPOINT_ID***"
WORKFLOW_FILE="medusa_i2v_adapter_v5_very_fast_upscale_cleaned.json"

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
