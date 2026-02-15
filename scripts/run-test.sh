#!/bin/bash

# ==========================================
# TEST RUNPOD COMFYUI - MEDUSA I2V
# ==========================================

# 🔑 Configuration
API_KEY="${RUNPOD_API_KEY:?RUNPOD_API_KEY non definie}"
ENDPOINT_ID="${RUNPOD_ENDPOINT_ID:?RUNPOD_ENDPOINT_ID non definie}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🎬 Test Medusa I2V sur RunPod..."
echo ""

# Lance le job
RESPONSE=$(curl -s -X POST "https://api.runpod.ai/v2/${ENDPOINT_ID}/run" \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer ${API_KEY}" \
  -d @"${SCRIPT_DIR}/../docs/example-request.json")

echo "📥 Réponse brute:"
echo "$RESPONSE"
echo ""

# Parse la réponse
JOB_ID=$(echo "$RESPONSE" | jq -r '.id // empty')
ERROR=$(echo "$RESPONSE" | jq -r '.error // empty')

if [ -n "$ERROR" ]; then
  echo "❌ Erreur: $ERROR"
  exit 1
fi

if [ -n "$JOB_ID" ]; then
  echo "✅ Job créé avec succès !"
  echo "🆔 Job ID: $JOB_ID"
  echo ""
  echo "🔍 Check le status avec:"
  echo "  ./check-status.sh $JOB_ID"
else
  echo "⚠️  Pas de Job ID reçu (voir la réponse ci-dessus)"
fi
