#!/bin/bash

# ==========================================
# TEST RUNPOD COMFYUI - MEDUSA I2V
# ==========================================

# 🔑 Configuration
API_KEY="***REMOVED_RUNPOD_API_KEY***"  # Set RUNPOD_API_KEY env var ou remplace ici
ENDPOINT_ID="***REMOVED_ENDPOINT_ID***"

echo "🎬 Test Medusa I2V sur RunPod..."
echo ""

# Lance le job
RESPONSE=$(curl -s -X POST "https://api.runpod.ai/v2/${ENDPOINT_ID}/run" \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer ${API_KEY}" \
  -d @example-request.json)

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
