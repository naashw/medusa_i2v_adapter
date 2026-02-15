#!/bin/bash

# ==========================================
# TEST WORKFLOW V2 - SPATIAL UPSCALER
# ==========================================

API_KEY="${RUNPOD_API_KEY:?RUNPOD_API_KEY non definie}"
ENDPOINT_ID="${RUNPOD_ENDPOINT_ID:?RUNPOD_ENDPOINT_ID non definie}"

# Image de test
IMAGE_URL="https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=1024"

echo "🎬 Test workflow V2 - Spatial Upscaler x2"
echo ""

RESPONSE=$(curl -s -X POST "https://api.runpod.ai/v2/${ENDPOINT_ID}/run" \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer ${API_KEY}" \
  -d '{
    "input": {
      "workflow_name": "v2-spatial-upscaler.json",
      "inputs": {
        "10": {
          "image": "'"${IMAGE_URL}"'"
        },
        "3": {
          "text": "Smooth cinematic dolly-in movement, epic landscape"
        },
        "4": {
          "text": "blurry, low quality, distorted"
        }
      }
    }
  }')

echo "📥 Réponse:"
echo "$RESPONSE" | jq .
echo ""

JOB_ID=$(echo "$RESPONSE" | jq -r '.id // empty')
ERROR=$(echo "$RESPONSE" | jq -r '.error // empty')

if [ -n "$ERROR" ]; then
  echo "❌ Erreur: $ERROR"
  exit 1
fi

if [ -n "$JOB_ID" ]; then
  echo "✅ Job créé: $JOB_ID"
  echo "🔍 Check status: ./check-status.sh $JOB_ID"
  echo ""
  echo "📊 Pipeline V2:"
  echo "  1️⃣  Pass 1: Generation half-res (8 steps)"
  echo "  🔺 Spatial upscaler: 2x latent upscale"
  echo "  2️⃣  Pass 2: Refinement full-res (4 steps, denoise 0.4)"
  echo "  🎬 Output: Video full-res (~2x original)"
fi
