#!/bin/bash

# ==========================================
# CHECK JOB STATUS - RUNPOD
# ==========================================

API_KEY="${RUNPOD_API_KEY:?RUNPOD_API_KEY non definie}"
ENDPOINT_ID="${RUNPOD_ENDPOINT_ID:?RUNPOD_ENDPOINT_ID non definie}"
JOB_ID="${1}"  # Passe le job ID en argument

if [ -z "$JOB_ID" ]; then
  echo "❌ Usage: ./check-status.sh JOB_ID"
  exit 1
fi

echo "🔍 Checking status for job: ${JOB_ID}"
echo ""

curl -s -H "Authorization: Bearer ${API_KEY}" \
  "https://api.runpod.ai/v2/${ENDPOINT_ID}/status/${JOB_ID}" | jq .

echo ""
echo "💡 Status:"
echo "   - IN_QUEUE: En attente"
echo "   - IN_PROGRESS: En cours de génération"
echo "   - COMPLETED: ✅ Terminé (check 'output' pour l'URL)"
echo "   - FAILED: ❌ Erreur"
