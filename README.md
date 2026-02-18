# Medusa I2V - ltx-pipelines + LTX-2 19B

Pipeline Image-to-Video avec effets camera dolly, inference directe via ltx-pipelines (sans ComfyUI).

## Structure

```
medusa_i2v_adapter/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── src/
│   ├── start.sh                # Telechargement modeles + lancement
│   ├── warmup_embeddings.py    # Warmup embeddings low-RAM (process isole)
│   ├── pipeline.py             # MedusaPipeline (inference ltx-pipelines)
│   └── handler.py              # Handler RunPod serverless
├── workflows/             # Reference ComfyUI (plus utilises en prod)
├── scripts/               # Utilitaires
├── docs/
└── test-data/
```

## Docker

### Build

```bash
DOCKER_BUILDKIT=1 docker build --platform linux/amd64 -t medusa-i2v .
```

### Run (GPU Pod)

```bash
docker run --gpus all -p 8888:8888 -v /workspace:/workspace medusa-i2v
```

### Run (Serverless)

```bash
docker run --gpus all -e SERVERLESS=true -v /workspace:/workspace medusa-i2v
```

### Docker Compose

```bash
docker compose up
```

## API

### Input

```json
{
  "input": {
    "image": "https://example.com/photo.jpg",
    "camera": "dolly-in",
    "seed": 12345,
    "num_frames": 25,
    "frame_rate": 24,
    "image_strength": 1.0
  }
}
```

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `image` | string | **requis** | URL https ou base64 |
| `camera` | string | `dolly-in` | dolly-in/out/left/right, jib-up/down, static |
| `seed` | int | random | Seed generation |
| `num_frames` | int | `25` | Nombre de frames (doit etre k*8+1) |
| `frame_rate` | float | `24` | FPS |
| `image_strength` | float | `1.0` | Force conditioning image |
| `prompt` | string | auto | Override prompt (sinon genere depuis camera) |
| `negative_prompt` | string | default | Override negative prompt |

### Output

```json
{
  "images": [
    {
      "filename": "medusa_i2v_job-123.mp4",
      "content_type": "video",
      "size_mb": 1.5,
      "volume_path": "/runpod-volume/output/job-123/medusa_i2v_job-123.mp4",
      "s3_key": "output/job-123/medusa_i2v_job-123.mp4"
    }
  ]
}
```

### Exemple curl (RunPod)

```bash
curl -X POST "https://api.runpod.ai/v2/${ENDPOINT_ID}/runsync" \
  -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "image": "https://example.com/photo.jpg",
      "camera": "dolly-in",
      "seed": 42
    }
  }'
```

## Modeles

Telecharges automatiquement au runtime sur le network volume :

- **LTX-2 19B** (FP8, ~10GB) — checkpoint principal
- **Gemma 3 12B** (BF16, ~24GB) — text encoder (format HuggingFace, CPU)
- **LoRA distilled** — acceleration inference (8 steps)
- **I2V Adapter** — conditioning image
- **Camera LoRAs** (x7) — controle camera (dolly, jib, static)
