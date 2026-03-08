# Medusa I2V - ltx-pipelines + LTX-2 19B

Pipeline Image-to-Video avec effets camera dolly, inference directe via ltx-pipelines (sans ComfyUI).

## Stack Technique

| Composant | Version |
|-----------|---------|
| CUDA | 12.8.1 + cuDNN |
| Python | 3.12 (Ubuntu 24.04 natif) |
| PyTorch | >=2.7.1 (wheel cu128) |
| ltx-core + ltx-pipelines | Lightricks/LTX-2 commit `28c3c73` |
| transformers | >=4.52, <5.0 |
| huggingface-hub | >=0.28 (avec HF XET) |
| runpod | >=1.7, <2.0 |
| boto3 | >=1.34 (S3 OVH) |
| Docker | Multi-stage (devel builder -> runtime) |

## Structure

```
medusa_i2v_adapter/
├── Dockerfile              # Multi-stage (cuda:12.8.1-devel -> runtime)
├── docker-compose.yml      # Lancement local avec GPU
├── requirements.txt        # Dependencies Python
├── src/
│   ├── start.sh            # Download modeles + validation + lancement
│   ├── warmup_embeddings.py # Warmup embeddings low-RAM (process isole)
│   ├── pipeline.py         # MedusaPipeline (inference ltx-pipelines)
│   ├── handler.py          # Handler RunPod serverless
│   ├── prompts.py          # Prompts camera + negative
│   └── audit_volume.py     # Audit fichiers inutiles sur volume
├── workflows/              # Reference ComfyUI (plus utilises en prod)
├── scripts/                # Utilitaires (test, envoi, conversion)
├── docs/                   # Documentation et exemples
└── test-data/              # Images de test
```

## Modeles

Telecharges automatiquement au runtime sur le network volume via HF XET.

| Modele | Taille | VRAM | Role |
|--------|--------|------|------|
| LTX-2 19B (BF16) | ~38GB | ~35GB | Transformer principal (velocity model) |
| Gemma 3 12B IT (BF16) | ~24GB | CPU only | Text encoder (warmup, puis libere) |
| Distilled LoRA (strength 0.7) | ~50MB | fuse dans transformer | Acceleration inference (8 steps) |
| I2V Adapter (strength 0.8) | ~100MB | fuse dans transformer | Conditioning image |
| Camera LoRAs x7 (strength 1.0) | ~100MB chaque | fuse/unfuse dynamique | Mouvement camera |
| Spatial Upscaler x2 | ~1GB | ~1GB | Upscale latent (pipeline 2-stage) |
| Video Encoder | inclus checkpoint | ~1GB | Image -> latent |
| Video Decoder | inclus checkpoint | ~2GB | Latent -> pixels |

**Budget VRAM total : ~50-55GB sur H100 80GB (marge ~25-30GB)**

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

## Flow de Demarrage

```
start.sh
  │
  ├─ 1. Signal handlers (SIGTERM/SIGINT/SIGQUIT)
  ├─ 2. tcmalloc (LD_PRELOAD) pour optimisation memoire
  ├─ 3. Creation arborescence /runpod-volume/models/
  ├─ 4. Download modeles (hf_xet) + validation safetensors via safe_open()
  ├─ 5. Audit volume (dry-run)
  └─ 6. Mode detection:
       ├── SERVERLESS → warmup embeddings + handler.py
       └── GPU POD → JupyterLab :8888
```

## Flow d'Init Pipeline

Execute une seule fois au demarrage, AVANT `runpod.serverless.start()` (eager init).

```
MedusaPipeline(models_dir)
  │
  ├─ 1. warmup_embeddings()
  │     └─ Cache .pt existe → charger
  │     └─ Sinon → Gemma 3 12B CPU (~35GB RAM peak)
  │        → encode 7 prompts camera + 1 negative
  │        → sauvegarder cache → liberer Gemma
  │
  ├─ 2. get_transformer(dolly-in)
  │     └─ Cache transformer pre-fusionne (fingerprint OK) → charger
  │     └─ Sinon → ModelLedger(checkpoint + distilled + I2V) → sauvegarder cache
  │     └─ torch.compile(mode="reduce-overhead")
  │     └─ Fuse camera dolly-in delta in-place
  │
  ├─ 3. load_video_encoder()     → VRAM (~1GB, persistent)
  ├─ 4. load_video_decoder()     → VRAM (~2GB, persistent)
  └─ 5. load_spatial_upsampler() → VRAM (~1GB, persistent)
```

## Flow d'Inference (par job)

```
handler(job)
  │
  ├─ 1. Hash input (SHA256) → check dedup cache → return si hit
  ├─ 2. Parse input (image, camera, seed, resolution...)
  ├─ 3. Download images en parallele (ThreadPoolExecutor)
  ├─ 4. Calcul resolution cible (aspect ratio preserve)
  │     ├─ 720p: ~0.92M px, align 32px → 1-stage
  │     └─ 1080p: ~2M px, align 64px → 2-stage
  │
  ├─ 5. pipeline.generate()
  │     ├─ Embeddings depuis cache
  │     ├─ Camera LoRA switch si differente (~0.1s)
  │     │   (delta = lora_B @ lora_A * alpha/rank * strength)
  │     ├─ Setup: GaussianNoiser + EulerDiffusionStep
  │     │   CFG=1.0, STG=0.0, audio skip_step=99
  │     ├─ Image → latent via video_encoder
  │     │
  │     ├── 720p (1-stage) ────────────────────────
  │     │   denoise 8 steps (DISTILLED_SIGMA_VALUES)
  │     │   multi_modal_guider_denoising_func()
  │     │
  │     ├── 1080p (2-stage) ───────────────────────
  │     │   Stage 1: denoise ~540p, 8 steps (guiders)
  │     │   Upscale: upsample_video() x2 en latent
  │     │   Stage 2: refine ~1080p, 3 steps (simple_denoising)
  │     │
  │     ├─ VAE decode (sans tiling)
  │     └─ encode_video() → H264 MP4
  │
  ├─ 6. Sauvegarde /runpod-volume/output/{job_id}/
  ├─ 7. Upload S3 OVH (si S3_BUCKET set)
  ├─ 8. Sauvegarde dedup cache
  └─ 9. Return metadata
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
    "image_strength": 1.0,
    "last_image": "https://example.com/target.jpg",
    "last_image_strength": 1.0,
    "resolution": "720p"
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
| `last_image` | string | - | URL/base64 image cible (last frame) |
| `last_image_strength` | float | `1.0` | Force conditioning last image |
| `resolution` | string | `720p` | `720p` (1-stage) ou `1080p` (2-stage) |
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
      "s3_key": "generated/videos/medusa_i2v_job-123.mp4",
      "s3_url": "https://s3.sbg.io.cloud.ovh.net/bucket/generated/videos/medusa_i2v_job-123.mp4"
    }
  ],
  "cached": false
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

## Variables d'Environnement

| Variable | Default | Description |
|----------|---------|-------------|
| `TORCH_COMPILE` | `1` | Active torch.compile reduce-overhead |
| `TRANSFORMER_CACHE` | `1` | Cache transformer pre-fusionne (skip 1-2 min cold start) |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | Allocation VRAM dynamique |
| `S3_BUCKET` | - | Active upload S3 OVH |
| `S3_ENDPOINT_URL` | `https://s3.sbg.io.cloud.ovh.net` | Endpoint S3 |
| `S3_REGION` | `sbg` | Region S3 |
| `AWS_ACCESS_KEY_ID` | - | Credentials S3 |
| `AWS_SECRET_ACCESS_KEY` | - | Credentials S3 |
| `RUNPOD_INIT_TIMEOUT` | `600` | Timeout init worker (secondes) |
| `HF_XET_HIGH_PERFORMANCE` | `1` | Download rapide HF XET |
