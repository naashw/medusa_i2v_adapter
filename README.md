# Medusa I2V - ltx-pipelines + LTX-2.3 22B FP8

Pipeline Image-to-Video avec effets camera, inference directe via ltx-pipelines (sans ComfyUI).

## Stack Technique

| Composant | Version |
|-----------|---------|
| CUDA | 12.8.1 + cuDNN |
| Python | 3.12 (Ubuntu 24.04 natif) |
| PyTorch | >=2.7.1 (wheel cu128) |
| ltx-core + ltx-pipelines | Lightricks/LTX-2 commit `9e8a28e` |
| transformers | >=4.52, <5.0 |
| huggingface-hub | >=0.28 (avec HF XET) |
| runpod | >=1.7, <2.0 |
| boto3 | >=1.34 (S3 OVH) |
| SageAttention | 2++ from source (sm_90, INT8-QK/FP8-PV) |
| Docker | Multi-stage (cuda:12.8.1-devel -> runtime) |

## Structure

```
medusa_i2v_adapter/
├── Dockerfile              # Multi-stage (cuda:12.8.1-devel -> runtime)
├── docker-compose.yml      # Lancement local avec GPU
├── requirements.txt        # Dependencies Python
├── src/
│   ├── start.sh            # Download modeles + migration + validation + lancement
│   ├── warmup_embeddings.py # Warmup embeddings low-RAM (process isole)
│   ├── pipeline.py         # MedusaPipeline (inference ltx-pipelines, FP8 cast)
│   ├── handler.py          # Handler RunPod serverless
│   ├── prompts.py          # Presets camera_motion + negative prompt
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
| LTX-2.3 22B FP8 | ~22GB | ~19-20GB | Transformer principal (FP8 cast, compute BF16) |
| Gemma 3 12B IT (BF16) | ~24GB | GPU on-demand | Text encoder (warmup presets, custom prompts) |
| Distilled LoRA (strength 0.7) | ~50MB | fuse dans transformer | Acceleration inference (8 steps) |
| Spatial Upscaler x2 | ~1GB | ~1GB | Upscale latent (pipeline 2-stage) |
| Temporal Upscaler x2 | ~1GB | - | Reporte (aucun pipeline officiel) |
| Video Encoder | inclus checkpoint | ~1GB | Image -> latent |
| Video Decoder | inclus checkpoint | ~2GB | Latent -> pixels |

**Budget VRAM total : ~35-40GB sur H100 80GB (marge ~40-45GB)**

## Docker

### Build

```bash
DOCKER_BUILDKIT=1 docker build --platform linux/amd64 -t medusa-i2v:2.3-fp8 .
```

### Run (GPU Pod)

```bash
docker run --gpus all -p 8888:8888 -v /workspace:/workspace medusa-i2v:2.3-fp8
```

### Run (Serverless)

```bash
docker run --gpus all -e SERVERLESS=true -v /workspace:/workspace medusa-i2v:2.3-fp8
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
  ├─ 3. Migration volume (cleanup anciens modeles LTX-2, download LTX-2.3)
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
  │     └─ Sinon → Gemma 3 12B GPU (device_map)
  │        → encode 7 presets camera_motion + 1 negative
  │        → sauvegarder cache → liberer VRAM
  │
  ├─ 2. get_transformer()
  │     └─ Cache transformer pre-fusionne (fingerprint OK) → charger
  │     └─ Sinon → ModelLedger(checkpoint FP8 + distilled LoRA, quantization=fp8_cast)
  │     └─ Patch SageAttention2++ (~288 modules Attention)
  │     └─ torch.compile(mode="reduce-overhead")
  │
  ├─ 3. load_video_encoder()     → VRAM (~1GB, persistent)
  ├─ 4. load_video_decoder()     → VRAM (~2GB, persistent) + torch.compile
  └─ 5. load_spatial_upsampler() → VRAM (~1GB, persistent)
```

## Flow d'Inference (par job)

```
handler(job)
  │
  ├─ 1. Hash input (SHA256) → check dedup cache → return si hit
  ├─ 2. Parse input (image/images, camera_motion, seed, resolution...)
  ├─ 3. Download images en parallele (ThreadPoolExecutor)
  ├─ 4. Calcul resolution cible (aspect ratio preserve)
  │     ├─ 720p: ~0.92M px, align 32px → 1-stage
  │     └─ 1080p: ~2M px, align 64px → 2-stage
  │
  ├─ 5. pipeline.generate_frames()
  │     ├─ Embeddings depuis cache (preset) ou Gemma on-demand (custom)
  │     ├─ Setup: GaussianNoiser + EulerDiffusionStep
  │     │   CFG=1.0, STG=0.0, audio disabled
  │     ├─ Image → latent via video_encoder
  │     │
  │     ├── 720p (1-stage) ────────────────────────
  │     │   denoise 8 steps (DISTILLED_SIGMA_VALUES)
  │     │
  │     ├── 1080p (2-stage) ───────────────────────
  │     │   Stage 1: denoise ~540p, 8 steps
  │     │   Upscale: upsample_video() x2 en latent
  │     │   Stage 2: refine ~1080p, 3 steps (simple_denoising)
  │     │
  │     └─ VAE decode (sans tiling) → frames CPU
  │
  ├─ 6. Post-processing async (MP4 encode + S3 upload)
  ├─ 7. Sauvegarde /runpod-volume/output/{job_id}/
  ├─ 8. Sauvegarde dedup cache
  └─ 9. Return metadata
```

## API

### Single Image (retro-compatible)

```json
{
  "input": {
    "image": "https://example.com/photo.jpg",
    "camera_motion": "dolly-in",
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

### Batch (multi-images)

```json
{
  "input": {
    "images": ["https://example.com/photo1.jpg", "https://example.com/photo2.jpg"],
    "camera_motion": "dolly-in",
    "seed": 12345,
    "resolution": "720p"
  }
}
```

### Parametres

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `image` | string | **requis** | URL https ou base64 (single) |
| `images` | string[] | - | Liste d'URLs/base64 (batch, prioritaire sur `image`) |
| `camera_motion` | string | `static` | Preset (`dolly-in`, `dolly-out`, `dolly-left`, `dolly-right`, `jib-up`, `jib-down`, `static`) ou texte libre |
| `camera` | string | - | Alias retro-compatible pour `camera_motion` |
| `seed` | int | random | Seed generation (batch: `seed + index` par item) |
| `num_frames` | int | `25` | Nombre de frames (doit etre k*8+1) |
| `frame_rate` | float | `24` | FPS |
| `image_strength` | float | `1.0` | Force conditioning image |
| `last_image` | string | - | URL/base64 image cible (last frame) |
| `last_image_strength` | float | `1.0` | Force conditioning last image |
| `resolution` | string | `720p` | `720p` (1-stage) ou `1080p` (2-stage) |
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
      "camera_motion": "dolly-in",
      "seed": 42
    }
  }'
```

## Variables d'Environnement

| Variable | Default | Description |
|----------|---------|-------------|
| `SAGE_ATTENTION` | `1` | Active SageAttention2++ sur le transformer |
| `SAGE_COMPILE_DISABLE` | `0` | Wrappe sageattn dans `torch.compiler.disable` si CUDA graphs posent probleme |
| `TORCH_COMPILE` | `1` | Active torch.compile reduce-overhead sur le transformer |
| `VAE_COMPILE` | `1` | Active torch.compile reduce-overhead sur le video decoder |
| `TRANSFORMER_CACHE` | `1` | Cache transformer pre-fusionne (skip cold start LoRA fusion) |
| `BATCH_SIZE` | `2` | Taille max du sous-batch pour le denoising transformer |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | Allocation VRAM dynamique |
| `S3_BUCKET` | - | Active upload S3 OVH |
| `S3_ENDPOINT_URL` | `https://s3.sbg.io.cloud.ovh.net` | Endpoint S3 |
| `S3_REGION` | `sbg` | Region S3 |
| `AWS_ACCESS_KEY_ID` | - | Credentials S3 |
| `AWS_SECRET_ACCESS_KEY` | - | Credentials S3 |
| `RUNPOD_INIT_TIMEOUT` | `600` | Timeout init worker (secondes) |
| `HF_XET_HIGH_PERFORMANCE` | `1` | Download rapide HF XET |
