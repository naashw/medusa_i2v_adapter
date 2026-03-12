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
| flash-attn | FlashAttention 3 (sm_90, SDPA dispatch auto) |
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
| LTX-2.3 22B Distilled (BF16 + fp8_cast) | ~46GB | ~19-20GB | Transformer distilled (stockage FP8, compute BF16) |
| Gemma 3 12B IT (BF16) | ~24GB | GPU on-demand | Text encoder (warmup presets, custom prompts) |
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
  │     └─ Sinon → ModelLedger(checkpoint distilled BF16, quantization=fp8_cast)
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
  ├─ 2. Normalize input (items[] / images[] / image → liste uniforme)
  ├─ 3. Download ALL images en parallele (image + last_image)
  ├─ 4. Calcul resolution cible (premiere image, une seule fois)
  │     ├─ 720p: ~0.92M px, align 32px → 1-stage
  │     └─ 1080p: ~2M px, align 64px → 2-stage
  │
  ├─ 5. Grouper items par camera_motion (= meme prompt)
  │
  ├─ 6. Pour chaque groupe de prompt:
  │     ├─ Decouper en sub-batches de BATCH_SIZE
  │     ├─ 1 item  → pipeline.generate_frames()
  │     ├─ N items → pipeline.generate_batch_frames()
  │     └─ Post-processing async par item
  │
  ├─ 7. Reordonner resultats par index original
  ├─ 8. Ajouter "id" dans chaque result (si fourni)
  └─ 9. Return {"images": [...]}
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

### Batch (multi-images, retro-compatible)

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

### Batch multi-client (items[])

```json
{
  "input": {
    "items": [
      {
        "id": "job-abc-user1",
        "image": "https://example.com/photo1.jpg",
        "camera_motion": "dolly-in",
        "seed": 111
      },
      {
        "id": "job-def-user2",
        "image": "https://example.com/photo2.jpg",
        "camera_motion": "dolly-out",
        "seed": 222,
        "last_image": "https://example.com/target.jpg",
        "last_image_strength": 0.8
      }
    ],
    "resolution": "720p"
  }
}
```

Les items sont regroupes par `camera_motion` (= meme prompt) pour le batching GPU. L'ordre des resultats correspond a l'ordre des items en input.

### Parametres

**Top-level (partages)**

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `image` | string | - | URL https ou base64 (format single) |
| `images` | string[] | - | Liste d'URLs/base64 (format batch, prioritaire sur `image`) |
| `items` | object[] | - | Liste d'items multi-client (prioritaire sur `images` et `image`) |
| `camera_motion` | string | `static` | Preset (`dolly-in`, `dolly-out`, etc.) ou texte libre |
| `camera` | string | - | Alias retro-compatible pour `camera_motion` |
| `seed` | int | random | Seed generation (batch `images[]`: `seed + index` par item) |
| `num_frames` | int | `25` | Nombre de frames (doit etre k*8+1) |
| `frame_rate` | float | `24` | FPS |
| `image_strength` | float | `1.0` | Force conditioning image |
| `last_image` | string | - | URL/base64 image cible (last frame) |
| `last_image_strength` | float | `1.0` | Force conditioning last image |
| `resolution` | string | `720p` | `720p` (1-stage) ou `1080p` (2-stage) |
| `negative_prompt` | string | default | Override negative prompt |

**Per-item (dans `items[]`)**

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `id` | string | - | Identifiant client (inclus dans la response) |
| `image` | string | **requis** | URL https ou base64 |
| `camera_motion` | string | top-level | Preset ou texte libre |
| `camera` | string | - | Alias pour `camera_motion` |
| `seed` | int | random | Seed generation |
| `last_image` | string | - | URL/base64 image cible (last frame) |
| `last_image_strength` | float | `1.0` | Force conditioning last image |

### Output

```json
{
  "images": [
    {
      "id": "job-abc-user1",
      "filename": "medusa_i2v_job-abc-user1.mp4",
      "content_type": "video",
      "size_mb": 1.5,
      "volume_path": "/runpod-volume/output/job-123/medusa_i2v_job-abc-user1.mp4",
      "s3_key": "generated/videos/medusa_i2v_job-abc-user1.mp4",
      "s3_url": "https://s3.sbg.io.cloud.ovh.net/bucket/generated/videos/medusa_i2v_job-abc-user1.mp4"
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
| `TORCH_COMPILE` | `1` | Active torch.compile reduce-overhead sur le transformer |
| `VAE_COMPILE` | `1` | Active torch.compile reduce-overhead sur le video decoder |
| `TRANSFORMER_CACHE` | `1` | Cache transformer pre-fusionne (skip cold start LoRA fusion) |
| `BATCH_SIZE` | `5` | Taille max du sous-batch pour le denoising transformer |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | Allocation VRAM dynamique |
| `S3_BUCKET` | - | Active upload S3 OVH |
| `S3_ENDPOINT_URL` | `https://s3.sbg.io.cloud.ovh.net` | Endpoint S3 |
| `S3_REGION` | `sbg` | Region S3 |
| `AWS_ACCESS_KEY_ID` | - | Credentials S3 |
| `AWS_SECRET_ACCESS_KEY` | - | Credentials S3 |
| `RUNPOD_INIT_TIMEOUT` | `600` | Timeout init worker (secondes) |
| `HF_XET_HIGH_PERFORMANCE` | `1` | Download rapide HF XET |
