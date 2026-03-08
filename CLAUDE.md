# CLAUDE.md - Medusa I2V (ltx-pipelines)
<!-- last-mirror-test: 2026-02-23 -->

## Projet

Pipeline Image-to-Video utilisant LTX-2 19B avec effet camera dolly.
Inference directe via ltx-pipelines (sans ComfyUI).
Objectif : generation rapide de videos dolly a partir d'images, qualite correcte.

## Stack Technique

| Composant | Version |
|-----------|---------|
| CUDA | 12.8.1 + cuDNN |
| Python | 3.12 (Ubuntu 24.04 natif) |
| PyTorch | >=2.7.1 (wheel cu128) |
| ltx-core + ltx-pipelines | Lightricks/LTX-2 commit `28c3c73` |
| transformers | >=4.52, <5.0 (v5 casse Gemma3TextConfig) |
| huggingface-hub | >=0.28 (avec HF XET) |
| runpod | >=1.7, <2.0 |
| boto3 | >=1.34 (S3 OVH) |
| SageAttention | 2++ from source (sm_90, INT8-QK/FP8-PV) |
| Docker | Multi-stage (cuda:12.8.1-devel -> runtime) |

## Contexte Technique

- **Runtime** : ltx-pipelines (Python direct) sur RunPod Serverless (H100 80GB, HBM3)
- **Modele principal** : LTX-2 19B AV model (BF16, ~35GB VRAM base + camera LoRA fuse dynamique)
- **Text encoder** : Gemma 3 12B IT (BF16, CPU — ~24GB RAM, format HuggingFace)
- **Approche** : Pipeline 1-stage (720p) ou 2-stage (540p → upscale x2 → refine 1080p), Euler distilled, audio skip
- **Output** : H264 MP4, 24fps, ~1 seconde (25 frames), 720p ou 1080p
- **S3** : Upload OVH S3 optionnel (boto3, active via `S3_BUCKET` env var)

## Structure

- `Dockerfile` — image Docker multi-stage (devel builder -> runtime), sans ComfyUI
- `docker-compose.yml` — lancement local avec GPU
- `src/start.sh` — script de demarrage (telecharge modeles via hf_xet + valide safetensors + lance handler)
- `src/warmup_embeddings.py` — warmup embeddings low-RAM (safe_open direct, sans ModelLedger)
- `src/pipeline.py` — classe MedusaPipeline (inference ltx-pipelines)
- `src/handler.py` — handler RunPod serverless (API simplifiee, eager init, download images parallele)
- `workflows/` — reference ComfyUI (plus utilises en production)
- `scripts/` — scripts utilitaires (test, envoi, conversion)
- `docs/` — documentation et exemples
- `test-data/` — images de test

## API Input

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

`last_image` et `last_image_strength` sont optionnels. Si `last_image` est fourni, la video est guidee vers cette image cible en derniere frame.

`resolution` : `"720p"` (defaut) ou `"1080p"`. 720p = pipeline 1-stage (~0.92M px). 1080p = pipeline 2-stage (540p → upscale x2 → refine, ~2M px).

Cameras supportees : dolly-in, dolly-out, dolly-left, dolly-right, jib-down, jib-up, static.

## Modeles

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

## Architecture Pipeline

- **MedusaPipeline** : encapsule ltx-pipelines avec gestion lifecycle modeles
  - Video encoder persistent en VRAM (~1.3GB)
  - Video decoder persistent en VRAM (~2GB)
  - Spatial upsampler x2 persistent en VRAM (~1GB)
  - Transformer base (distilled + I2V) permanent en VRAM (~35GB), compile une seule fois
  - SageAttention2++ patch runtime sur ~288 modules Attention du transformer (INT8-QK/FP8-PV, kernel sm_90 auto-selectionne), fallback SDPA si mask present ou non installe
  - Camera LoRA fuse/unfuse dynamiquement in-place (delta = lora_up @ lora_down, ~0.1s de switch)
  - Deltas camera precalcules et caches sur CPU, transferes vers GPU a la demande
  - Embeddings pre-caches sur disque (generes par warmup_embeddings.py)
  - Cache transformer pre-fusionne dans `/runpod-volume/cache/transformer/` (checkpoint safetensors avec distilled + I2V deja fusionnes, elimine ~1-2 min de LoRA fusion aux cold starts suivants, invalidation auto par hash checkpoint + LoRAs + strengths, desactivable via `TRANSFORMER_CACHE=0`)
- **Pipeline 2-stage** :
  - Stage 1 : denoise a ~540p (half-res), 8 steps distilled, guiders (CFG=1, STG=0, audio skip)
  - Upscale : `upsample_video()` x2 en espace latent (un-normalize → upsampler → normalize)
  - Stage 2 : refine a ~1080p (full-res), 3 steps distilled, `simple_denoising_func` (1 forward/step)
  - Meme transformer pour les 2 stages (pas de rebuild), camera LoRA reste fuse
- **Eager init** : pipeline init complet AVANT `runpod.serverless.start()` — premier job sans cold start
- **Download parallele** : `image` et `last_image` telecharges en parallele via ThreadPoolExecutor
- **torch.compile** : `torch.compile(mode="reduce-overhead")` sur le transformer base (desactivable via `TORCH_COMPILE=0`). 2 shapes (540p + 1080p) → 2 CUDA graph captures au premier job, cachees ensuite.
- **SageAttention2++** : patch runtime `attention_function` sur les modules `Attention` du transformer uniquement (pas VAE, pas encoder, pas upsampler). Kernel INT8-QK/FP8-PV auto-selectionne sur H100 sm_90. Fallback `PytorchAttention` (SDPA) si mask present. Desactivable via `SAGE_ATTENTION=0`.
- **Ordre d'init** : warmup embeddings (process isole) → base transformer (distilled + I2V) → patch SageAttention → torch.compile → fuse camera dolly-in → video encoder → video decoder → spatial upsampler
- **warmup_embeddings.py** : charge uniquement les 59 cles TE via safe_open (2.7GB) + Gemma `low_cpu_mem_usage=True`. Peak ~35GB.
- **Audio skip** : `skip_step=99` sur audio guider → audio compute seulement au step 0/8
- **CFG desactive** : `cfg_scale=1.0, stg_scale=0.0` → 1 seul forward par step
- **Sigmas distilled** : stage 1 = 8 steps (DISTILLED_SIGMA_VALUES), stage 2 = 3 steps (STAGE_2_DISTILLED_SIGMA_VALUES)
- **Resolution** : ~2M px cible, alignement 64px (half-res doit etre multiple de 32)
- **Budget VRAM** : ~50-55 GB sur H100 80GB (marge ~25-30 GB)

## Flow de Demarrage (start.sh)

1. Signal handlers (SIGTERM/SIGINT/SIGQUIT)
2. tcmalloc (LD_PRELOAD) pour optimisation memoire
3. Creation arborescence `/runpod-volume/models/`
4. Download modeles (hf_xet) + validation safetensors via `safe_open()`
5. Audit volume (dry-run)
6. Mode detection : SERVERLESS → warmup + handler.py | GPU POD → JupyterLab :8888

## Flow d'Init Pipeline (eager, 1 seule fois)

1. `warmup_embeddings()` — cache .pt ou Gemma 3 12B CPU (~35GB RAM peak) → encode 7 prompts camera + 1 negative → liberer
2. `get_transformer(dolly-in)` — cache pre-fusionne ou ModelLedger(checkpoint + distilled + I2V) → patch SageAttention2++ → torch.compile(reduce-overhead) → fuse camera delta
3. `load_video_encoder()` → VRAM (~1GB, persistent)
4. `load_video_decoder()` → VRAM (~2GB, persistent)
5. `load_spatial_upsampler()` → VRAM (~1GB, persistent)

## Flow d'Inference (par job)

1. Hash input SHA256 → check dedup cache → return si hit
2. Parse input + download images en parallele (ThreadPoolExecutor)
3. Calcul resolution cible (aspect ratio preserve) : 720p align 32px (1-stage), 1080p align 64px (2-stage)
4. `pipeline.generate()` :
   - Embeddings depuis cache (ou encode CPU si override)
   - Camera LoRA switch si differente (~0.1s, delta = lora_B @ lora_A * alpha/rank * strength)
   - Setup : GaussianNoiser + EulerDiffusionStep, CFG=1.0, STG=0.0, audio skip_step=99
   - Image → latent via video_encoder
   - **720p** : denoise 8 steps (DISTILLED_SIGMA_VALUES) avec guiders
   - **1080p** : Stage 1 denoise ~540p 8 steps → upsample_video() x2 latent → Stage 2 refine ~1080p 3 steps (simple_denoising)
   - VAE decode (sans tiling) → encode_video() → H264 MP4
5. Sauvegarde volume + upload S3 + dedup cache

## Conventions

- Prefixe output : `medusa_i2v`
- Reponse API : `images[]` avec `s3_key` + `volume_path` + `s3_url` (si S3 active)
- Input images : supporte URL https OU base64
- Output videos : sauvegardees dans `/runpod-volume/output/{job_id}/`
- S3 upload : `generated/videos/{filename}` (OVH S3, optionnel via `S3_BUCKET`)
- Dedup cache par hash dans `/runpod-volume/cache/dedup/`
- Embeddings cache dans `/runpod-volume/cache/embeddings/`

## Points d'Attention

- Le text encoder Gemma DOIT etre format HuggingFace (pas Comfy-Org single file)
- num_frames doit etre k*8+1 (ex: 25, 49, 97)
- height/width doivent etre multiples de 64 (pipeline 2-stage : half-res doit etre multiple de 32)
- `transformers` DOIT etre pince `>=4.52,<5.0` (v5 supprime `rope_local_base_freq` de Gemma3TextConfig)
- `huggingface-cli` n'est pas dans l'image Docker, utiliser `huggingface_hub.snapshot_download()` a la place
- `start.sh` lance le warmup avec `LD_PRELOAD=""` pour desactiver tcmalloc (inutile sur process ephemere)
- `start.sh` valide les fichiers `.safetensors` existants via `safe_open()` avant de skip le telechargement (detecte les fichiers corrompus/partiels)
- S3 env vars : `S3_BUCKET`, `S3_ENDPOINT_URL` (defaut OVH SBG), `S3_REGION`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`
- `SAGE_ATTENTION=1` (defaut) : active SageAttention2++ sur le transformer. `0` pour rollback complet vers SDPA
- `SAGE_COMPILE_DISABLE=0` (defaut) : `1` wrappe sageattn dans `torch.compiler.disable` si CUDA graphs posent probleme
- `TORCH_COMPILE=0` pour desactiver torch.compile (debug ou compatibilite)
- `TRANSFORMER_CACHE=0` pour desactiver le cache transformer pre-fusionne (force rebuild a chaque cold start)
