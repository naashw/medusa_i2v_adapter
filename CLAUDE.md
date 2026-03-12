# CLAUDE.md - Medusa I2V (ltx-pipelines)

## Projet

Pipeline Image-to-Video utilisant LTX-2.3 22B Distilled avec effets camera.
Inference directe via ltx-pipelines (sans ComfyUI).
Objectif : generation rapide de videos a partir d'images, qualite correcte.

## Stack Technique

| Composant | Version |
|-----------|---------|
| CUDA | 12.8.1 + cuDNN |
| Python | 3.12 (Ubuntu 24.04 natif) |
| PyTorch | >=2.9 (wheel cu128) |
| ltx-core + ltx-pipelines | Lightricks/LTX-2 commit `9e8a28e` |
| transformers | >=4.52, <5.0 (v5 casse Gemma3TextConfig) |
| huggingface-hub | >=0.28 (avec HF XET) |
| runpod | >=1.7, <2.0 |
| boto3 | >=1.34 (S3 OVH) |
| flash-attn | FlashAttention 3 (sm_90, SDPA dispatch auto) |
| Docker | Multi-stage (cuda:12.8.1-devel -> runtime) |

## Contexte Technique

- **Runtime** : ltx-pipelines (Python direct) sur RunPod Serverless (H100 80GB, HBM3)
- **Modele principal** : LTX-2.3 22B Distilled (checkpoint BF16 + `QuantizationPolicy.fp8_cast()` → stockage FP8 ~19-20GB VRAM, compute BF16)
- **Text encoder** : Gemma 3 12B IT (BF16, GPU on-demand, format HuggingFace)
- **Approche** : Pipeline 1-stage (720p) ou 2-stage (540p → upscale x2 → refine 1080p), 8 steps distilled, audio disabled
- **Output** : H264 MP4, 24fps, ~1 seconde (25 frames), 720p ou 1080p
- **S3** : Upload OVH S3 optionnel (boto3, active via `S3_BUCKET` env var)

## Structure

- `Dockerfile` — image Docker multi-stage (devel builder -> runtime), sans ComfyUI
- `docker-compose.yml` — lancement local avec GPU
- `src/start.sh` — script de demarrage (migration volume + telecharge modeles via hf_xet + valide safetensors + lance handler)
- `src/warmup_embeddings.py` — warmup embeddings (safe_open direct + Gemma GPU)
- `src/pipeline.py` — classe MedusaPipeline (inference ltx-pipelines, FP8 cast)
- `src/handler.py` — handler RunPod serverless (API simplifiee, eager init, download images parallele)
- `src/prompts.py` — presets camera_motion + negative prompt
- `src/audit_volume.py` — audit fichiers inutiles sur volume
- `workflows/` — reference ComfyUI (plus utilises en production)
- `scripts/` — scripts utilitaires (test, envoi, conversion)
- `docs/` — documentation et exemples
- `test-data/` — images de test

## API Input

### Single image (retro-compatible)
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

`items[]` : batch multi-client. Chaque item a ses propres `id`, `image`, `camera_motion`, `seed`, `last_image`, `last_image_strength`. Les items sont regroupes par `camera_motion` (= meme prompt = memes embeddings) pour le batching GPU. L'ordre des resultats correspond a l'ordre des items en input.

`last_image` et `last_image_strength` sont optionnels (per-item dans `items[]`, top-level dans `image`/`images`). Si `last_image` est fourni, la video est guidee vers cette image cible en derniere frame.

`resolution` : `"720p"` (defaut) ou `"1080p"`. 720p = pipeline 1-stage (~0.92M px). 1080p = pipeline 2-stage (540p → upscale x2 → refine, ~2M px).

`images` (pluriel) : liste d'URLs/base64 pour generation batch. Chaque image recoit un seed unique (`seed + index`). Le denoising tourne en batch=N sur un seul forward transformer. Retro-compatible avec `image` (singulier).

`camera_motion` : preset connu (`dolly-in`, `dolly-out`, `dolly-left`, `dolly-right`, `jib-up`, `jib-down`, `static`) ou texte libre. Retro-compatible via `camera` (alias). Per-item dans `items[]`, top-level dans `image`/`images`.

## Modeles

| Modele | Taille | VRAM | Role |
|--------|--------|------|------|
| LTX-2.3 22B Distilled (BF16 + fp8_cast) | ~46GB | ~19-20GB | Transformer distilled (stockage FP8, compute BF16) |
| Gemma 3 12B IT (BF16) | ~24GB | GPU on-demand | Text encoder (warmup presets, custom prompts) |
| Spatial Upscaler x2 | ~1GB | ~1GB | Upscale latent (pipeline 2-stage) |
| Temporal Upscaler x2 | ~262MB | - | Reporte (aucun pipeline officiel) |
| Video Encoder | inclus checkpoint | ~1GB | Image -> latent |
| Video Decoder | inclus checkpoint | ~2GB | Latent -> pixels |

## Architecture Pipeline

- **MedusaPipeline** : encapsule ltx-pipelines avec gestion lifecycle modeles
  - Video encoder persistent en VRAM (~1GB)
  - Video decoder persistent en VRAM (~2GB)
  - Spatial upsampler x2 persistent en VRAM (~1GB)
  - Transformer distilled permanent en VRAM (~19-20GB via FP8 cast), compile une seule fois
  - Embeddings pre-caches sur disque (generes par warmup_embeddings.py)
  - Cache transformer dans `/runpod-volume/cache/transformer/` (checkpoint safetensors, invalidation auto par hash checkpoint, desactivable via `TRANSFORMER_CACHE=0`)
- **Pipeline 2-stage** :
  - Stage 1 : denoise a ~540p (half-res), 8 steps distilled
  - Upscale : `upsample_video()` x2 en espace latent
  - Stage 2 : refine a ~1080p (full-res), 3 steps distilled, `simple_denoising_func`
  - Meme transformer pour les 2 stages
- **Eager init** : pipeline init complet AVANT `runpod.serverless.start()` — premier job sans cold start
- **Download parallele** : toutes les images (image + last_image) de tous les items telecharges en parallele via ThreadPoolExecutor au debut du job
- **torch.compile** : `torch.compile(dynamic=True)` sur le transformer ET le video decoder (desactivable via `TORCH_COMPILE=0` et `VAE_COMPILE=0`). `mode=max-autotune` par defaut (CUDA graphs + autotuning Triton), configurable via `COMPILE_MODE`. Cache Triton + TorchInductor persistant sur volume (`TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR`). Cache Inductor versionne par build hash (`/app/.build_hash` = md5 source + pip freeze) → invalidation auto a chaque nouveau build Docker. `cache_size_limit=32`, `recompile_limit=16`, `automatic_dynamic_shapes=True` pour eviter les recompilations entre stages 1/2.
- **Batching multi-client** : items regroupes par `camera_motion` (= meme prompt). Toujours traite via `generate_batch_frames()` avec padding a `MAX_BATCH` (defaut 3) pour shape fixe dans le compile cache. Resultats reordonnes par `_original_index`.
- **Batching GPU** : `generate_batch_frames()` traite N images en un seul forward transformer (batch=N). Per-item noise (seeds differents), image encoding individuel. Spatial upscaler et VAE decode en sub-batches de `BATCH_SIZE`. `torch.cuda.empty_cache()` entre chaque stage (transformer → upscaler → stage 2 → VAE). Configurable via `BATCH_SIZE` (defaut 5).
- **Async post-processing** : MP4 encode + S3 upload en parallele du GPU via ThreadPoolExecutor(3)
- **Ordre d'init** : warmup embeddings (process isole) → transformer distilled (FP8 cast) → torch.compile → video encoder → video decoder → spatial upsampler
- **warmup_embeddings.py** : charge les cles TE via safe_open + Gemma sur GPU (device_map) si dispo. Encoding ~5-10s GPU vs ~30-40s CPU. Cleanup VRAM complet apres.
- **CFG desactive** : `cfg_scale=1.0, stg_scale=0.0` → 1 seul forward par step
- **Audio disabled** : audio modality `enabled=False` dans le forward transformer
- **Sigmas distilled** : stage 1 = 8 steps (DISTILLED_SIGMA_VALUES), stage 2 = 3 steps (STAGE_2_DISTILLED_SIGMA_VALUES)
- **Resolution** : ~2M px cible pour 1080p, ~0.92M px pour 720p, alignement 64/32px
- **Budget VRAM** : ~35-40 GB sur H100 80GB (marge ~40-45 GB)

## Flow de Demarrage (start.sh)

1. Signal handlers (SIGTERM/SIGINT/SIGQUIT)
2. tcmalloc (LD_PRELOAD) pour optimisation memoire
3. Migration volume (cleanup anciens modeles LTX-2, download LTX-2.3)
4. Setup caches (Inductor versionne par build hash `/app/.build_hash`, Triton)
5. Download modeles (hf_xet) + validation safetensors via `safe_open()`
6. Mode detection : SERVERLESS → warmup + handler.py | GPU POD → JupyterLab :8888

## Flow d'Init Pipeline (eager, 1 seule fois)

1. `warmup_embeddings()` — cache .pt ou Gemma 3 12B GPU (device_map) → encode 7 presets camera_motion + 1 negative → cleanup VRAM complet
2. `get_transformer()` — cache pre-fusionne ou ModelLedger(checkpoint distilled BF16, quantization=fp8_cast) → torch.compile(dynamic=True)
3. `load_video_encoder()` → VRAM (~1GB, persistent)
4. `load_video_decoder()` → VRAM (~2GB, persistent) + torch.compile(dynamic=True)
5. `load_spatial_upsampler()` → VRAM (~1GB, persistent)

## Flow d'Inference (par job)

1. Hash input SHA256 → check dedup cache → return si hit
2. Normaliser input (3 formats : `items[]`, `images[]`, `image`) → liste uniforme d'items
3. Download ALL images en parallele (image + last_image de chaque item) via ThreadPoolExecutor
4. Calcul resolution cible (premiere image, une seule fois) : 720p align 32px (1-stage), 1080p align 64px (2-stage)
5. Grouper items par `camera_motion` (= meme prompt = memes embeddings)
6. Pour chaque groupe de prompt :
   - Decouper en sub-batches de `BATCH_SIZE`
   - Pour chaque sub-batch :
     - Si 1 item → `pipeline.generate_frames()` (supporte `negative_override`)
     - Si N items → `pipeline.generate_batch_frames()` (batch homogene, N items en un forward)
   - Post-processing async par item (MP4 encode + S3 upload)
7. Reordonner resultats par `_original_index`, ajouter `id` si fourni
8. Return `{"images": [...]}`

## Conventions

- Prefixe output : `medusa_i2v`
- Reponse API : `images[]` avec `s3_key` + `volume_path` + `s3_url` (si S3 active) + `id` (si fourni dans l'input `items[]`)
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
- `TORCH_COMPILE=1` (defaut) : torch.compile sur le transformer. `0` pour desactiver
- `COMPILE_MODE=max-autotune` (defaut) : mode torch.compile. Valeurs : `default`, `reduce-overhead`, `max-autotune`, `max-autotune-no-cudagraphs`
- `TRITON_CACHE_DIR` : repertoire cache Triton persistant (defaut `/runpod-volume/cache/triton/`). Autotuning sauvegarde les configs kernel optimales
- `TORCHINDUCTOR_CACHE_DIR` : repertoire cache TorchInductor persistant (defaut `/runpod-volume/cache/inductor/{build_hash}/`). Versionne automatiquement par build hash Docker — invalide auto a chaque rebuild
- `VAE_COMPILE=1` (defaut) : torch.compile(dynamic=True) sur le video decoder. `0` pour desactiver
- `TRANSFORMER_CACHE=1` (defaut) : cache transformer pre-fusionne. `0` pour desactiver
- `SAMPLER=euler` (defaut) : stepper de denoising. `res2s` pour Res2sDiffusionStep (second ordre)
- `VAE_TILING=0` (defaut) : `1` pour activer le tiled VAE decode (reduit VRAM en 1080p, risque ghosting temporal)
- `BATCH_SIZE=5` (defaut) : taille max du sous-batch pour le denoising transformer en batch mode
- `MAX_BATCH=3` (defaut) : handler pad toujours le sub-batch a cette taille pour shape fixe dans le compile cache Dynamo (evite recompilations)
- Le checkpoint FP8 scaled (`ltx-2.3-22b-dev-fp8.safetensors`) est INCOMPATIBLE avec la fusion LoRA dans ltx-core — utiliser le checkpoint distilled BF16 avec `fp8_cast()` a la place
