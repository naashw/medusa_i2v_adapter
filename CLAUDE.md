# CLAUDE.md - Medusa I2V (ltx-pipelines)

## Projet

Pipeline Image-to-Video (LTX-2.3 22B Distilled) via ltx-pipelines sur RunPod Serverless H100 80GB.
Inference directe Python, sans ComfyUI. Output H264 MP4, 3 tiers : 540p (1-stage), 720p (2-stage), 1080p (2-stage).

## Stack Technique

| Composant | Version | Contraintes |
|-----------|---------|-------------|
| PyTorch | >=2.9 (cu128) | |
| ltx-core + ltx-pipelines | commit `9e8a28e` | |
| transformers | >=4.52, **<5.0** | v5 casse `Gemma3TextConfig` (supprime `rope_local_base_freq`) |
| SDPA | cuDNN Fused Flash Attention | Natif PyTorch, H100 sm_90 |
| Docker | Multi-stage cuda:12.8.1 | devel builder -> runtime |

## API Input

3 formats retro-compatibles : `image` (single), `images[]` (batch), `items[]` (multi-client).
Voir `src/handler.py` pour le schema complet. Points cles :
- `camera_motion` : `"depth"` pour IC-LoRA depth dolly-in. Presets texte (`dolly-in`, `dolly-out`, `dolly-left`, `dolly-right`, `jib-up`, `jib-down`, `static`) gardes pour futur rendu trajectoires depth. Alias `camera`
- `resolution` : `"540p"` (1-stage preview), `"720p"` (defaut, 2-stage), `"1080p"` (2-stage). Suffixe `-portrait` pour 9:16
- `last_image` + `last_image_strength` : optionnels, guidage derniere frame
- `items[]` : regroupes par prompt pour batching GPU, resultats reordonnes par `_original_index`

## Depth IC-LoRA (controle camera 3D)

Le mouvement de camera est controle par depth estimation + IC-LoRA Union Control, pas par des camera LoRAs individuels.

**Flow** : image source → DA3-LARGE-1.1 (depth map) → mesh 3D → rasterise N depth frames (trajectoire dolly-in GPU) → VAE encode a 0.5× resolution Stage 1 → `VideoConditionByReferenceLatent(downscale_factor=2)` → conditioning Stage 1 uniquement.

**Modeles** :
- `depth-anything/DA3-LARGE-1.1` : estimation profondeur (~1.64GB, offloadable CPU)
- `Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control` : IC-LoRA fuse permanent dans transformer (654MB)

**Mecanisme** : le IC-LoRA reste fuse en permanence (pas de swap). `ensure_lora()` est un no-op. Le `camera_motion` servira a terme a choisir la trajectoire de rendu depth (dolly-in, pan, etc.).

## Commits Git (override global)

- Titre : `type: description courte`
- Body : description de ce qu'on a fait et **pourquoi** (contexte, motivation, trade-offs)
- Format HEREDOC avec ligne vide entre titre et body

## Gotchas

- Gemma DOIT etre format **HuggingFace** (pas Comfy-Org single file)
- `num_frames` doit etre `k*8+1` (ex: 25, 49, 97)
- `height/width` multiples de 64 (2-stage : half-res multiple de 32)
- `huggingface-cli` absent de l'image Docker → utiliser `huggingface_hub.snapshot_download()`
- `start.sh` lance le warmup avec `LD_PRELOAD=""` (desactive tcmalloc sur process ephemere)
- Checkpoint FP8 scaled (`ltx-2.3-22b-dev-fp8.safetensors`) INCOMPATIBLE avec fusion LoRA → utiliser distilled BF16 + `fp8_cast()`
- IC-LoRA fuse permanent dans transformer, pas de swap dynamique
- DA3-LARGE-1.1 : 1.64GB (pas 0.4GB), install depuis GitHub `ByteDance-Seed/depth-anything-3`
- `VideoConditionByReferenceLatent` : conditioning Stage 1 only, Stage 2 sans depth
- Depth conditioning ajoute des tokens reference a la sequence d'attention → impact VRAM quadratique

## Variables d'environnement

| Variable | Defaut | Description |
|----------|--------|-------------|
| `TORCH_COMPILE` | `1` | torch.compile transformer (`0` desactive) |
| `VAE_COMPILE` | `1` | torch.compile VAE decoder (`0` desactive) |
| `COMPILE_MODE` | `default` | `default`, `reduce-overhead`, `max-autotune`, `max-autotune-no-cudagraphs` |
| `DYNAMIC_COMPILE` | `0` | `1` active dynamic shapes (defaut: static, optimal pour 8 shapes fixes) |
| `TRANSFORMER_CACHE` | `1` | Cache transformer pre-fusionne sur volume |
| `SAMPLER` | `euler` | `res2s` pour Res2sDiffusionStep (second ordre) |
| `VAE_TILING` | `0` | Tiled VAE decode (reduit VRAM 1080p, risque ghosting) |
| `BATCH_SIZE` | `5` | Max items par sub-batch denoising |
| `MAX_BATCH` | `9` | Plafond items par sub-batch handler |
| `CLEAN_OLD_CACHE` | `1` | Purge cache inductor/triton/artifacts des builds precedents au demarrage (`0` desactive) |
| `LOG_LEVEL` | `info` | `debug` active les logs Inductor verbose |
| `S3_BUCKET` | _(vide)_ | Active S3 upload OVH si defini |
| `S3_ENDPOINT_URL` | OVH SBG | Endpoint S3 |
| `ENCODE_PRESET` | `veryfast` | Preset x264 (`ultrafast`, `veryfast`, `medium`, etc.) |
| `ENCODE_CRF` | `23` | CRF qualite (0-51, lower = meilleur) |
| `DEPTH_LORA` | `1` | IC-LoRA depth control + DA3 estimation (`0` desactive) |
| `DEPTH_LORA_STRENGTH` | `1.0` | Strength du VideoConditionByReferenceLatent (0.0-1.0) |
| `DEPTH_DISPLACEMENT` | `0.07` | Amplitude deplacement camera (fraction du depth range, ex: 0.03=subtil, 0.15=agressif) |

## Documentation

| Fichier | Contenu |
|---------|---------|
| `docs/changelog.md` | Historique detaille des changements (contexte, modifications, verification, rollback) |
| `docs/audit-report-2026-03-09.md` | Audit architecture et performance du 2026-03-09 |

## Conventions

- Prefixe output : `medusa_i2v`
- Reponse API : `images[]` avec `s3_key`, `volume_path`, `s3_url`, `id`
- Output videos : `/runpod-volume/output/{job_id}/`
- S3 path : `generated/videos/{filename}`
- Caches volume : `cache/dedup/`, `cache/embeddings/`, `cache/transformer/`, `cache/inductor/{build_hash}/`, `cache/triton/`
