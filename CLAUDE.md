# CLAUDE.md - Medusa I2V (ltx-pipelines)

## Projet

Pipeline Image-to-Video (LTX-2.3 22B Distilled) via ltx-pipelines sur RunPod Serverless H100 80GB.
Inference directe Python, sans ComfyUI. Output H264 MP4, 3 tiers : 540p (1-stage), 720p (2-stage), 1080p (2-stage).

## Stack Technique

| Composant | Version | Contraintes |
|-----------|---------|-------------|
| PyTorch | >=2.11 (cu128) | cuDNN Fused Flash Attention H100 |
| ltx-core + ltx-pipelines | commit `59ca828` | Blocs lifecycle, Fp8CastLinear, layer streaming |
| transformers | >=4.57, **<5.0** | v5 restructure API RoPE Gemma3 (breaking) |
| SDPA | cuDNN Fused Flash Attention | Natif PyTorch, H100 sm_90 |
| Docker | Multi-stage cuda:12.8.1 | devel builder -> runtime |

## Modes de génération

4 modes supportes, auto-detectes par les champs presents dans l'item :

| Mode | Champs requis | Description |
|------|---------------|-------------|
| **t2v** | `prompt` | Text-to-video (pas d'image source) |
| **i2v** | `image`, `prompt` | Image-to-video standard |
| **i2v_depth** | `image`, `camera_motion="depth"` | I2V avec depth IC-LoRA pour controle camera 3D |
| **flf2v** | `image`, `last_image` | First+last frame guidance (optionnel: depth ignoré) |

Auto-detection : `detect_mode(item)` cheche presence de `image`, `last_image`, et `camera_motion == "depth"`.
Si un item a `last_image + camera_motion="depth"`, le depth est droppé (incompatible, FLF2V prioritaire).

## API Input

3 formats retro-compatibles : `image` (single), `images[]` (batch), `items[]` (multi-client).
Voir `src/handler.py` pour le schema complet. Points cles :
- `image` : optionnel (None pour t2v). URL https ou base64
- `prompt` : optionnel, custom. Fallback sur `camera_motion` si absent
- `camera_motion` : trigger depth via valeur `"depth"`. Alias `camera`. Autres valeurs ignorees
- `camera_speed_ms` : vitesse camera en m/s (optionnel, defaut env `CAMERA_SPEED_MS=0.5`). Per-item ou shared
- `camera_path` : list[dict] keyframes `{t, translation[3], rotation_quat[4]}` (optionnel, fallback dolly-in si absent). Mode 6-DOF
- `interpolation` : "linear" | "cubic" (optionnel, defaut "linear"). Mode interpolation keyframes
- `fov_degrees` : FOV camera source en degres (optionnel, defaut 60.0)
- `last_image` + `last_image_strength` : optionnels, FLF2V guidage derniere frame
- `resolution` : `"540p"` (1-stage preview), `"720p"` (defaut, 2-stage), `"1080p"` (2-stage). Suffixe `-portrait` pour 9:16
- `items[]` : regroupes par prompt pour batching GPU, resultats reordonnes par `_original_index`

## Depth IC-LoRA (controle camera 3D)

Declenche uniquement si `camera_motion == "depth"` sur un item I2V. Le mouvement de camera est controle par depth estimation metrique + IC-LoRA Union Control, pas par des camera LoRAs individuels.

**Flow** : image source → DA3METRIC-LARGE (depth metres + sky mask) → 6-DOF warp N depth frames → normalisation per-frame [0,1] → VAE encode a 0.5× resolution Stage 1 → `VideoConditionByReferenceLatent(downscale_factor=2)` → conditioning Stage 1 uniquement.

**Modeles** :
- `depth-anything/DA3METRIC-LARGE` : estimation profondeur metrique + sky segmentation (~1.64GB, offloadable CPU)
- `Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control` (`ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors`) : IC-LoRA fuse permanent dans transformer (654MB)

**Mecanisme** : le IC-LoRA est fuse une seule fois au demarrage (pas de swap, pas de unfuse). Le depth conditioning utilise un warp 6-DOF (translation + rotation_quat) via unprojection 3D, transformation, reprojection, puis forward splatting + z-buffer. Normalisation per-frame [0,1] (fonctionne car le pattern spatial change entre frames). Le ciel n'est pas warpe (sky_mask → 1.0 apres normalisation).

### Camera path 6-DOF avec Catmull-Rom

Nouveau support pour trajectoires camera scriptes (via keyframes interpolees) : translation XYZ + rotation quaternion [w, x, y, z].

**Format input** — item dict optionnel (fallback backward compat si absent) :
- `camera_path` : list[dict] avec keyframes `{t, translation[3], rotation_quat[4]}`. Temps `t` normalise [0, 1]. Si absent, fallback sur dolly-in lineaire via `camera_speed_ms`
- `interpolation` : "linear" | "cubic" (Catmull-Rom 3D + quaternion SLERP). Default "linear"
- `fov_degrees` : FOV camera source en degres. Default 60°. Calcul focale : `focal_px = width / (2 * tan(fov/2))`

**Backward compat stricte** : si `camera_path=None` ou absent, genere path dolly-in 2-kf interne identique ancien comportement (translation `[0,0,δ]`, rotation identity).

**Implémentation** : module `src/camera_path.py` contient fonctions quaternion math (quat_to_matrix, slerp), Catmull-Rom vec3 + quat, interpolate_camera_path. Fonction `_warp_depth_generic` remplace `_warp_depth_dolly` pour support 6-DOF.

## Commits Git (override global)

- Titre : `type: description courte`
- Body : description de ce qu'on a fait et **pourquoi** (contexte, motivation, trade-offs)
- Format HEREDOC avec ligne vide entre titre et body

## Gotchas

- Gemma DOIT etre format **HuggingFace** (pas Comfy-Org single file). Le `config.json` doit contenir `rope_local_base_freq` dans `text_config` (auto-corrige par `start.sh` via `AutoConfig.save_pretrained`)
- `num_frames` doit etre `k*8+1` (ex: 25, 49, 97)
- `height/width` multiples de 64 (2-stage : half-res multiple de 32)
- `huggingface-cli` absent de l'image Docker → utiliser `huggingface_hub.snapshot_download()`
- `start.sh` lance le warmup avec `LD_PRELOAD=""` (desactive tcmalloc sur process ephemere)
- Checkpoint FP8 scaled (`ltx-2.3-22b-dev-fp8.safetensors`) INCOMPATIBLE avec fusion LoRA → utiliser distilled BF16 + `fp8_cast()`
- IC-LoRA fuse permanent dans transformer, pas de swap dynamique
- DA3METRIC-LARGE : ~1.64GB, install depuis GitHub `ByteDance-Seed/depth-anything-3`
- `VideoConditionByReferenceLatent` : conditioning Stage 1 only, Stage 2 sans depth
- Depth conditioning ajoute des tokens reference a la sequence d'attention → impact VRAM quadratique

## Variables d'environnement

| Variable | Defaut | Description |
|----------|--------|-------------|
| `TORCH_COMPILE` | `1` | torch.compile transformer (`0` desactive) |
| `VAE_COMPILE` | `1` | torch.compile VAE decoder (`0` desactive) |
| `COMPILE_MODE` | `default` | `default`, `reduce-overhead`, `max-autotune`, `max-autotune-no-cudagraphs` |
| `FULLGRAPH` | `1` | `1` active fullgraph torch.compile (graphe complet sans breaks, CUDAGraphs compatible) |
| `DYNAMIC_COMPILE` | `0` | `1` active dynamic shapes transformer (defaut: static, optimal pour 8 shapes fixes) |
| `VAE_COMPILE_MODE` | `COMPILE_MODE` | Mode compile VAE independant. Fallback sur `COMPILE_MODE` |
| `VAE_FULLGRAPH` | `FULLGRAPH` | Fullgraph VAE independant. Fallback sur `FULLGRAPH` |
| `VAE_DYNAMIC_COMPILE` | `DYNAMIC_COMPILE` | `1` active dynamic shapes VAE decoder independamment du transformer. Fallback sur `DYNAMIC_COMPILE` |
| `TRANSFORMER_CACHE` | `1` | Cache transformer pre-fusionne sur volume |
| `SAMPLER` | `euler` | `res2s` pour Res2sDiffusionStep (second ordre) |
| `VAE_TILING` | `0` | Tiled VAE decode (reduit VRAM 1080p, risque ghosting) |
| `MAX_BATCH` | `9` | Plafond items par sub-batch handler |
| `CLEAN_OLD_CACHE` | `1` | Purge cache inductor/triton/artifacts des builds precedents au demarrage (`0` desactive) |
| `LOG_LEVEL` | `info` | `debug` active les logs Inductor verbose |
| `S3_BUCKET` | _(vide)_ | Active S3 upload OVH si defini |
| `S3_ENDPOINT_URL` | `https://s3.sbg.io.cloud.ovh.net` | Endpoint S3 |
| `S3_REGION` | `sbg` | Region S3 |
| `ENCODE_PRESET` | `veryfast` | Preset x264 (`ultrafast`, `veryfast`, `medium`, etc.) |
| `ENCODE_CRF` | `23` | CRF qualite (0-51, lower = meilleur) |
| `DEPTH_LORA` | `1` | IC-LoRA depth control + DA3METRIC estimation (`0` desactive) |
| `DEPTH_LORA_STRENGTH` | `1.0` | Strength du VideoConditionByReferenceLatent (0.0-1.0) |
| `CAMERA_SPEED_MS` | `0.5` | Vitesse camera defaut en m/s (0.3=subtil, 0.5=modere, 1.0=rapide) |

## Documentation

| Fichier | Contenu |
|---------|---------|
| `docs/changelog.md` | Historique detaille des changements (contexte, modifications, verification, rollback) |
| `docs/audit-report-2026-03-09.md` | Audit architecture et performance du 2026-03-09 |

## Conventions

- Prefixe output : `medusa_i2v`
- Reponse API : `images[]` avec `filename`, `content_type`, `size_mb`, `s3_key`, `s3_url` (si S3), `volume_path` (si volume), `id` (si fourni)
- Output videos : `/runpod-volume/output/{job_id}/`
- S3 path : `generated/videos/{filename}`
- Caches volume : `cache/dedup/`, `cache/embeddings/`, `cache/transformer/`, `cache/inductor/{build_hash}/`, `cache/triton/`
