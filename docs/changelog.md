# Changelog — Medusa I2V

## 2026-04-11 — Phase A : Camera path 6-DOF avec Catmull-Rom (trajet cinématographique scriptable)

Support trajectoires camera 6-DOF (translation X/Y/Z + rotation quaternion) via keyframes interpolees. Refactor warp depth dolly → generic 6-DOF avec fallback backward compat strict.

### Fichiers ajoutes/modifies

- `src/camera_path.py` : **NOUVEAU** (~200 lignes). Fonctions quaternion math (quat_to_matrix, slerp), Catmull-Rom vec3 + quat, interpolate_camera_path
- `src/pipeline.py` :
  - Imports : `import math`, `from camera_path import interpolate_camera_path, quat_to_matrix`
  - Nouvelle fonction `_warp_depth_generic` : 6-DOF warp via unproject→transform→reproject + forward splatting (remplace `_warp_depth_dolly` pour cas general)
  - `render_depth_sequence()` : nouvelle signature `camera_path`, `interpolation`, `focal_px`. Fallback dolly-in 2-kf si `camera_path=None`
  - `create_depth_conditioning()` : params optionnels `camera_path`, `interpolation`, `fov_degrees`. Calcul `focal_px = width / (2 * tan(fov/2))`
  - `generate_batch_frames()` : 2 appels `create_depth_conditioning` etendus pour lire et passer `camera_path`, `interpolation`, `fov_degrees` per-item
- `src/handler.py` : enrichir `pipeline_items` avec passthrough `camera_path`, `interpolation`, `fov_degrees` (conditionnel si present)
- `scripts/test_camera_path.py` : **NOUVEAU**. Unit tests quaternion math, Catmull-Rom, interpolate_camera_path, backward compat dolly-in
- `CLAUDE.md` : section "Depth IC-LoRA" etendue avec "Camera path 6-DOF". Mise a jour "API Input" avec 3 nouveaux champs optionnels

### Backward compatibility

**Stricte** : si `camera_path` absent ou None, genere path dolly-in 2-kf identique ancien comportement (translation=[0,0,δ], rotation=identity, interpolation=linear). Aucune regression pixel-perfect.

### Conventions

- Axes : X=droite, Y=haut, Z=avant (main droite)
- Quaternion : Hamilton [w, x, y, z]
- Temps t : normalise [0, 1]
- Catmull-Rom edge mode : clamped (duplication keyframe bord)
- FOV par defaut 60°

## 2026-04-11 — Phase A : Support multi-mode (t2v, i2v, i2v_depth, flf2v)

Refactor Python pour support 4 modes de generation avec auto-detection par champs presents. Suppression des CAMERA_PRESETS presets (legacy), activation depth via trigger explicite `camera_motion="depth"` uniquement.

### Fichiers modifies

- `src/prompts.py` : Suppression dict `CAMERA_PRESETS` (lignes 9-17)
- `src/pipeline.py` :
  - Import `image_conditionings_by_adding_guiding_latent` (support FLF2V last_image)
  - `warmup_embeddings()` : encode uniquement `DEFAULT_NEGATIVE_PROMPT` (pas de presets)
  - `_load_embeddings_cache()` : check legacy presets, regenere si detectes
  - `warmup_compile()` : prompt stub fixe `"A test scene for compilation warmup"` au lieu de presets
  - `encode_prompt()` : lookup presets supprime, docstring simplifiee
  - `generate_frames()` : signature `image_path: str | None = None`, param `mode`, refactor conditioning (replacement + guiding latent pour last_image)
  - `generate_batch_frames()` : signature items avec optionnel `image_path`, `mode`, `use_depth` per-item, gating depth par item
- `src/handler.py` :
  - `normalize_items()` : accepter items sans `image` (t2v)
  - Nouvelle fonction `detect_mode()` : auto-detection t2v/i2v/i2v_depth/flf2v
  - Download parallele : skip items sans `image`, gestion resolution cible sans premiere image
  - Groupement par prompt : direct (pas lookup CAMERA_PRESETS)
  - Ajout `mode` et `use_depth` par item aux pipeline_items
- `src/warmup_embeddings.py` : encode uniquement `DEFAULT_NEGATIVE_PROMPT`
- `CLAUDE.md` : ajout section "Modes de generation" (4 modes), mise a jour "API Input" et "Depth IC-LoRA"

### Backward compatibility

Default `mode="i2v_depth"` preserve comportement actuel (image requise, depth active). Items sans `use_depth` sont treated comme `False` (pas de depth).

### Trigger depth explicite

Depth IC-LoRA s'active seulement si `camera_motion == "depth"` sur un item I2V (pas d'image → t2v, image+last → flf2v, sinon i2v standard).

## 2026-04-07 — Fix Gemma config + auto-migration

### Fix : `rope_local_base_freq` manquant dans Gemma config

Le `config.json` de Gemma 3 12B sur le volume avait ete genere avec `transformers 4.50.0.dev0`, qui n'incluait pas le champ `rope_local_base_freq`. ltx-core commit `59ca828` accede a cet attribut dans `encoder_configurator.py:create_and_populate()` lors de l'encodage on-demand de prompts custom (cache miss). Les prompts pre-caches fonctionnaient car ils ne passent pas par le builder ltx-core.

**Symptome** : `AttributeError: 'Gemma3TextConfig' object has no attribute 'rope_local_base_freq'` sur tout prompt custom non present dans le cache embeddings.

**Corrections** :
- Ajout `rope_local_base_freq: 10000.0` dans `text_config` du `config.json` sur le volume (fix immediat)
- Ajout dans `start.sh` d'un re-save automatique du config via `AutoConfig.from_pretrained(local_files_only=True)` + `save_pretrained()` a chaque demarrage. Re-serialise le config avec les defaults de la version transformers installee, corrigeant automatiquement tout champ manquant lors de futures mises a jour

## 2026-04-02 — Version stable : Depth IC-LoRA + perf pipeline + guard idempotence

> **Tag stable** — Cette version est validee en production sur RunPod H100 80GB.
> 37 commits depuis 2026-03-14. Resume par phase ci-dessous.

### Phase 1 : Resolution tiers + static shapes + encodeur (mars 14-18)

- **3 tiers de resolution** : 540p (1-stage preview), 720p (2-stage, defaut), 1080p (2-stage). Suffixe `-portrait` pour 9:16
- **Static shapes par defaut** (`DYNAMIC_COMPILE=0`) : optimal pour les 8 shapes fixes du warmup, evite les recompilations Dynamo
- **Warmup compile toutes shapes** + `force_parameter_static_shapes` pour couvrir les 8 configs (landscape + portrait, stage 1 + 2)
- **Encodeur MP4 veryfast** (`video_encoder.py`) + pool post-processing aligne sur `MAX_BATCH` (encode + S3 upload en parallele du VAE decode)
- **Purge automatique cache ancien build** au demarrage (`CLEAN_OLD_CACHE=1`)
- **Audio.enabled=False unifie** partout (halve Dynamo specializations, deja documente)

### Phase 2 : Camera LoRA → Depth IC-LoRA (mars 18-28)

Migration complete du controle camera : abandon des camera LoRAs individuels (dolly-in, etc.) au profit du depth estimation + IC-LoRA Union Control.

- **IC-LoRA depth control** : `LTX-2.3-22b-IC-LoRA-Union-Control` fuse permanent dans le transformer au demarrage (654MB, pas de swap dynamique)
- **DA3METRIC-LARGE** : estimation profondeur metrique + sky segmentation (~1.64GB, offloadable CPU). Install depuis GitHub `ByteDance-Seed/depth-anything-3`
- **Parallax warp 2D** : forward splatting + z-buffer, `scale = d/(d-δ)`, normalisation per-frame [0,1]. Le ciel n'est pas warpe (sky_mask → 1.0)
- **Depth conditioning per-item** dans le batch : chaque item peut avoir son propre `camera_speed_ms`
- **VideoConditionByReferenceLatent** (downscale_factor=2) : conditioning Stage 1 uniquement, Stage 2 sans depth
- **Suppression systeme LoRA generique** : code mort (`ensure_lora`, `_unfuse_lora`, `LTXV_LORA_COMFY_RENAMING_MAP`) nettoye
- **Migration DA3METRIC-LARGE** : shift lineaire remplace reprojection 3D et log norm (plus stable, pas de focale requise)

Corrections depth :
- Dilatation morphologique des trous du forward splatting
- Clamp indices apres round dans scatter_reduce
- Gestion `intrinsics=None` dans DA3METRIC
- Normalisation log depth displacement → shift lineaire

### Phase 3 : Optimisations GPU zero-copy (mars 20-25)

- **load_safetensors direct GPU** : zero passage CPU pour les poids modele
- **Deltas LoRA sur GPU** : calcul matmul BF16 H100 + stockage GPU permanent (zero transfert CPU-GPU par requete)
- **Embeddings custom direct GPU** : cache RAM GPU, zero transfert par requete
- **Embeddings Gemma stockes GPU** dans le cache RAM
- **build_hash base sur pip freeze** uniquement (plus stable que file hash)
- **Triton cache versionne** + mega cache sauvegarde au warmup (pas apres 1er job)

### Phase 4 : Warmup depth + guard idempotence (mars 28 - avril 2)

- **Warmup compile avec depth conditioning** pour les shapes Stage 1 (evite recompilation Dynamo au 1er job avec depth)
- **VAE_DYNAMIC_COMPILE** : variable independante du transformer, fallback sur `DYNAMIC_COMPILE`
- **Guard idempotence RunPod** : `_completed_jobs` set intercepte les double delivery du SDK 1.8.2 (at-least-once). Le SDK re-delivre le meme job ~1s apres completion lors des cold starts (job marine dans la queue sans ACK)
- **Lock threading PyAV** pour eviter crash concurrent libx264 dans le pool post-processing
- **dtype bf16 pour dummy depth video** du warmup (coherence avec inference reelle)

### Fichiers principaux modifies

| Fichier | Modifications |
|---------|---------------|
| `src/handler.py` | Guard idempotence, lock PyAV, resolution tiers, depth per-item batch, pool post-processing |
| `src/pipeline.py` | Depth IC-LoRA, parallax warp 2D, DA3METRIC, static shapes, warmup depth, embeddings GPU, LoRA GPU |
| `src/video_encoder.py` | Nouvel encodeur MP4 veryfast (x264) |
| `src/start.sh` | Purge cache, VAE_DYNAMIC_COMPILE, rm -f |
| `src/prompts.py` | Presets camera texte |
| `Dockerfile` | video_encoder.py, DA3METRIC install |
| `CLAUDE.md` | Depth IC-LoRA, env vars, gotchas |

### Etat de la production (2026-04-02)

- **RunPod H100 80GB** : stable, cold start ~4 min (warmup compile 8 shapes + depth)
- **Performances warm** : ~7-8s/item 720p 25f, ~15s/item 720p 49f
- **Batch** : jusqu'a 9 items/batch (MAX_BATCH), regroupe par prompt
- **Depth conditioning** : parallax warp 2D stable, camera_speed_ms 0.3-1.0 m/s
- **S3 upload** : OVH SBG, concurrent avec VAE decode via ThreadPool
- **Guard idempotence** : double delivery RunPod intercepte sans impact

### Problemes connus (non bloquants)

- **Double delivery RunPod** : le SDK re-delivre le meme job ~1s apres completion sur cold start. Le guard `_completed_jobs` le skip, mais le 2nd `/job-done` envoie `{"images": []}` a RunPod. Si RunPod ecrase l'output reel, le polling Elixir (5s interval) peut recevoir le resultat vide. Fix envisage : cache du resultat reel dans le guard + `concurrency_modifier=2` pour prefetch (stash `prefetch concurrency + gpu_lock + cache idempotence`). Non deploye car necessite validation.
- **Auto-scale down entre jobs** : avec `concurrency=1` (defaut SDK), la queue locale tombe a 0 entre chaque job, l'auto-scaler RunPod peut couper le worker. Workaround : configurer `Min Workers=1` sur l'endpoint RunPod ou appliquer le stash prefetch.

---

## 2026-03-14 — Unification audio.enabled=False (halve Dynamo specializations)

### Contexte

Le pipeline avait 2 patterns Dynamo distincts : stage 1 passait `audio.enabled=False`, stage 2 utilisait le default (`enabled=True`). Dynamo compilait donc 2 chemins par bloc transformer (8 shapes x 2 patterns = 16 specialisations). Vu qu'on ne genere pas d'audio (videos immobilieres), unifier sur `audio.enabled=False` partout divise les specialisations par 2.

### Modifications

- **Warmup** — Suppression du flag `explicit_audio_disabled` et du branch if/else. Toutes les configs passent `enabled=True` (video) / `enabled=False` (audio). Tuple configs passe de 5 a 4 colonnes
- **Stage 2 single** — Remplacement de `simple_denoising_func` (qui utilisait `enabled=True` par defaut pour audio) par une `denoise_step_s2` inline avec `audio.enabled=False` explicite + `cudagraph_mark_step_begin()`
- **Stage 2 batch** — Ajout `enabled=True` / `enabled=False` explicites sur les appels `modality_from_latent_state`
- **Import `simple_denoising_func`** — Supprime (plus utilise)

### Fichiers modifies

| Fichier | Modifications |
|---------|---------------|
| `src/pipeline.py` | Warmup configs, stage 2 single denoise_fn, stage 2 batch enabled flags, suppression import simple_denoising_func |

### Impact attendu

- Warmup : 8 graphes au lieu de 16 (~50% du temps)
- Aucun impact qualite video (audio jamais utilise)

### Verification

1. Job 720p 2-stage — output identique, pas de regression
2. Job 1080p 2-stage — output identique, pas de regression
3. Warmup logs — 8 entries au lieu de 16 compilations Dynamo distinctes

---

## 2026-03-14 — Mega Cache artifacts + diagnostic compact + fix VAE batch recompilation

### Ajouts

- **Mega Cache (`save/load_cache_artifacts`)** — Bundle les 5 caches torch.compile (PGO, AOTAutograd, Inductor, Triton, Autotuning) en un blob portable sur le volume (`cache/compile_artifacts/{build_hash}.bin`). Load dans `__init__()` avant tout `torch.compile()`, save apres le 1er job reussi. Cold start 2+ devrait skip la recompilation (~18s/step transformer, ~170s VAE)
- **Timing VAE decode par item** — Log warning si un item prend >5s (diagnostic recompilation)
- **Cache counters enrichis** — Filtre elargi avec "autograd" en plus de "cache"
- **`BUILD_HASH` env var** — Exporte dans start.sh pour versionner les compile artifacts

### Corrections

- **`.contiguous()` sur VAE decode batch** — Le slice `video_state.latent[i:i+1]` d'un tensor batche peut avoir des strides/metadata differents du tensor single-item pour lequel le VAE a ete compile initialement → recompilation Dynamo de 347s. `.contiguous()` normalise le layout memoire. Cout negligeable (~quelques MB BF16)

### Ameliorations

- **Diagnostic Inductor compact** — Reduit de ~200 lignes a ~6 lignes. Supprime le dump des 419 keys inductor config, le listing des 61 sous-repertoires fxgraph, torch_key et system_info. Detail complet disponible en `LOG_LEVEL=debug`

### Fichiers modifies

| Fichier | Modifications |
|---------|---------------|
| `src/pipeline.py` | `_load_compile_artifacts()`, `save_compile_artifacts()`, `_compile_artifacts_path()`, diagnostic compact, `.contiguous()` VAE, timing VAE par item, counters enrichis |
| `src/handler.py` | Flag `_artifacts_saved` + trigger save apres 1er job |
| `src/start.sh` | `mkdir compile_artifacts`, `export BUILD_HASH` |

### Verification en production

1. **Cold start 1** : "No compile artifacts found" → recompilation normale → "Compile artifacts saved" apres 1er job
2. **Cold start 2+** : "Compile artifacts loaded" → step 0 transformer < 1s (au lieu de 18s)
3. **Batch=5** : gap VAE decode < 10s (au lieu de 347s)
4. **Logs** : diagnostic ~6 lignes au lieu de ~200
5. **Warm perf** : jobs single-item restent a 7-8s (pas de regression)

### Rollback

- Supprimer le fichier `cache/compile_artifacts/{build_hash}.bin` sur le volume pour forcer la recompilation
- `.contiguous()` est sans risque (copie memoire negligeable)

---

## 2026-03-09 — PyTorch 2.9 + SageAttention 2.2.0 + max-autotune + cache Triton

### Upgrade majeur

- **PyTorch `>=2.9`** — Support `torch.compile(dynamic=True)` + custom_op SA 2.2
- **SageAttention `>=2.2.0` from source** — Plus de pin commit `d1a57a5`, install depuis GitHub main. `@torch.library.custom_op` natif → zero graph breaks. Pas sur PyPI (max 1.0.6)
- **`compiler.disable(sageattn)` supprime** — SA 2.2.0 custom_op rend ce workaround inutile
- **`torch.compile(dynamic=True)`** sur transformer ET VAE decoder — evite les recompilations Dynamo entre stage 1 (half-res) et stage 2 (full-res)
- **Dynamo config** : `cache_size_limit=32`, `recompile_limit=16`, `automatic_dynamic_shapes=True`
- **`mode=max-autotune-no-cudagraphs`** quand SA actif (remplace `default`) — autotuning Triton (selection optimale des configs kernel) sans CUDA graphs (incompatibles avec kernels SA). Premiere execution lente (~5-10 min autotuning), runs suivants rapides via cache
- **Cache Triton + TorchInductor persistant** sur volume (`TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR` → `/runpod-volume/cache/triton/` et `/runpod-volume/cache/inductor/`)

### Corrections

- **SageAttention from source** — SA 2.2.0 absent de PyPI, Dockerfile corrige pour install depuis `git+https://github.com/thu-ml/SageAttention.git`
- **`g++` ajoute au runtime Docker** — TorchInductor (backend max-autotune) genere et compile des kernels C++ au runtime. `g++` manquait dans le stage runtime → `InvalidCxxCompiler` au premier job

### Fichiers modifies

| Fichier | Modifications |
|---------|---------------|
| `Dockerfile` | PyTorch >=2.9, SA 2.2.0 from source (git+), g++ dans runtime |
| `src/pipeline.py` | dynamic=True, max-autotune-no-cudagraphs, suppression compiler.disable |
| `src/start.sh` | mkdir cache/triton + cache/inductor, export TRITON_CACHE_DIR + TORCHINDUCTOR_CACHE_DIR |
| `CLAUDE.md` | Mise a jour mode compile + env vars cache |

### Rollback

- `COMPILE_MODE=default` → retour au mode sans autotuning
- `SAGE_ATTENTION=0` → rollback complet vers SDPA (+ `COMPILE_MODE=reduce-overhead`)

---

## 2026-03-09 — Optimisations torch.compile + SageAttention + env vars

### Breaking changes

Aucun. Tous les defauts sont retro-compatibles.

### Corrections

- **SageAttention pin commit `d1a57a5`** — Dockerfile utilise `--filter=blob:none` + `git checkout d1a57a5` au lieu de `--depth 1`. Ce commit inclut le fix SM90 (issue #320).
- **`torch.compiler.disable(sageattn)` conserve** (commit `99721dc`) — Le plan initial supposait que SA 2.2.0 (`d1a57a5`) supportait `custom_op + register_fake` pour torch.compile natif. En production, les extensions pybind11 (`transpose_pad_permute_cuda`, `scale_fuse_quant_cuda`) restent opaques a Dynamo → graph breaks → CUDA graphs vides avec `reduce-overhead` → output noise. Fix : `compiler.disable(sageattn)` restaure, `mode=default` force quand SA actif.
- **`COMPILE_MODE` env var** — Configurable uniquement quand SA desactive (`SAGE_ATTENTION=0`). Quand SA actif, force `mode=default` (graph breaks incompatibles avec CUDA graphs).

### Ajouts

- **`automatic_dynamic_shapes`** — `torch._dynamo.config.automatic_dynamic_shapes = True` avant `torch.compile`. Dynamo generalise les shapes au premier passage → les jobs suivants avec batch_size ou resolution differents ne declenchent plus de recompilation (~25s economises par job).
- **`COMPILE_MODE` env var** — Mode torch.compile configurable (defaut `reduce-overhead`). Valeurs acceptees : `default`, `reduce-overhead`, `max-autotune`, `max-autotune-no-cudagraphs`. Validation avec fallback `reduce-overhead` si invalide.
- **`SAMPLER` env var** — Stepper de denoising configurable (defaut `euler`). `res2s` active `Res2sDiffusionStep` (integration second ordre midpoint). Applique dans `generate_frames()` et `generate_batch_frames()`.
- **`VAE_TILING` env var** — Tiled VAE decode optionnel (defaut `0`). `1` active `TilingConfig.default()` dans `vae_decode_video()` pour reduire la VRAM en 1080p. Risque de ghosting temporal (ComfyUI #11767). Applique dans `generate_frames()` et `generate_batch_frames()`.
- **runpod `>=1.8`** — Upgrade minimum pour lazy-load boto3/fastapi/pydantic → cold start plus rapide.

### Suppressions

- **`SAGE_COMPILE_DISABLE` env var** — Jamais implemente dans le code, supprime de la documentation. Remplace par `COMPILE_MODE`.

### Fichiers modifies

| Fichier | Modifications |
|---------|---------------|
| `Dockerfile` | Pin SageAttention commit `d1a57a5`, `--filter=blob:none` |
| `requirements.txt` | `runpod>=1.8,<2.0` |
| `src/pipeline.py` | Imports (Res2sDiffusionStep, TilingConfig), compiler.disable conserve, automatic_dynamic_shapes, COMPILE_MODE (conditionnel SA), SAMPLER, VAE_TILING |
| `CLAUDE.md` | Suppression SAGE_COMPILE_DISABLE, ajout COMPILE_MODE + SAMPLER + VAE_TILING, mise a jour torch.compile |

### Tests requis (deploiement H100)

1. Build Docker — SageAttention SM90 compile OK
2. 2 jobs consecutifs meme resolution → step 0 du Job 2 < 5s (pas de recompilation Dynamo)
3. SA + mode=default — batch de 10 images de reference avec PSNR/SSIM vs baseline
4. Single 720p — output non-bruit, qualite identique baseline
5. 2-stage 1080p — qualite identique baseline
6. Batch 2 items — denoising batche OK

### Tests optionnels

7. `COMPILE_MODE=max-autotune-no-cudagraphs` — cold start 5-10 min sans cache Triton
8. `SAMPLER=res2s` — test A/B PSNR/SSIM sur 10+ images vs euler
9. `VAE_TILING=1` en 1080p — verifier pas de ghosting temporal

### Rollback

Chaque fonctionnalite est controlable par env var :
- `COMPILE_MODE=default` → retour au mode pre-fix
- `SAMPLER=euler` → stepper Euler (defaut)
- `VAE_TILING=0` → pas de tiling (defaut)
- `SAGE_ATTENTION=0` → rollback complet vers SDPA
