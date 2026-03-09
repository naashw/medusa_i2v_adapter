# Changelog — Medusa I2V

## 2026-03-09 — PyTorch 2.9 + SageAttention 2.2.0 + max-autotune + cache Triton

### Upgrade majeur

- **PyTorch `>=2.9`** — Support `torch.compile(dynamic=True)` + custom_op SA 2.2
- **SageAttention `>=2.2.0` via pip** — Plus de build from source, plus de pin commit `d1a57a5`. `@torch.library.custom_op` natif → zero graph breaks
- **`compiler.disable(sageattn)` supprime** — SA 2.2.0 custom_op rend ce workaround inutile
- **`torch.compile(dynamic=True)`** sur transformer ET VAE decoder — evite les recompilations Dynamo entre stage 1 (half-res) et stage 2 (full-res)
- **Dynamo config** : `cache_size_limit=32`, `recompile_limit=16`, `automatic_dynamic_shapes=True`
- **`mode=max-autotune-no-cudagraphs`** quand SA actif (remplace `default`) — autotuning Triton (selection optimale des configs kernel) sans CUDA graphs (incompatibles avec kernels SA). Premiere execution lente (~5-10 min autotuning), runs suivants rapides via cache
- **Cache Triton + TorchInductor persistant** sur volume (`TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR` → `/runpod-volume/cache/triton/` et `/runpod-volume/cache/inductor/`)

### Fichiers modifies

| Fichier | Modifications |
|---------|---------------|
| `Dockerfile` | PyTorch >=2.9, SageAttention >=2.2.0 pip |
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
