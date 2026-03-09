# Changelog — Medusa I2V

## 2026-03-09 — Optimisations torch.compile + SageAttention + env vars

### Breaking changes

Aucun. Tous les defauts sont retro-compatibles.

### Corrections

- **SageAttention pin commit `d1a57a5`** — Dockerfile utilise `--filter=blob:none` + `git checkout d1a57a5` au lieu de `--depth 1`. Ce commit inclut le fix SM90 (issue #320) et le support `custom_op + register_fake` pour torch.compile natif.
- **Suppression `torch.compiler.disable(sageattn)`** — SA 2.2.0+ gere nativement torch.compile via `custom_op`. Le wrapper `compiler.disable` qui forcait des graph breaks (et imposait `mode="default"`) est supprime.
- **Restauration `reduce-overhead`** — Le mode torch.compile n'est plus conditionne par SageAttention. `reduce-overhead` est le defaut pour tous les cas.

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
| `src/pipeline.py` | Imports (Res2sDiffusionStep, TilingConfig), suppression compiler.disable, automatic_dynamic_shapes, COMPILE_MODE, SAMPLER, VAE_TILING |
| `CLAUDE.md` | Suppression SAGE_COMPILE_DISABLE, ajout COMPILE_MODE + SAMPLER + VAE_TILING, mise a jour torch.compile |

### Tests requis (deploiement H100)

1. Build Docker — SageAttention SM90 compile OK
2. 2 jobs consecutifs meme resolution → step 0 du Job 2 < 5s (pas de recompilation Dynamo)
3. SA 2.2.0 + reduce-overhead — batch de 10 images de reference avec PSNR/SSIM avant/apres
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
