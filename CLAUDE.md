# CLAUDE.md - Medusa I2V ComfyUI Project

## Projet

Pipeline ComfyUI Image-to-Video utilisant LTX-2 19B avec effet camera dolly.
Objectif : generation rapide de videos dolly a partir d'images, qualite correcte.

## Contexte Technique

- **Runtime** : ComfyUI 0.13.0 sur RunPod Serverless (RTX 4090, 24GB VRAM)
- **Modele principal** : LTX-2 19B AV model (FP8, ~20GB VRAM)
- **Text encoder** : Gemma 3 12B (FP8, CPU — ~15GB RAM)
- **Approche** : Pipeline 1 passe, 720p natif, 8 steps Euler distilled
- **Output** : H264 MP4, 24fps, ~1 seconde (25 frames)
- **PyTorch** : 2.10.0+cu128

## Structure

- `Dockerfile` — image Docker multi-stage (devel builder -> runtime)
- `docker-compose.yml` — lancement local avec GPU
- `src/start.sh` — script de demarrage (telecharge modeles + lance ComfyUI)
- `src/handler_wrapper.py` — wrapper RunPod (URL->base64, sauvegarde volume, dedup cache)
- `src/extra_model_paths.yaml` — template des chemins modeles
- `workflows/` — 2 workflows actifs + 1 archive
- `scripts/` — scripts utilitaires (test, envoi, conversion)
- `docs/` — documentation et exemples
- `test-data/` — images de test

## Workflows

- `workflows/720p_native_1pass_api.json` — **production** : 720p natif, 1 passe, 8 steps
- `workflows/2pass_spatial_upscale_api.json` — ancien : 2 passes half-res + spatial upscale 2x
- `workflows/_unusable_720p_Q8_kernels_api.json` — archive : tentative Q8 (incompatible AV model)

## Etat Actuel

- 1 passe 720p natif fonctionne (8 steps Euler, distilled + camera LoRA)
- `--disable-smart-memory` actif (transformer reste en VRAM entre jobs)
- I2V Adapter actif, resolution adaptative OK
- Dedup cache par hash (meme input + meme workflow = skip)
- Q8 Kernels incompatible avec LTX-2 19B (architecture AV dual-stream)

## Conventions

- Prefixe output : `medusa_i2v`
- Les workflows sont au format API RunPod (workflow neste dans `input.workflow`)
- Le noeud camera LoRA est marque "SWAP ICI" pour changement rapide d'effet

## Points d'Attention

- Ne pas modifier la structure des liens sans verifier les IDs dans le JSON
- Le text encoder DOIT rester sur CPU (VRAM insuffisante pour tout charger sur GPU)
- Le VAE decode doit rester en mode tiled pour eviter les OOM
- Q8 Kernels (`q8_kernels`) ne supporte pas le modele AV (tuple dual-stream video+audio)
- ComfyUI-LTXVideo pinne au commit `82bd963c` (2026-02-11)
