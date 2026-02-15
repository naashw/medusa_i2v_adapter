# CLAUDE.md - Medusa I2V ComfyUI Project

## Projet

Pipeline ComfyUI Image-to-Video utilisant LTX-2 19B avec effet camera dolly.
Objectif : generation rapide de videos dolly a partir d'images, qualite correcte.

## Contexte Technique

- **Runtime** : ComfyUI sur RunPod (GPU cloud)
- **Modele principal** : LTX-2 19B (FP8)
- **Text encoder** : Gemma 3 12B (FP8, CPU)
- **Approche** : Pipeline 2 passes (generation half-res + refinement full-res)
- **Output** : H264 MP4, 24fps

## Structure

- `Dockerfile` — image Docker (build context = ce dossier)
- `docker-compose.yml` — lancement local avec GPU
- `src/start.sh` — script de demarrage (telecharge modeles + lance ComfyUI)
- `src/extra_model_paths.yaml` — template des chemins modeles
- `workflows/` — 8 workflows actifs (JSON ComfyUI)
- `scripts/` — 5 scripts utilitaires (test, envoi, conversion)
- `docs/` — documentation et exemples
- `test-data/` — images de test

## Fichier Principal

`workflows/medusa_i2v_v5_fast_api.json` — workflow API le plus rapide.

## Etat Actuel

- Pass 1 fonctionne (8 steps Euler, distilled + camera LoRA)
- Pass 2 deconnectee (latent upscaler pas lie a Pass 1)
- I2V Adapter bypasse (mode 4)
- Resolution adaptative OK
- Output ~1 seconde (25 frames)

## Conventions

- Prefixe output : `medusa_i2v`
- Les workflows sont des fichiers JSON ComfyUI standard
- Les noeuds importants ont des titres avec emojis pour reperage visuel
- Le noeud camera LoRA est marque "SWAP ICI" pour changement rapide d'effet

## Points d'Attention

- Ne pas modifier la structure des liens sans verifier les IDs dans le JSON
- Le text encoder DOIT rester sur CPU (VRAM insuffisante pour tout charger sur GPU)
- Le VAE decode doit rester en mode tiled pour eviter les OOM
