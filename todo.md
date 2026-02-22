# Optimisations pipeline - TODO

## Fait

- [x] Sampling 4 steps au lieu de 8 (node 31) — gain ~5s
- [x] VAE tile_size 768 au lieu de 512 (node 60) — gain potentiel sur decode

## A faire

### torch.compile (VAE + transformer)
- ComfyUI 0.13 a un noeud built-in `TorchCompileModel`
- Ajouter dans le workflow entre la chaine LoRA et le CFGGuider
- Aussi tester `TorchCompileVAE` si disponible
- Premiere compilation lente (~30-60s) mais amortie sur les jobs suivants
- Compatible avec le pre-warm (compiler pendant le warmup)
- Potentiel : ~3-5s sur le VAE, plus sur le transformer

### comfy_kitchen backends (cuda/triton)
- Les logs montrent cuda et triton backends disponibles mais desactives
- Chercher comment activer (env var, CLI flag, pip install)
- Modifier `src/start.sh` si applicable

### S3 externe pour les outputs
- Remplacer le volume path par upload direct vers S3 (R2, AWS)
- Le wrapper a deja eu le code `rp_upload.upload_image()` — a reactiver
- Env vars : `BUCKET_ENDPOINT_URL`, `BUCKET_ACCESS_KEY_ID`, `BUCKET_SECRET_ACCESS_KEY`

## Pas pratique

- Pipeline async (overlap VAE decode N / encode N+1) — refonte architecturale
- Batching (batch_size=2) — pas assez de VRAM (23/24 GB utilises)
