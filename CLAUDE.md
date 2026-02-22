# CLAUDE.md - Medusa I2V (ltx-pipelines)
<!-- last-mirror-test: 2026-02-21 -->

## Projet

Pipeline Image-to-Video utilisant LTX-2 19B avec effet camera dolly.
Inference directe via ltx-pipelines (sans ComfyUI).
Objectif : generation rapide de videos dolly a partir d'images, qualite correcte.

## Contexte Technique

- **Runtime** : ltx-pipelines (Python direct) sur RunPod Serverless (H100 80GB, HBM3)
- **Modele principal** : LTX-2 19B AV model (FP8, ~22GB VRAM avec 3 LoRAs fusionnees)
- **Text encoder** : Gemma 3 12B (BF16, CPU — ~24GB RAM, format HuggingFace)
- **Approche** : Pipeline 1 passe, 720p natif, 8 steps Euler distilled, audio skip
- **Output** : H264 MP4, 24fps, ~1 seconde (25 frames)
- **PyTorch** : ~2.7 (requis par ltx-core)
- **Packages** : ltx-core + ltx-pipelines (depuis Lightricks/LTX-2)

## Structure

- `Dockerfile` — image Docker multi-stage (devel builder -> runtime), sans ComfyUI
- `docker-compose.yml` — lancement local avec GPU
- `src/start.sh` — script de demarrage (telecharge modeles + lance handler)
- `src/warmup_embeddings.py` — warmup embeddings low-RAM (safe_open direct, sans ModelLedger)
- `src/pipeline.py` — classe MedusaPipeline (inference ltx-pipelines)
- `src/handler.py` — handler RunPod serverless (API simplifiee)
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
    "image_strength": 1.0
  }
}
```

Cameras supportees : dolly-in, dolly-out, dolly-left, dolly-right, jib-down, jib-up, static.

## Architecture Pipeline

- **MedusaPipeline** : encapsule ltx-pipelines avec gestion lifecycle modeles
  - Video encoder persistent en VRAM (~1GB)
  - Video decoder persistent en VRAM (~2GB)
  - Transformer cache par camera LoRA, rebuild standard via ModelLedger si camera change (~5-8s)
  - 3 LoRAs fusionnees a chaque build : distilled + I2V + camera
  - Embeddings pre-caches sur disque (generes par warmup_embeddings.py)
- **Ordre d'init** : warmup embeddings (process isole) → transformer (dolly-in) → video encoder → video decoder
- **warmup_embeddings.py** : charge uniquement les 59 cles TE via safe_open (2.7GB) + Gemma `low_cpu_mem_usage=True`. Peak ~35GB.
- **Audio skip** : `skip_step=99` sur audio guider → audio compute seulement au step 0/8
- **CFG desactive** : `cfg_scale=1.0, stg_scale=0.0` → 1 seul forward par step
- **Sigmas distilled** : 8 steps hardcodes (DISTILLED_SIGMA_VALUES)

## Conventions

- Prefixe output : `medusa_i2v`
- Reponse API : `images[]` avec `s3_key` + `volume_path`
- Input images : supporte URL https OU base64
- Output videos : sauvegardees dans `/runpod-volume/output/{job_id}/`
- Dedup cache par hash dans `/runpod-volume/cache/dedup/`
- Embeddings cache dans `/runpod-volume/cache/embeddings/`

## Points d'Attention

- Le text encoder Gemma DOIT etre format HuggingFace (pas Comfy-Org single file)
- num_frames doit etre k*8+1 (ex: 25, 49, 97)
- height/width doivent etre multiples de 32
- `transformers` DOIT etre pince `>=4.52,<5.0` (v5 supprime `rope_local_base_freq` de Gemma3TextConfig)
- `huggingface-cli` n'est pas dans l'image Docker, utiliser `huggingface_hub.snapshot_download()` a la place
- `start.sh` lance le warmup avec `LD_PRELOAD=""` pour desactiver tcmalloc (inutile sur process ephemere)
