# Medusa I2V - ComfyUI + LTX-2 19B

Pipeline Image-to-Video avec effets camera dolly, base sur ComfyUI et LTX-2 19B.

## Structure

```
medusa_i2v_adapter/
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── src/
│   ├── start.sh                  # Script de demarrage
│   └── extra_model_paths.yaml    # Template chemins modeles
├── workflows/                    # 8 workflows actifs
│   ├── medusa_i2v_1pass_fast.json
│   ├── medusa_i2v_1pass_upscale.json
│   ├── medusa_i2v_1pass_upscale_clean.json
│   ├── medusa_i2v_1pass_upscale_api.json
│   ├── medusa_i2v_2pass_adaptive.json
│   ├── medusa_i2v_v2_spatial_api.json
│   ├── medusa_i2v_v3_native_api.json
│   └── medusa_i2v_v5_fast_api.json
├── scripts/                      # Utilitaires
│   ├── check-status.sh
│   ├── send-workflow.sh
│   ├── convert_ui_to_api.py
│   ├── test-v2-spatial.sh
│   └── run-test.sh
├── docs/
│   ├── WORKFLOWS.md
│   └── example-request.json
└── test-data/
    └── images_test_immo/
```

## Docker

### Build

```bash
DOCKER_BUILDKIT=1 docker build -t medusa-i2v .
```

### Run (GPU Pod)

```bash
docker run --gpus all -p 8188:8188 -p 8888:8888 -v /workspace:/workspace medusa-i2v
```

### Run (Serverless)

```bash
docker run --gpus all -e SERVERLESS=true -v /workspace:/workspace medusa-i2v
```

### Docker Compose

```bash
docker compose up
```

## Workflows

| Workflow | Description |
|----------|-------------|
| `medusa_i2v_v5_fast_api.json` | API rapide (recommande) |
| `medusa_i2v_1pass_fast.json` | 1 passe rapide (~1s) |
| `medusa_i2v_1pass_upscale.json` | 1 passe + upscale |
| `medusa_i2v_1pass_upscale_clean.json` | 1 passe + upscale (nettoye) |
| `medusa_i2v_1pass_upscale_api.json` | 1 passe + upscale (API) |
| `medusa_i2v_2pass_adaptive.json` | 2 passes resolution adaptative |
| `medusa_i2v_v2_spatial_api.json` | v2 spatial upscaler (API) |
| `medusa_i2v_v3_native_api.json` | v3 noeuds natifs (API) |

## Modeles

Les modeles sont telecharges automatiquement au runtime sur le network volume :
- LTX-2 19B (FP8) — checkpoint principal
- Gemma 3 12B (FP8) — text encoder
- LoRA distilled + camera control
- Spatial/temporal upscalers
