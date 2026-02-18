"""Script standalone de warmup embeddings.

Charge le text encoder Gemma 3 12B sur CPU, encode tous les prompts camera,
sauvegarde le cache sur le volume, puis termine.
L'OS recupere 100% de la RAM a la fin du process.

Usage :
    CUDA_VISIBLE_DEVICES="" python /app/warmup_embeddings.py

Variables d'environnement :
    MODELS_DIR  : chemin vers les modeles (defaut: /workspace/models)
    CACHE_DIR   : chemin vers le cache (defaut: /workspace/cache)
"""

from __future__ import annotations

import logging
import os
import resource
import sys

import torch

# Importer les constantes depuis pipeline.py (pas de duplication)
sys.path.insert(0, os.path.dirname(__file__) if "__file__" in dir() else "/app")
from pipeline import CAMERA_PROMPTS, DEFAULT_NEGATIVE_PROMPT  # noqa: E402

from ltx_core.text_encoders.gemma import encode_text
from ltx_pipelines.utils import ModelLedger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [warmup] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("warmup")


def log_ram_mb() -> int:
    """Retourne et logue la RAM max utilisee (en MB)."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # Linux : ru_maxrss en kilobytes
    mb = usage.ru_maxrss // 1024
    log.info("RAM pic: %d MB", mb)
    return mb


def main() -> int:
    models_dir = os.environ.get("MODELS_DIR", "/workspace/models")
    cache_dir = os.environ.get("CACHE_DIR", "/workspace/cache")
    cache_path = os.path.join(cache_dir, "embeddings", "embeddings_cache.pt")

    # Si le cache existe deja, skip immediat
    if os.path.isfile(cache_path):
        log.info("Cache embeddings existant: %s — skip", cache_path)
        return 0

    log.info("Generation du cache embeddings: %s", cache_path)

    checkpoint_path = os.path.join(models_dir, "checkpoints", "ltx-2-19b-dev-fp8.safetensors")
    gemma_root = os.path.join(models_dir, "text_encoders", "gemma-3-12b-it")

    if not os.path.isfile(checkpoint_path):
        log.error("Checkpoint introuvable: %s", checkpoint_path)
        return 1
    if not os.path.isdir(gemma_root):
        log.error("Gemma introuvable: %s", gemma_root)
        return 1

    log.info("Checkpoint: %s", checkpoint_path)
    log.info("Gemma: %s", gemma_root)

    # ModelLedger CPU minimal : checkpoint + gemma uniquement
    # Pas de loras, pas de quantization → seul Gemma est charge en memoire
    log.info("Chargement text encoder Gemma 3 12B sur CPU...")
    cpu_ledger = ModelLedger(
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
        checkpoint_path=checkpoint_path,
        gemma_root_path=gemma_root,
    )
    text_encoder = cpu_ledger.text_encoder()
    log_ram_mb()

    # Encoder tous les prompts : 7 cameras + 1 negative
    all_prompts = list(CAMERA_PROMPTS.values()) + [DEFAULT_NEGATIVE_PROMPT]
    all_keys = list(CAMERA_PROMPTS.keys()) + ["_negative"]
    log.info("Encoding %d prompts...", len(all_prompts))
    results = encode_text(text_encoder, prompts=all_prompts)
    log_ram_mb()

    # Construire et sauvegarder le cache
    cache_data: dict[str, dict[str, torch.Tensor]] = {}
    for key, (v_ctx, a_ctx) in zip(all_keys, results):
        cache_data[key] = {"video": v_ctx.cpu(), "audio": a_ctx.cpu()}

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(cache_data, cache_path)
    log.info("Cache sauvegarde: %s (%d prompts)", cache_path, len(cache_data))

    final_ram = log_ram_mb()
    log.info(
        "Warmup termine. RAM pic: %d MB. Process va terminer (OS recupere la RAM).",
        final_ram,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
