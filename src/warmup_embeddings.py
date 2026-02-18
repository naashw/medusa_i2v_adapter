"""Script standalone de warmup embeddings (low-RAM version).

Charge le text encoder Gemma 3 12B directement (sans ModelLedger)
pour eviter le chargement du checkpoint complet de 27 GB en RAM.

Peak RAM : ~35 GB (vs ~106 GB avec ModelLedger)
Compatible RunPod serverless (57 GB RAM dispo).

Usage :
    python /app/warmup_embeddings.py

Variables d'environnement :
    MODELS_DIR  : chemin vers les modeles (defaut: /runpod-volume/models)
    CACHE_DIR   : chemin vers le cache (defaut: /runpod-volume/cache)
"""

from __future__ import annotations

import gc
import json
import logging
import os
import resource
import sys

import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(__file__) if "__file__" in dir() else "/app")
from pipeline import CAMERA_PROMPTS, DEFAULT_NEGATIVE_PROMPT  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [warmup] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("warmup")


def ram_mb() -> int:
    """RAM pic du process en MB (Linux)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024


def build_text_encoder(checkpoint_path: str, gemma_root: str) -> torch.nn.Module:
    """Construit le AVGemmaTextEncoderModel sans ModelLedger.

    Charge seulement les 59 cles text encoder (2.7 GB) du checkpoint,
    puis Gemma depuis HuggingFace avec low_cpu_mem_usage=True.
    Peak RAM ~35 GB au lieu de ~106 GB.
    """
    from ltx_core.text_encoders.gemma.encoders.av_encoder import (
        AVGemmaTextEncoderModelConfigurator,
        create_and_populate,
    )
    from ltx_core.text_encoders.gemma.tokenizer import LTXVGemmaTokenizer
    from transformers import Gemma3ForConditionalGeneration

    # 1. Extraire seulement les 59 cles text encoder du checkpoint
    log.info("Extraction des cles text encoder du checkpoint...")
    f = safe_open(checkpoint_path, framework="pt")
    metadata = f.metadata()
    config = json.loads(metadata["config"])

    te_prefixes = (
        "text_embedding_projection",
        "model.diffusion_model.video_embeddings_connector",
        "model.diffusion_model.audio_embeddings_connector",
    )
    remapped: dict[str, torch.Tensor] = {}
    for key in f.keys():
        if key.startswith(te_prefixes):
            new_key = key
            new_key = new_key.replace(
                "text_embedding_projection.", "feature_extractor_linear."
            )
            new_key = new_key.replace(
                "model.diffusion_model.video_embeddings_connector.",
                "embeddings_connector.",
            )
            new_key = new_key.replace(
                "model.diffusion_model.audio_embeddings_connector.",
                "audio_embeddings_connector.",
            )
            remapped[new_key] = f.get_tensor(key).to(torch.bfloat16)
    del f
    gc.collect()
    log.info("59 cles TE extraites (2.7 GB). RAM: %d MB", ram_mb())

    # 2. Creer le modele shell sur meta device
    te_model = AVGemmaTextEncoderModelConfigurator.from_config(config["transformer"])
    log.info("Modele shell cree. RAM: %d MB", ram_mb())

    # 3. Charger Gemma depuis HuggingFace (low memory)
    log.info("Chargement Gemma 3 12B (low_cpu_mem_usage)...")
    gemma = Gemma3ForConditionalGeneration.from_pretrained(
        gemma_root,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    te_model.model = gemma
    log.info("Gemma charge. RAM: %d MB", ram_mb())

    # 4. Charger les connectors/projection
    missing, unexpected = te_model.load_state_dict(remapped, strict=False)
    del remapped
    gc.collect()
    log.info(
        "Connectors charges. Missing: %d (Gemma keys — attendu), Unexpected: %d",
        len(missing),
        len(unexpected),
    )

    # 5. Tokenizer
    te_model.tokenizer = LTXVGemmaTokenizer(gemma_root)

    # 6. RoPE setup
    te_model = create_and_populate(te_model)
    log.info("Text encoder pret. RAM: %d MB", ram_mb())

    return te_model


def main() -> int:
    models_dir = os.environ.get("MODELS_DIR", "/runpod-volume/models")
    cache_dir = os.environ.get("CACHE_DIR", "/runpod-volume/cache")
    cache_path = os.path.join(cache_dir, "embeddings", "embeddings_cache.pt")

    # Si le cache existe deja, skip immediat
    if os.path.isfile(cache_path):
        log.info("Cache embeddings existant: %s — skip", cache_path)
        return 0

    log.info("Generation du cache embeddings: %s", cache_path)

    checkpoint_path = os.path.join(
        models_dir, "checkpoints", "ltx-2-19b-dev-fp8.safetensors"
    )
    gemma_root = os.path.join(models_dir, "text_encoders", "gemma-3-12b-it")

    if not os.path.isfile(checkpoint_path):
        log.error("Checkpoint introuvable: %s", checkpoint_path)
        return 1
    if not os.path.isdir(gemma_root):
        log.error("Gemma introuvable: %s", gemma_root)
        return 1

    # Construire le text encoder (low RAM)
    te_model = build_text_encoder(checkpoint_path, gemma_root)

    # Encoder tous les prompts : 7 cameras + 1 negative
    from ltx_core.text_encoders.gemma import encode_text

    all_prompts = list(CAMERA_PROMPTS.values()) + [DEFAULT_NEGATIVE_PROMPT]
    all_keys = list(CAMERA_PROMPTS.keys()) + ["_negative"]
    log.info("Encoding %d prompts...", len(all_prompts))

    results = encode_text(te_model, prompts=all_prompts)
    log.info("Encoding termine. RAM: %d MB", ram_mb())

    # Construire et sauvegarder le cache
    cache_data: dict[str, dict[str, torch.Tensor]] = {}
    for key, (v_ctx, a_ctx) in zip(all_keys, results):
        cache_data[key] = {"video": v_ctx.cpu(), "audio": a_ctx.cpu()}

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(cache_data, cache_path)
    log.info("Cache sauvegarde: %s (%d prompts)", cache_path, len(cache_data))

    # Cleanup
    del te_model, results
    gc.collect()

    log.info("Warmup termine. RAM pic: %d MB", ram_mb())
    return 0


if __name__ == "__main__":
    sys.exit(main())
