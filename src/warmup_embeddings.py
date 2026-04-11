"""Script standalone de warmup embeddings (GPU-accelerated).

Charge le text encoder Gemma 3 12B et encode les prompts camera.
Utilise le GPU si disponible pour accelerer l'encodage (~5-10s vs ~30-40s CPU).

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
from prompts import DEFAULT_NEGATIVE_PROMPT  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [warmup] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("warmup")


def ram_mb() -> int:
    """RAM pic du process en MB (Linux)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024


def vram_gb() -> str:
    """VRAM allouee en GB (ou 'N/A' si pas de GPU)."""
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 2**30
        return f"{alloc:.2f}GB"
    return "N/A"


def build_encoder_and_processor(checkpoint_path: str, gemma_root: str) -> tuple:
    """Construit GemmaTextEncoder + EmbeddingsProcessor sans ModelLedger.

    Charge seulement les cles embeddings processor du checkpoint,
    puis Gemma depuis HuggingFace. Utilise le GPU si disponible.
    Peak RAM ~35 GB au lieu de ~106 GB.

    Returns:
        (GemmaTextEncoder, EmbeddingsProcessor) prets a l'emploi sur GPU.
    """
    from ltx_core.text_encoders.gemma.encoders.encoder_configurator import (
        EmbeddingsProcessorConfigurator,
        GemmaTextEncoderConfigurator,
        create_and_populate,
    )
    from ltx_core.text_encoders.gemma.tokenizer import LTXVGemmaTokenizer
    from transformers import Gemma3ForConditionalGeneration

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Text encoder device: %s", device)

    # 1. Extraire les cles embeddings processor du checkpoint
    log.info("Extraction des cles embeddings processor du checkpoint...")
    f = safe_open(checkpoint_path, framework="pt")
    metadata = f.metadata()
    config = json.loads(metadata["config"])

    ep_prefixes = (
        "text_embedding_projection",
        "model.diffusion_model.video_embeddings_connector",
        "model.diffusion_model.audio_embeddings_connector",
    )
    remapped: dict[str, torch.Tensor] = {}
    for key in f.keys():
        if key.startswith(ep_prefixes):
            new_key = key
            new_key = new_key.replace(
                "text_embedding_projection.", "feature_extractor."
            )
            new_key = new_key.replace(
                "model.diffusion_model.video_embeddings_connector.",
                "video_connector.",
            )
            new_key = new_key.replace(
                "model.diffusion_model.audio_embeddings_connector.",
                "audio_connector.",
            )
            remapped[new_key] = f.get_tensor(key).to(torch.bfloat16)
    del f
    gc.collect()
    log.info("Cles embeddings processor extraites (%d). RAM: %d MB", len(remapped), ram_mb())

    # 2. Creer le text encoder shell sur meta device
    te = GemmaTextEncoderConfigurator.from_config(config["transformer"])
    log.info("Text encoder shell cree. RAM: %d MB", ram_mb())

    # 3. Charger Gemma depuis HuggingFace (GPU direct si disponible)
    log.info("Chargement Gemma 3 12B (device=%s)...", device)
    try:
        gemma = Gemma3ForConditionalGeneration.from_pretrained(
            gemma_root,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map=device,
        )
        log.info(
            "Gemma charge via device_map=%s. RAM: %d MB, VRAM: %s",
            device, ram_mb(), vram_gb(),
        )
    except Exception as e:
        log.warning("device_map=%s echoue (%s), fallback CPU + .to()", device, e)
        gemma = Gemma3ForConditionalGeneration.from_pretrained(
            gemma_root,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        if device == "cuda":
            gemma = gemma.to(device)
        log.info("Gemma charge (fallback). RAM: %d MB, VRAM: %s", ram_mb(), vram_gb())

    te.model = gemma
    log.info("Gemma assigne. RAM: %d MB", ram_mb())

    # 4. Tokenizer
    te.tokenizer = LTXVGemmaTokenizer(gemma_root)

    # 5. RoPE setup
    te = create_and_populate(te)

    # 6. Deplacer text encoder vers device
    if device == "cuda":
        te = te.to(device)
        log.info("Text encoder deplace vers %s. VRAM: %s", device, vram_gb())

    log.info("Text encoder pret. RAM: %d MB, VRAM: %s", ram_mb(), vram_gb())

    # 7. Creer l'embeddings processor (connectors + feature extractor)
    emb_proc = EmbeddingsProcessorConfigurator.from_config(config)
    missing, unexpected = emb_proc.load_state_dict(remapped, strict=False)
    del remapped
    gc.collect()
    log.info(
        "Embeddings processor charge. Missing: %d, Unexpected: %d",
        len(missing),
        len(unexpected),
    )

    # 8. Deplacer embeddings processor vers device (bfloat16 pour matcher Gemma)
    emb_proc = emb_proc.to(device=device, dtype=torch.bfloat16)
    if device == "cuda":
        log.info("Embeddings processor deplace vers %s. VRAM: %s", device, vram_gb())

    return te, emb_proc


def main() -> int:
    models_dir = os.environ.get("MODELS_DIR", "/runpod-volume/models")
    cache_dir = os.environ.get("CACHE_DIR", "/runpod-volume/cache")
    cache_path = os.path.join(cache_dir, "embeddings", "embeddings_cache.pt")

    # Si le cache existe deja, skip immediat
    if os.path.isfile(cache_path):
        log.info("Cache embeddings existant: %s — skip", cache_path)
        return 0

    log.info("Generation du cache embeddings: %s", cache_path)

    if torch.cuda.is_available():
        log.info("VRAM avant warmup: %s", vram_gb())

    checkpoint_path = os.path.join(
        models_dir, "checkpoints", "ltx-2.3-22b-distilled.safetensors"
    )
    gemma_root = os.path.join(models_dir, "text_encoders", "gemma-3-12b-it")

    if not os.path.isfile(checkpoint_path):
        log.error("Checkpoint introuvable: %s", checkpoint_path)
        return 1
    if not os.path.isdir(gemma_root):
        log.error("Gemma introuvable: %s", gemma_root)
        return 1

    # Construire text encoder + embeddings processor (GPU si disponible)
    te, emb_proc = build_encoder_and_processor(checkpoint_path, gemma_root)

    # Encoder uniquement le prompt negative
    all_prompts = [DEFAULT_NEGATIVE_PROMPT]
    all_keys = ["_negative"]
    log.info("Encoding %d prompts...", len(all_prompts))

    cache_data: dict[str, dict[str, torch.Tensor]] = {}
    for key, prompt in zip(all_keys, all_prompts):
        hidden_states, mask = te.encode(prompt)
        output = emb_proc.process_hidden_states(hidden_states, mask)
        cache_data[key] = {"video": output.video_encoding.cpu(), "audio": output.audio_encoding.cpu()}

    log.info("Encoding termine. RAM: %d MB, VRAM: %s", ram_mb(), vram_gb())

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(cache_data, cache_path)
    log.info("Cache sauvegarde: %s (%d prompts)", cache_path, len(cache_data))

    # Cleanup complet (CPU + GPU)
    del te, emb_proc, cache_data
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        log.info("VRAM apres cleanup: %s", vram_gb())

    log.info("Warmup termine. RAM pic: %d MB", ram_mb())
    return 0


if __name__ == "__main__":
    sys.exit(main())
