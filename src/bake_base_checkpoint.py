"""
bake_base_checkpoint.py — Pre-fusionne distilled + I2V LoRAs dans le checkpoint de base.

Genere ltx-2-19b-dev-fp8-baked.safetensors (+.json) sur le volume.
Lance depuis start.sh avant handler.py.

Usage:
    MODELS_DIR=/runpod-volume/models python /app/bake_base_checkpoint.py
"""
from __future__ import annotations

import json
import logging
import os

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP
from ltx_core.loader.fuse_loras import _fuse_delta_with_cast_fp8, _prepare_deltas
from ltx_core.loader.primitives import LoraStateDictWithStrength
from ltx_core.loader.sft_loader import SafetensorsModelStateDictLoader

logging.basicConfig(level=logging.INFO, format="[bake] %(message)s")
log = logging.getLogger("bake")

MODELS_DIR = os.environ.get("MODELS_DIR", "/runpod-volume/models")
CHECKPOINT_FILENAME = "ltx-2-19b-dev-fp8.safetensors"
BAKED_FILENAME = "ltx-2-19b-dev-fp8-baked.safetensors"
BAKE_VERSION = "1"

DISTILLED_LORA_STRENGTH = 0.7
I2V_ADAPTER_STRENGTH = 0.8


def _checkpoint_path() -> str:
    return os.path.join(MODELS_DIR, "checkpoints", CHECKPOINT_FILENAME)


def _baked_path() -> str:
    return os.path.join(MODELS_DIR, "checkpoints", BAKED_FILENAME)


def _baked_meta_path() -> str:
    return _baked_path() + ".json"


def _distilled_lora_path() -> str:
    return os.path.join(MODELS_DIR, "loras", "ltx-2-19b-distilled-lora-384.safetensors")


def _i2v_adapter_path() -> str:
    return os.path.join(MODELS_DIR, "loras", "LTX-2-Image2Vid-Adapter.safetensors")


def is_baked_valid() -> bool:
    """Verifie si le baked checkpoint existe et correspond aux strengths actuels."""
    if not (os.path.isfile(_baked_path()) and os.path.isfile(_baked_meta_path())):
        return False
    try:
        with open(_baked_meta_path()) as f:
            meta = json.load(f)
        return (
            meta.get("distilled_strength") == str(DISTILLED_LORA_STRENGTH)
            and meta.get("i2v_strength") == str(I2V_ADAPTER_STRENGTH)
            and meta.get("bake_version") == BAKE_VERSION
        )
    except Exception:
        return False


def main() -> None:
    if is_baked_valid():
        log.info("Baked checkpoint deja valide, skip: %s", BAKED_FILENAME)
        return

    log.info("Generation du baked checkpoint (distilled + I2V pre-fusionnes)...")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    log.info("Device: %s", device)

    # Charger les LoRAs en RAM CPU
    loader = SafetensorsModelStateDictLoader()
    log.info("Chargement distilled LoRA...")
    distilled_sd = loader.load(
        [_distilled_lora_path()],
        sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
        device=torch.device("cpu"),
    )
    log.info("Chargement I2V adapter...")
    i2v_sd = loader.load(
        [_i2v_adapter_path()],
        sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
        device=torch.device("cpu"),
    )

    lora_items = [
        LoraStateDictWithStrength(distilled_sd, DISTILLED_LORA_STRENGTH),
        LoraStateDictWithStrength(i2v_sd, I2V_ADAPTER_STRENGTH),
    ]

    ckpt_path = _checkpoint_path()
    merged: dict[str, torch.Tensor] = {}
    original_metadata: dict[str, str] = {}

    log.info("Lecture checkpoint source: %s", CHECKPOINT_FILENAME)
    with safe_open(ckpt_path, framework="pt", device="cpu") as f:
        original_metadata = f.metadata() or {}
        keys = list(f.keys())
        log.info("Nombre de cles: %d", len(keys))

        for i, key in enumerate(keys):
            if i % 200 == 0:
                log.info("Progression: %d/%d", i, len(keys))

            # Skip les .weight_scale keys (force le path cast FP8, pas scaled FP8)
            if key.endswith(".weight_scale"):
                continue

            # Cles non-.weight : copier tel quel (bias, norm, etc.)
            if not key.endswith(".weight"):
                merged[key] = f.get_tensor(key)
                continue

            # Cle .weight : calculer le delta LoRA
            # Les cles checkpoint = "model.diffusion_model.XXX.weight"
            # Les cles LoRA (apres RENAMING_MAP) = "XXX.lora_A.weight"
            # → strip le prefixe pour le matching
            stripped = key.replace("model.diffusion_model.", "")
            weight = f.get_tensor(key)

            delta = _prepare_deltas(lora_items, stripped, torch.bfloat16, device)
            if delta is None:
                # Aucun LoRA pour cette cle, copier tel quel
                merged[key] = weight
                continue

            if weight.dtype == torch.float8_e4m3fn:
                # Fusionner avec cast FP8 (kernel Triton si GPU disponible)
                weight_dev = weight.to(device)
                result = _fuse_delta_with_cast_fp8(
                    delta, weight_dev, stripped, torch.float8_e4m3fn, device
                )
                merged[key] = result[stripped].cpu()
            else:
                # Poids non-FP8 : addition directe
                merged[key] = (weight.to(torch.bfloat16) + delta.cpu()).to(weight.dtype)

    out_path = _baked_path()
    log.info("Sauvegarde baked checkpoint: %s", BAKED_FILENAME)
    # La metadata originale (config transformer) est obligatoire pour ModelLedger
    save_file(merged, out_path, metadata=original_metadata)

    distilled_size = os.path.getsize(_distilled_lora_path())
    i2v_size = os.path.getsize(_i2v_adapter_path())
    meta_json = {
        "distilled_strength": str(DISTILLED_LORA_STRENGTH),
        "i2v_strength": str(I2V_ADAPTER_STRENGTH),
        "distilled_size": str(distilled_size),
        "i2v_size": str(i2v_size),
        "bake_version": BAKE_VERSION,
    }
    with open(_baked_meta_path(), "w") as mf:
        json.dump(meta_json, mf)

    size_gb = os.path.getsize(out_path) / 2**30
    log.info("Bake termine: %s (%.1f GB)", BAKED_FILENAME, size_gb)


if __name__ == "__main__":
    main()
