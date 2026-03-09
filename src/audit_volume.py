"""
audit_volume.py — Verifie la presence des fichiers requis sur le network volume RunPod.

Listing leger jusqu a 3 niveaux de profondeur, sans calcul de taille (lent sur NFS).

Usage:
    python scripts/audit_volume.py [--volume /runpod-volume]
"""
from __future__ import annotations

import argparse
import os


def is_empty(path: str) -> bool:
    """Verifie si un dossier est vide (un seul listdir, pas de walk)."""
    try:
        return not bool(os.listdir(path))
    except (PermissionError, OSError):
        return False


def print_tree(path: str, max_depth: int = 3, depth: int = 0) -> None:
    """Affiche l arborescence jusqu a max_depth niveaux, sans calcul de taille."""
    if depth >= max_depth:
        return
    try:
        entries = sorted(os.listdir(path))
    except (PermissionError, OSError):
        return
    for entry in entries:
        full = os.path.join(path, entry)
        indent = "  " * depth
        if os.path.isdir(full):
            empty = " (vide)" if is_empty(full) else ""
            print(f"{indent}  [{entry}/]{empty}")
            print_tree(full, max_depth, depth + 1)
        else:
            print(f"{indent}  {entry}")


def audit_volume(volume_root: str) -> list[dict]:
    """Verifie la presence des fichiers/dossiers requis. Retourne la liste des manquants."""
    models_dir = os.path.join(volume_root, "models")

    required: dict[str, str] = {
        os.path.join(models_dir, "checkpoints", "ltx-2.3-22b-distilled.safetensors"):
            "Checkpoint LTX-2.3 22B Distilled BF16",
        os.path.join(models_dir, "upscalers", "ltx-2.3-spatial-upscaler-x2-1.0.safetensors"):
            "Spatial upscaler x2",
        os.path.join(models_dir, "upscalers", "ltx-2.3-temporal-upscaler-x2-1.0.safetensors"):
            "Temporal upscaler x2",
        os.path.join(models_dir, "text_encoders", "gemma-3-12b-it"):
            "Text encoder Gemma 3 12B",
        os.path.join(volume_root, "cache", "embeddings"):
            "Cache embeddings (warmup)",
        os.path.join(volume_root, "cache", "triton"):
            "Cache Triton (torch.compile)",
        os.path.join(volume_root, "cache", "inductor"):
            "Cache TorchInductor (torch.compile)",
    }

    missing = []
    print(f"\n{'='*60}")
    print(f"  AUDIT VOLUME: {volume_root}")
    print(f"{'='*60}\n")

    print("--- FICHIERS REQUIS ---\n")
    for path, label in required.items():
        exists = os.path.exists(path)
        if exists and os.path.isdir(path):
            empty = " (vide)" if is_empty(path) else ""
            status = f"[OK]{empty}"
        elif exists:
            status = "[OK]"
        else:
            status = "[MANQUANT]"
            missing.append({"path": path, "reason": label})
        name = os.path.basename(path.rstrip("/"))
        print(f"  {status:<12} {name}  — {label}")

    print(f"\n--- STRUCTURE VOLUME (3 niveaux) ---\n")
    print_tree(volume_root, max_depth=3)

    if missing:
        print(f"\n{'='*60}")
        print(f"  MANQUANTS CRITIQUES: {len(missing)}")
        for m in missing:
            print(f"  [!!] {m['path']}  ({m['reason']})")
    print(f"\n{'='*60}\n")

    return missing


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit leger du network volume RunPod")
    parser.add_argument("--volume", default="/runpod-volume", help="Chemin du volume")
    args = parser.parse_args()

    if not os.path.isdir(args.volume):
        print(f"ERREUR: Volume non trouve: {args.volume}")
        return

    audit_volume(args.volume)


if __name__ == "__main__":
    main()
