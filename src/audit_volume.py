"""
audit_volume.py — Identifie les fichiers inutilises sur le network volume RunPod.

Scan /runpod-volume et compare avec la liste des fichiers requis par MedusaPipeline.

Usage:
    python scripts/audit_volume.py [--volume /runpod-volume] [--json]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil


def human_size(size_bytes: int) -> str:
    """Formate une taille en bytes en format lisible."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(size_bytes) < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def dir_size(path: str) -> int:
    """Calcule la taille totale d'un dossier recursivement."""
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
    return total


def audit_volume(volume_root: str) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """Scan le volume et classifie chaque element.

    Returns:
        (used, unused, dynamic, missing) — listes de dicts {path, size, reason}
    """
    models_dir = os.path.join(volume_root, "models")

    # --- Fichiers REQUIS (pipeline les utilise directement) ---
    required_files: dict[str, str] = {
        # Checkpoint principal (distilled BF16, fp8_cast au runtime)
        os.path.join(models_dir, "checkpoints", "ltx-2.3-22b-distilled.safetensors"):
            "Checkpoint LTX-2.3 22B Distilled BF16 (pipeline.py)",
        # Spatial upscaler
        os.path.join(models_dir, "upscalers", "ltx-2.3-spatial-upscaler-x2-1.0.safetensors"):
            "Spatial upscaler x2 (pipeline.py)",
        # Temporal upscaler
        os.path.join(models_dir, "upscalers", "ltx-2.3-temporal-upscaler-x2-1.0.safetensors"):
            "Temporal upscaler x2 (pipeline.py)",
    }

    # --- Dossiers REQUIS ---
    required_dirs: dict[str, str] = {
        os.path.join(models_dir, "text_encoders", "gemma-3-12b-it"):
            "Text encoder Gemma 3 12B HF (warmup_embeddings.py)",
        # Cache persistant — sans ce dossier, le warmup doit recharger Gemma 12B (~24GB RAM)
        os.path.join(volume_root, "cache", "embeddings"):
            "Cache embeddings persistant — NE PAS SUPPRIMER (warmup_embeddings.py, pipeline.py)",
    }

    # --- Dossiers DYNAMIQUES (generes au runtime, supprimables) ---
    dynamic_dirs: dict[str, str] = {
        os.path.join(volume_root, "cache", "dedup"):
            "Cache dedup par hash input",
        os.path.join(volume_root, "cache", "transformer"):
            "Cache transformer pre-fusionne (pipeline.py, regenerable)",
        os.path.join(volume_root, "output"):
            "Outputs video des jobs",
    }

    used: list[dict] = []
    unused: list[dict] = []
    dynamic: list[dict] = []

    # Normaliser les paths
    required_files_norm = {os.path.normpath(k): v for k, v in required_files.items()}
    required_dirs_norm = {os.path.normpath(k): v for k, v in required_dirs.items()}
    dynamic_dirs_norm = {os.path.normpath(k): v for k, v in dynamic_dirs.items()}

    def is_under_required_dir(path: str) -> str | None:
        """Verifie si un path est sous un dossier requis."""
        norm = os.path.normpath(path)
        for d in required_dirs_norm:
            if norm == d or norm.startswith(d + os.sep):
                return required_dirs_norm[d]
        return None

    def is_under_dynamic_dir(path: str) -> str | None:
        """Verifie si un path est sous un dossier dynamique."""
        norm = os.path.normpath(path)
        for d in dynamic_dirs_norm:
            if norm == d or norm.startswith(d + os.sep):
                return dynamic_dirs_norm[d]
        return None

    # Scanner les dossiers models/
    for dirpath, dirnames, filenames in os.walk(os.path.join(volume_root, "models")):
        # Gerer les dossiers requis (gemma) — ne pas descendre dedans
        dirs_to_skip = []
        for d in dirnames:
            full = os.path.normpath(os.path.join(dirpath, d))
            reason = is_under_required_dir(full)
            if reason is not None:
                size = dir_size(full)
                used.append({"path": full, "size": size, "reason": reason})
                dirs_to_skip.append(d)
        for d in dirs_to_skip:
            dirnames.remove(d)

        for f in filenames:
            fp = os.path.normpath(os.path.join(dirpath, f))
            size = os.path.getsize(fp) if os.path.isfile(fp) else 0
            if fp in required_files_norm:
                used.append({"path": fp, "size": size, "reason": required_files_norm[fp]})
            else:
                unused.append({"path": fp, "size": size, "reason": "NON UTILISE par le pipeline"})

    # Scanner les dossiers dynamiques (cache, output)
    for ddir, reason in dynamic_dirs_norm.items():
        if os.path.isdir(ddir):
            size = dir_size(ddir)
            count = sum(1 for _, _, files in os.walk(ddir) for _ in files)
            dynamic.append({"path": ddir, "size": size, "reason": f"{reason} ({count} fichiers)"})

    # Scanner les dossiers requis hors models/ (ex: cache/embeddings)
    found_dirs = {item["path"] for item in used}
    for dpath, reason in required_dirs_norm.items():
        if dpath not in found_dirs:
            if os.path.isdir(dpath):
                size = dir_size(dpath)
                used.append({"path": dpath, "size": size, "reason": reason})

    # Scanner tout le reste a la racine du volume (hors models/, cache/, output/)
    known_top_dirs = {"models", "cache", "output"}
    if os.path.isdir(volume_root):
        for entry in sorted(os.listdir(volume_root)):
            if entry in known_top_dirs:
                continue
            full = os.path.join(volume_root, entry)
            if os.path.isdir(full):
                size = dir_size(full)
                unused.append({"path": full + "/", "size": size, "reason": "Dossier inconnu a la racine du volume"})
            elif os.path.isfile(full):
                size = os.path.getsize(full)
                unused.append({"path": full, "size": size, "reason": "Fichier inconnu a la racine du volume"})

    # Scanner sous-dossiers inconnus dans cache/
    cache_dir = os.path.normpath(os.path.join(volume_root, "cache"))
    if os.path.isdir(cache_dir):
        known_cache_subdirs: set[str] = set()
        for d in list(dynamic_dirs_norm.keys()) + list(required_dirs_norm.keys()):
            if d.startswith(cache_dir + os.sep):
                known_cache_subdirs.add(os.path.relpath(d, cache_dir).split(os.sep)[0])
        for entry in sorted(os.listdir(cache_dir)):
            if entry in known_cache_subdirs:
                continue
            full = os.path.join(cache_dir, entry)
            if os.path.isdir(full):
                size = dir_size(full)
                unused.append({"path": full + "/", "size": size, "reason": "Dossier inconnu dans cache/"})
            elif os.path.isfile(full):
                size = os.path.getsize(full)
                unused.append({"path": full, "size": size, "reason": "Fichier inconnu dans cache/"})

    # Detection des fichiers/dossiers MANQUANTS
    missing: list[dict] = []
    found_paths = {item["path"] for item in used}
    for fpath, reason in required_files_norm.items():
        if fpath not in found_paths:
            missing.append({"path": fpath, "reason": reason})
    for dpath, reason in required_dirs_norm.items():
        if dpath not in found_paths:
            missing.append({"path": dpath, "reason": reason})

    return used, unused, dynamic, missing


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit des fichiers sur le network volume")
    parser.add_argument("--volume", default="/runpod-volume", help="Chemin du volume (default: /runpod-volume)")
    parser.add_argument("--json", action="store_true", help="Sortie JSON pour exploitation programmatique")
    args = parser.parse_args()

    volume = args.volume
    if not os.path.isdir(volume):
        print(f"ERREUR: Volume non trouve: {volume}")
        return

    # 1. Audit des fichiers
    used, unused, dynamic, missing = audit_volume(volume)
    total_used = sum(item["size"] for item in used)
    total_unused = sum(item["size"] for item in unused)

    # 2. Espace disque
    disk = shutil.disk_usage(volume)

    # 3. Mode JSON
    if args.json:
        print(json.dumps({
            "volume": volume,
            "disk_total_bytes": disk.total,
            "disk_used_bytes": disk.used,
            "disk_free_bytes": disk.free,
            "pipeline_used_bytes": total_used,
            "recoverable_bytes": total_unused,
            "used": sorted(used, key=lambda x: x["path"]),
            "unused": sorted(unused, key=lambda x: -x["size"]),
            "dynamic": sorted(dynamic, key=lambda x: x["path"]),
            "missing": sorted(missing, key=lambda x: x["path"]),
        }))
        return

    # 4. Header texte avec infos disque
    print(f"\n{'='*70}")
    print(f"  AUDIT VOLUME: {volume}")
    print(f"{'='*70}")
    print(f"  Volume          : {volume}")
    print(f"  Disque total    : {human_size(disk.total)}")
    print(f"  Disque utilise  : {human_size(disk.used)}")
    print(f"  Disque libre    : {human_size(disk.free)}")
    print(f"  Utilise pipeline: {human_size(total_used)}")
    print(f"  Recuperable     : {human_size(total_unused)}")
    print(f"  Manquants       : {len(missing)}")
    print()

    # --- Fichiers MANQUANTS ---
    if missing:
        print(f"--- MANQUANTS — CRITIQUES ({len(missing)} elements) ---\n")
        for item in sorted(missing, key=lambda x: x["path"]):
            print(f"  [!!] MANQUANT  {item['path']}")
            print(f"       → {item['reason']}")
        print()

    # --- Fichiers UTILISES ---
    print(f"--- UTILISES PAR LE PIPELINE ({len(used)} elements) ---\n")
    for item in sorted(used, key=lambda x: x["path"]):
        print(f"  [OK] {human_size(item['size']):>10}  {item['path']}")
        print(f"       → {item['reason']}")
    print(f"\n  Total utilise: {human_size(total_used)}\n")

    # --- Fichiers DYNAMIQUES (ne pas toucher) ---
    if dynamic:
        total_dynamic = sum(item["size"] for item in dynamic)
        print(f"--- CACHE / OUTPUT ({len(dynamic)} dossiers, {human_size(total_dynamic)}) ---\n")
        for item in sorted(dynamic, key=lambda x: x["path"]):
            print(f"  [OK] {human_size(item['size']):>10}  {item['path']}")
            print(f"       → {item['reason']}")
        print()

    # --- Fichiers NON UTILISES ---
    if unused:
        print(f"--- NON UTILISES — SUPPRIMABLES ({len(unused)} elements) ---\n")
        for item in sorted(unused, key=lambda x: -x["size"]):
            print(f"  [!!] {human_size(item['size']):>10}  {item['path']}")
            print(f"       → {item['reason']}")
        print(f"\n  Total non utilise: {human_size(total_unused)}")
        print(f"\n  Pour supprimer:")
        for item in sorted(unused, key=lambda x: -x["size"]):
            if item["path"].endswith("/"):
                print(f"    trash-put -r {item['path']}")
            else:
                print(f"    trash-put {item['path']}")
    else:
        print("--- AUCUN FICHIER INUTILISE ---\n")
        print("  Le volume est propre.")

    print(f"\n{'='*70}")
    total_all = total_used + sum(d["size"] for d in dynamic) + total_unused
    print(f"  Resume: {human_size(total_used)} utilise"
          f" + {human_size(sum(d['size'] for d in dynamic))} cache/output"
          f" + {human_size(total_unused)} inutilise"
          f" = {human_size(total_all)} total")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
