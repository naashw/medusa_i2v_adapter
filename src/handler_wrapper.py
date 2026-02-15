"""
Wrapper autour du handler worker-comfyui pour RunPod Serverless.
- Copie les videos/images generees vers le network volume
- Cleanup du disque ephemere apres chaque job
"""

import glob
import os
import shutil
import sys
import time

import requests
import runpod

# Import du handler original
sys.path.insert(0, "/worker-comfyui")
from handler import handler as original_handler

COMFYUI_URL = "http://127.0.0.1:8188"
COMFYUI_OUTPUT_DIR = "/ComfyUI/output"
CLEANUP_DIRS = ["/ComfyUI/output", "/ComfyUI/input", "/ComfyUI/temp"]
VIDEO_EXTENSIONS = {".mp4", ".webm", ".gif", ".webp"}
OUTPUT_VOLUME_DIR = os.environ.get("OUTPUT_VOLUME_DIR", "/runpod-volume/output")


def get_disk_usage_mb(path: str = "/") -> float:
    """Retourne l'espace utilise en MB."""
    stat = shutil.disk_usage(path)
    return stat.used / (1024 * 1024)


def collect_outputs(source_dir: str, job_id: str) -> list[dict]:
    """Copie les fichiers generes vers le network volume."""
    dest_dir = os.path.join(OUTPUT_VOLUME_DIR, job_id)
    collected: list[dict] = []

    if not os.path.isdir(source_dir):
        return collected

    files = [
        f
        for f in glob.glob(os.path.join(source_dir, "*"))
        if os.path.isfile(f)
    ]
    if not files:
        return collected

    os.makedirs(dest_dir, exist_ok=True)

    for filepath in files:
        filename = os.path.basename(filepath)
        ext = os.path.splitext(filename)[1].lower()
        dest_path = os.path.join(dest_dir, filename)
        try:
            shutil.copy2(filepath, dest_path)
            size_mb = os.path.getsize(dest_path) / (1024 * 1024)
            file_type = "video" if ext in VIDEO_EXTENSIONS else "image"
            collected.append({
                "filename": filename,
                "path": dest_path,
                "size_mb": round(size_mb, 2),
                "type": file_type,
            })
            print(f"[wrapper] Copie: {filename} -> {dest_dir}/ ({size_mb:.1f} MB)")
        except OSError as e:
            print(f"[wrapper] Erreur copie {filepath}: {e}")

    return collected


def cleanup_ephemeral(directories: list[str]) -> int:
    """Supprime tous les fichiers dans les dossiers ephemeres du container."""
    removed = 0
    for directory in directories:
        if not os.path.isdir(directory):
            continue
        for filepath in glob.glob(os.path.join(directory, "*")):
            if os.path.isfile(filepath):
                try:
                    os.remove(filepath)
                    removed += 1
                except OSError as e:
                    print(f"[wrapper] Erreur suppression {filepath}: {e}")
    return removed


def purge_comfyui_history() -> None:
    """Purge l'historique ComfyUI pour liberer la memoire."""
    try:
        resp = requests.post(
            f"{COMFYUI_URL}/history", json={"clear": True}, timeout=5
        )
        if resp.ok:
            print("[wrapper] Historique ComfyUI purge")
        else:
            print(f"[wrapper] Purge historique: HTTP {resp.status_code}")
    except requests.RequestException as e:
        print(f"[wrapper] Erreur purge historique: {e}")


def wrapped_handler(job: dict) -> dict:
    """Handler avec collecte des outputs sur le volume et cleanup post-job."""
    job_id = job.get("id", f"unknown-{int(time.time())}")
    disk_before = get_disk_usage_mb()
    print(f"[wrapper] Job {job_id} - Disque avant: {disk_before:.0f} MB")

    try:
        result = original_handler(job)
    except Exception:
        cleanup_ephemeral(CLEANUP_DIRS)
        purge_comfyui_history()
        raise

    # Copier les outputs vers le network volume AVANT cleanup
    outputs = collect_outputs(COMFYUI_OUTPUT_DIR, job_id)

    # Cleanup du disque ephemere du container
    removed = cleanup_ephemeral(CLEANUP_DIRS)
    purge_comfyui_history()

    disk_after = get_disk_usage_mb()
    freed = disk_before - disk_after
    print(
        f"[wrapper] Cleanup: {removed} fichiers supprimes, "
        f"disque apres: {disk_after:.0f} MB (libere: {freed:.0f} MB)"
    )

    # Enrichir le resultat avec les chemins sur le volume
    if outputs:
        videos = [o for o in outputs if o["type"] == "video"]
        images = [o for o in outputs if o["type"] == "image"]
        print(
            f"[wrapper] Outputs: {len(videos)} video(s), {len(images)} image(s) "
            f"-> {OUTPUT_VOLUME_DIR}/{job_id}/"
        )
        if isinstance(result, dict):
            result["output_volume"] = {
                "directory": f"{OUTPUT_VOLUME_DIR}/{job_id}",
                "files": outputs,
            }
        else:
            result = {
                "original_result": result,
                "output_volume": {
                    "directory": f"{OUTPUT_VOLUME_DIR}/{job_id}",
                    "files": outputs,
                },
            }
    else:
        print("[wrapper] Aucun output genere par le workflow")

    return result


runpod.serverless.start({"handler": wrapped_handler})
