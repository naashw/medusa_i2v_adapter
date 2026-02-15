"""
Wrapper autour du handler worker-comfyui pour RunPod Serverless.
- Supporte les images en input via URL (https) ou base64
- Retourne les outputs (video/image) en base64 dans la reponse du job
- Copie backup sur le network volume
- Cleanup du disque ephemere apres chaque job
"""

import base64
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
SKIP_FILES = {"_output_images_will_be_put_here"}
OUTPUT_VOLUME_DIR = os.environ.get("OUTPUT_VOLUME_DIR", "/runpod-volume/output")


def get_disk_usage_mb(path: str = "/") -> float:
    """Retourne l'espace utilise en MB."""
    stat = shutil.disk_usage(path)
    return stat.used / (1024 * 1024)


def collect_and_encode_outputs(source_dir: str, job_id: str) -> list[dict]:
    """Lit les outputs, les encode en base64, et copie un backup sur le volume."""
    outputs: list[dict] = []

    if not os.path.isdir(source_dir):
        return outputs

    files = [
        f
        for f in glob.glob(os.path.join(source_dir, "*"))
        if os.path.isfile(f) and os.path.basename(f) not in SKIP_FILES
    ]
    if not files:
        return outputs

    # Backup sur le network volume
    dest_dir = os.path.join(OUTPUT_VOLUME_DIR, job_id)
    os.makedirs(dest_dir, exist_ok=True)

    for filepath in files:
        filename = os.path.basename(filepath)
        ext = os.path.splitext(filename)[1].lower()
        file_type = "video" if ext in VIDEO_EXTENSIONS else "image"

        try:
            with open(filepath, "rb") as f:
                data = f.read()

            size_mb = len(data) / (1024 * 1024)
            b64_data = base64.b64encode(data).decode("utf-8")

            # Backup sur le volume
            shutil.copy2(filepath, os.path.join(dest_dir, filename))

            outputs.append({
                "filename": filename,
                "type": file_type,
                "size_mb": round(size_mb, 2),
                "data": b64_data,
            })
            print(
                f"[wrapper] {filename}: {size_mb:.1f} MB ({file_type}), "
                f"encode base64, backup -> {dest_dir}/"
            )
        except OSError as e:
            print(f"[wrapper] Erreur lecture {filepath}: {e}")

    return outputs


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


def resolve_image_urls(job: dict) -> dict:
    """Convertit les URLs d'images en base64 avant de passer au handler original."""
    images = job.get("input", {}).get("images")
    if not images:
        return job

    for img in images:
        image_data = img.get("image", "")
        if not image_data.startswith(("http://", "https://")):
            continue

        url = image_data
        name = img.get("name", "input.png")
        print(f"[wrapper] Telechargement image: {url}")
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            img["image"] = base64.b64encode(resp.content).decode("utf-8")
            size_kb = len(resp.content) / 1024
            print(f"[wrapper] Image {name}: {size_kb:.0f} KB telecharge, converti en base64")
        except requests.RequestException as e:
            print(f"[wrapper] Erreur telechargement {url}: {e}")
            raise ValueError(f"Impossible de telecharger l'image: {url} - {e}")

    return job


def wrapped_handler(job: dict) -> dict:
    """Handler avec outputs base64 dans la reponse et cleanup post-job."""
    job_id = job.get("id", f"unknown-{int(time.time())}")
    disk_before = get_disk_usage_mb()
    print(f"[wrapper] Job {job_id} - Disque avant: {disk_before:.0f} MB")

    # Convertir les URLs en base64 si necessaire
    job = resolve_image_urls(job)

    try:
        result = original_handler(job)
    except Exception:
        cleanup_ephemeral(CLEANUP_DIRS)
        purge_comfyui_history()
        raise

    # Encoder les outputs en base64 + backup volume AVANT cleanup
    outputs = collect_and_encode_outputs(COMFYUI_OUTPUT_DIR, job_id)

    # Cleanup du disque ephemere
    removed = cleanup_ephemeral(CLEANUP_DIRS)
    purge_comfyui_history()

    disk_after = get_disk_usage_mb()
    freed = disk_before - disk_after
    print(
        f"[wrapper] Cleanup: {removed} fichiers supprimes, "
        f"disque apres: {disk_after:.0f} MB (libere: {freed:.0f} MB)"
    )

    # Construire la reponse avec les outputs en base64
    if not isinstance(result, dict):
        result = {"original_result": result}

    if outputs:
        videos = [o for o in outputs if o["type"] == "video"]
        images = [o for o in outputs if o["type"] == "image"]
        print(f"[wrapper] Reponse: {len(videos)} video(s), {len(images)} image(s) en base64")

        # Remplacer le tableau images vide du handler original
        result["images"] = outputs
        result.pop("status", None)  # Retirer "success_no_images"
    else:
        print("[wrapper] Aucun output genere par le workflow")

    return result


runpod.serverless.start({"handler": wrapped_handler})
