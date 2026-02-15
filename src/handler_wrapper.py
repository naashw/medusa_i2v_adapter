"""
Wrapper autour du handler worker-comfyui pour RunPod Serverless.
- Supporte les images en input via URL (https) ou base64
- Retourne les outputs via URL presignee S3 (si configure) ou base64 (fallback)
- Backup sur le network volume
- Cleanup du disque ephemere apres chaque job
"""

import base64
import glob
import os
import shutil
import sys
import tempfile
import time

import requests
import runpod
from runpod.serverless.utils import rp_upload

# Import du handler original
sys.path.insert(0, "/worker-comfyui")
from handler import handler as original_handler

COMFYUI_URL = "http://127.0.0.1:8188"
COMFYUI_OUTPUT_DIR = "/ComfyUI/output"
CLEANUP_DIRS = ["/ComfyUI/output", "/ComfyUI/input", "/ComfyUI/temp"]
VIDEO_EXTENSIONS = {".mp4", ".webm", ".gif", ".webp"}
SKIP_FILES = {"_output_images_will_be_put_here"}
OUTPUT_VOLUME_DIR = os.environ.get("OUTPUT_VOLUME_DIR", "/runpod-volume/output")
S3_ENABLED = bool(os.environ.get("BUCKET_ENDPOINT_URL"))


def get_disk_usage_mb(path: str = "/") -> float:
    """Retourne l'espace utilise en MB."""
    stat = shutil.disk_usage(path)
    return stat.used / (1024 * 1024)


def upload_to_s3(job_id: str, filepath: str) -> str:
    """Upload un fichier vers S3 via le SDK RunPod, retourne l'URL presignee."""
    return rp_upload.upload_image(job_id, filepath)


def collect_outputs(source_dir: str, job_id: str) -> list[dict]:
    """Collecte les outputs : upload S3 (presigned URL) ou encode base64."""
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

    mode = "s3" if S3_ENABLED else "base64"
    print(f"[wrapper] Mode output: {mode}")

    for filepath in files:
        filename = os.path.basename(filepath)
        ext = os.path.splitext(filename)[1].lower()
        file_type = "video" if ext in VIDEO_EXTENSIONS else "image"
        size_mb = os.path.getsize(filepath) / (1024 * 1024)

        try:
            # Backup sur le volume
            shutil.copy2(filepath, os.path.join(dest_dir, filename))

            if S3_ENABLED:
                s3_url = upload_to_s3(job_id, filepath)
                outputs.append({
                    "filename": filename,
                    "type": "s3_url",
                    "content_type": file_type,
                    "size_mb": round(size_mb, 2),
                    "data": s3_url,
                })
                print(f"[wrapper] {filename}: {size_mb:.1f} MB -> S3: {s3_url[:80]}...")
            else:
                with open(filepath, "rb") as f:
                    raw = f.read()
                b64_data = base64.b64encode(raw).decode("utf-8")
                outputs.append({
                    "filename": filename,
                    "type": "base64",
                    "content_type": file_type,
                    "size_mb": round(size_mb, 2),
                    "data": b64_data,
                })
                print(f"[wrapper] {filename}: {size_mb:.1f} MB -> base64")
        except Exception as e:
            print(f"[wrapper] Erreur traitement {filename}: {e}")

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
    """Handler avec outputs S3/base64 dans la reponse et cleanup post-job."""
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

    # Collecter et uploader/encoder les outputs AVANT cleanup
    outputs = collect_outputs(COMFYUI_OUTPUT_DIR, job_id)

    # Cleanup du disque ephemere
    removed = cleanup_ephemeral(CLEANUP_DIRS)
    purge_comfyui_history()

    disk_after = get_disk_usage_mb()
    freed = disk_before - disk_after
    print(
        f"[wrapper] Cleanup: {removed} fichiers supprimes, "
        f"disque apres: {disk_after:.0f} MB (libere: {freed:.0f} MB)"
    )

    # Construire la reponse avec les outputs
    if not isinstance(result, dict):
        result = {"original_result": result}

    if outputs:
        videos = [o for o in outputs if o["content_type"] == "video"]
        images = [o for o in outputs if o["content_type"] == "image"]
        mode = "S3 presigned" if S3_ENABLED else "base64"
        print(f"[wrapper] Reponse: {len(videos)} video(s), {len(images)} image(s) via {mode}")

        result["images"] = outputs
        result.pop("status", None)
    else:
        print("[wrapper] Aucun output genere par le workflow")

    return result


runpod.serverless.start({"handler": wrapped_handler})
