"""
Wrapper autour du handler worker-comfyui pour RunPod Serverless.
- Supporte les images en input via URL (https) ou base64
- Sauvegarde les outputs sur le network volume
- Retourne les paths volume (accessible via RunPod S3 API + boto3)
- Cleanup du disque ephemere apres chaque job
"""

import base64
import glob
import hashlib
import json
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
CACHE_DIR = os.environ.get("CACHE_DIR", "/runpod-volume/cache")
VOLUME_ROOT = os.environ.get("VOLUME_ROOT", "/runpod-volume")

# Cold start : noeuds requis et retry
REQUIRED_NODES = {"LTXVImgToVideoInplace", "LTXVConditioning", "VHS_VideoCombine"}
COMFYUI_READY_TIMEOUT = 120
MAX_RETRIES = 10
RETRY_DELAY = 5  # secondes entre chaque retry
MIN_EXEC_TIME = 10  # secondes - en dessous, execution suspecte


def wait_for_comfyui_ready(timeout: int = COMFYUI_READY_TIMEOUT) -> None:
    """Attend que les custom nodes ComfyUI soient charges avant d'accepter des jobs."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(f"{COMFYUI_URL}/object_info", timeout=10)
            if resp.ok:
                nodes = set(resp.json().keys())
                missing = REQUIRED_NODES - nodes
                if not missing:
                    elapsed = time.time() - start
                    print(f"[wrapper] ComfyUI pret ({elapsed:.0f}s, {len(nodes)} noeuds charges)")
                    return
                print(f"[wrapper] Attente noeuds manquants: {missing}")
        except requests.RequestException:
            pass
        time.sleep(2)
    print(f"[wrapper] WARNING: timeout {timeout}s, noeuds pas confirmes - demarrage quand meme")


def get_disk_usage_mb(path: str = "/") -> float:
    """Retourne l'espace utilise en MB."""
    stat = shutil.disk_usage(path)
    return stat.used / (1024 * 1024)


def collect_outputs(source_dir: str, job_id: str) -> list[dict]:
    """Copie les outputs sur le volume et retourne les paths relatifs."""
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

    dest_dir = os.path.join(OUTPUT_VOLUME_DIR, job_id)
    os.makedirs(dest_dir, exist_ok=True)

    for filepath in files:
        filename = os.path.basename(filepath)
        ext = os.path.splitext(filename)[1].lower()
        file_type = "video" if ext in VIDEO_EXTENSIONS else "image"
        size_mb = os.path.getsize(filepath) / (1024 * 1024)

        try:
            dest_path = os.path.join(dest_dir, filename)
            shutil.copy2(filepath, dest_path)

            # Path relatif au volume (= cle S3 pour boto3)
            s3_key = os.path.relpath(dest_path, VOLUME_ROOT)

            outputs.append({
                "filename": filename,
                "content_type": file_type,
                "size_mb": round(size_mb, 2),
                "volume_path": dest_path,
                "s3_key": s3_key,
            })
            print(f"[wrapper] {filename}: {size_mb:.1f} MB ({file_type}) -> {dest_dir}/")
        except OSError as e:
            print(f"[wrapper] Erreur copie {filepath}: {e}")

    return outputs


def cleanup_ephemeral(directories: list[str]) -> int:
    """Supprime tous les fichiers et sous-dossiers dans les dossiers ephemeres du container."""
    removed = 0
    for directory in directories:
        if not os.path.isdir(directory):
            continue
        for entry in os.listdir(directory):
            path = os.path.join(directory, entry)
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
                removed += 1
            except OSError as e:
                print(f"[wrapper] Erreur suppression {path}: {e}")
    return removed


def clear_comfyui_cache() -> None:
    """Vide le cache d'execution et l'historique ComfyUI (sans decharger les modeles)."""
    try:
        requests.post(
            f"{COMFYUI_URL}/free",
            json={"unload_models": False, "free_memory": True},
            timeout=5,
        )
        requests.post(
            f"{COMFYUI_URL}/history", json={"clear": True}, timeout=5
        )
        print("[wrapper] Cache + historique ComfyUI vides")
    except requests.RequestException as e:
        print(f"[wrapper] Erreur vidage cache: {e}")


def compute_input_hash(job: dict) -> str:
    """Hash deterministe du workflow + images pour dedup."""
    payload = job.get("input", {})
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def lookup_cache(input_hash: str) -> list[dict] | None:
    """Cherche des outputs existants dans le cache volume pour ce hash."""
    cache_path = os.path.join(CACHE_DIR, input_hash)
    if not os.path.isdir(cache_path):
        return None

    files = [f for f in os.listdir(cache_path) if os.path.isfile(os.path.join(cache_path, f))]
    if not files:
        return None

    outputs = []
    for filename in files:
        filepath = os.path.join(cache_path, filename)
        ext = os.path.splitext(filename)[1].lower()
        file_type = "video" if ext in VIDEO_EXTENSIONS else "image"
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        s3_key = os.path.relpath(filepath, VOLUME_ROOT)
        outputs.append({
            "filename": filename,
            "content_type": file_type,
            "size_mb": round(size_mb, 2),
            "volume_path": filepath,
            "s3_key": s3_key,
        })
    return outputs


def save_to_cache(input_hash: str, source_dir: str) -> None:
    """Copie les outputs dans le cache volume indexe par hash."""
    cache_path = os.path.join(CACHE_DIR, input_hash)
    os.makedirs(cache_path, exist_ok=True)
    for f in os.listdir(source_dir):
        src = os.path.join(source_dir, f)
        if os.path.isfile(src) and f not in SKIP_FILES:
            shutil.copy2(src, os.path.join(cache_path, f))


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
    """Handler avec sauvegarde volume, retry cold start et cleanup post-job."""
    job_id = job.get("id", f"unknown-{int(time.time())}")
    disk_before = get_disk_usage_mb()
    print(f"[wrapper] Job {job_id} - Disque avant: {disk_before:.0f} MB")

    job = resolve_image_urls(job)

    # Dedup : si meme workflow + memes images, retourner le cache
    input_hash = compute_input_hash(job)
    cached = lookup_cache(input_hash)
    if cached:
        print(f"[wrapper] Cache hit ({input_hash}) - {len(cached)} fichier(s), skip execution")
        return {"images": cached, "cached": True}

    # Vider le cache ComfyUI avant execution
    clear_comfyui_cache()

    # Execution avec retry pour cold start (models pas encore en VRAM)
    result = None
    for attempt in range(MAX_RETRIES + 1):
        start_time = time.time()
        try:
            result = original_handler(job)
        except Exception:
            cleanup_ephemeral(CLEANUP_DIRS)
            clear_comfyui_cache()
            raise

        elapsed = time.time() - start_time
        output_files = [
            f for f in glob.glob(os.path.join(COMFYUI_OUTPUT_DIR, "*"))
            if os.path.isfile(f) and os.path.basename(f) not in SKIP_FILES
        ]

        if elapsed < MIN_EXEC_TIME and not output_files and attempt < MAX_RETRIES:
            print(
                f"[wrapper] Execution suspecte ({elapsed:.1f}s, 0 output) - "
                f"retry {attempt + 1}/{MAX_RETRIES} dans {RETRY_DELAY}s "
                f"(probable cold start, models pas encore en VRAM)..."
            )
            clear_comfyui_cache()
            time.sleep(RETRY_DELAY)
            continue

        if not output_files and elapsed < MIN_EXEC_TIME:
            print(f"[wrapper] WARNING: execution {elapsed:.1f}s sans output apres {MAX_RETRIES} retries")
        break

    # Sauvegarder les outputs sur le volume AVANT cleanup
    outputs = collect_outputs(COMFYUI_OUTPUT_DIR, job_id)

    # Sauvegarder dans le cache dedup
    if outputs:
        save_to_cache(input_hash, COMFYUI_OUTPUT_DIR)
        print(f"[wrapper] Cache sauvegarde ({input_hash})")

    # Cleanup du disque ephemere
    removed = cleanup_ephemeral(CLEANUP_DIRS)

    disk_after = get_disk_usage_mb()
    freed = disk_before - disk_after
    print(
        f"[wrapper] Cleanup: {removed} fichiers supprimes, "
        f"disque apres: {disk_after:.0f} MB (libere: {freed:.0f} MB)"
    )

    # Construire la reponse avec les paths volume
    if not isinstance(result, dict):
        result = {"original_result": result}

    if outputs:
        videos = [o for o in outputs if o["content_type"] == "video"]
        images = [o for o in outputs if o["content_type"] == "image"]
        print(f"[wrapper] Reponse: {len(videos)} video(s), {len(images)} image(s)")

        result["images"] = outputs
        result.pop("status", None)
    else:
        print("[wrapper] Aucun output genere par le workflow")

    return result


print("[wrapper] Verification noeuds custom ComfyUI...")
wait_for_comfyui_ready()
runpod.serverless.start({"handler": wrapped_handler})
