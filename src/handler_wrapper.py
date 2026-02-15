"""
Wrapper autour du handler worker-comfyui pour RunPod Serverless.
Ajoute le cleanup des fichiers generes apres chaque job.
"""

import glob
import os
import shutil
import sys
import time

import requests
import runpod

# Import du handler original
# network_volume est dans /worker-comfyui/src/, handler.py a la racine
sys.path.insert(0, "/worker-comfyui/src")
sys.path.insert(0, "/worker-comfyui")
os.chdir("/worker-comfyui")
from handler import handler as original_handler

COMFYUI_URL = "http://127.0.0.1:8188"
CLEANUP_DIRS = ["/ComfyUI/output", "/ComfyUI/input", "/ComfyUI/temp"]
FILE_AGE_THRESHOLD = 5  # secondes


def get_disk_usage_mb(path="/"):
    """Retourne l'espace utilise en MB."""
    stat = shutil.disk_usage(path)
    return stat.used / (1024 * 1024)


def cleanup_old_files(directories, min_age_seconds):
    """Supprime les fichiers plus vieux que min_age_seconds dans les dossiers donnes."""
    now = time.time()
    removed = 0
    for directory in directories:
        if not os.path.isdir(directory):
            continue
        for filepath in glob.glob(os.path.join(directory, "*")):
            if os.path.isfile(filepath):
                age = now - os.path.getmtime(filepath)
                if age > min_age_seconds:
                    try:
                        os.remove(filepath)
                        removed += 1
                    except OSError as e:
                        print(f"[wrapper] Erreur suppression {filepath}: {e}")
    return removed


def purge_comfyui_history():
    """Purge l'historique ComfyUI pour liberer la memoire."""
    try:
        resp = requests.post(f"{COMFYUI_URL}/history", json={"clear": True}, timeout=5)
        if resp.ok:
            print("[wrapper] Historique ComfyUI purge")
        else:
            print(f"[wrapper] Purge historique: HTTP {resp.status_code}")
    except requests.RequestException as e:
        print(f"[wrapper] Erreur purge historique: {e}")


def wrapped_handler(job):
    """Handler avec cleanup post-job."""
    disk_before = get_disk_usage_mb()
    print(f"[wrapper] Job {job.get('id', '?')} - Disque avant: {disk_before:.0f} MB")

    try:
        result = original_handler(job)
        return result
    finally:
        # Cleanup post-job (toujours execute)
        removed = cleanup_old_files(CLEANUP_DIRS, FILE_AGE_THRESHOLD)
        purge_comfyui_history()

        disk_after = get_disk_usage_mb()
        freed = disk_before - disk_after
        print(
            f"[wrapper] Cleanup: {removed} fichiers supprimes, "
            f"disque apres: {disk_after:.0f} MB (libere: {freed:.0f} MB)"
        )


runpod.serverless.start({"handler": wrapped_handler})
