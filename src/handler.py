"""
Handler RunPod Serverless pour Medusa I2V (ltx-pipelines direct).

API simplifiee : image + camera → video MP4.
Reprend les fonctionnalites cles de handler_wrapper.py :
  - Dedup cache par hash
  - Sauvegarde outputs sur network volume
  - Cleanup disque ephemere
  - Resolution dynamique (aspect ratio preserve, ~0.92M px, align 32px)
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import math
import os
import random
import shutil
import tempfile
import time

import boto3
import requests
import runpod
from botocore.config import Config
from PIL import Image

from pipeline import MedusaPipeline

logging.basicConfig(level=logging.INFO, format="[handler] %(message)s")
log = logging.getLogger("handler")

# --- Constantes ---

VIDEO_EXTENSIONS = {".mp4", ".webm", ".gif", ".webp"}
OUTPUT_VOLUME_DIR = os.environ.get("OUTPUT_VOLUME_DIR", "/runpod-volume/output")
CACHE_DIR = os.environ.get("CACHE_DIR", "/runpod-volume/cache")
VOLUME_ROOT = os.environ.get("VOLUME_ROOT", "/runpod-volume")
MODELS_DIR = os.environ.get("MODELS_DIR", "/runpod-volume/models")

# Mapping camera → (fichier LoRA, prompt par defaut)
CAMERAS: dict[str, tuple[str, str]] = {
    "dolly-in": (
        "ltx-2-19b-lora-camera-control-dolly-in.safetensors",
        "A steady dolly-in camera movement, smooth forward motion, cinematic.",
    ),
    "dolly-out": (
        "ltx-2-19b-lora-camera-control-dolly-out.safetensors",
        "A steady dolly-out camera movement, smooth backward motion, cinematic.",
    ),
    "dolly-left": (
        "ltx-2-19b-lora-camera-control-dolly-left.safetensors",
        "A steady dolly-left camera movement, smooth lateral motion to the left, cinematic.",
    ),
    "dolly-right": (
        "ltx-2-19b-lora-camera-control-dolly-right.safetensors",
        "A steady dolly-right camera movement, smooth lateral motion to the right, cinematic.",
    ),
    "jib-down": (
        "ltx-2-19b-lora-camera-control-jib-down.safetensors",
        "A steady jib-down camera movement, smooth downward motion, cinematic.",
    ),
    "jib-up": (
        "ltx-2-19b-lora-camera-control-jib-up.safetensors",
        "A steady jib-up camera movement, smooth upward motion, cinematic.",
    ),
    "static": (
        "ltx-2-19b-lora-camera-control-static.safetensors",
        "A static camera, no movement, cinematic.",
    ),
}

# --- S3 ---

S3_BUCKET = os.environ.get("S3_BUCKET")
S3_ENDPOINT_URL = os.environ.get("S3_ENDPOINT_URL", "https://s3.sbg.io.cloud.ovh.net")
S3_REGION = os.environ.get("S3_REGION", "sbg")
log.info("S3 config: bucket=%s, endpoint=%s", S3_BUCKET or "(disabled)", S3_ENDPOINT_URL)

_s3_client = None

S3_CONTENT_TYPES = {
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".gif": "image/gif",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}


def get_s3_client():
    """Client S3 singleton (lazy init)."""
    global _s3_client
    if _s3_client is None:
        access_key = os.environ.get("AWS_ACCESS_KEY_ID") or os.environ.get("aws_access_key_id")
        secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY") or os.environ.get("aws_secret_access_key")
        _s3_client = boto3.client(
            "s3",
            endpoint_url=S3_ENDPOINT_URL,
            region_name=S3_REGION,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=Config(retries={"max_attempts": 3, "mode": "standard"}),
        )
    return _s3_client


def upload_to_s3(filepath: str, s3_key: str) -> str | None:
    """Upload fichier vers S3. Retourne l'URL publique ou None si desactive/erreur."""
    if not S3_BUCKET:
        return None
    try:
        ext = os.path.splitext(filepath)[1].lower()
        content_type = S3_CONTENT_TYPES.get(ext, "application/octet-stream")
        client = get_s3_client()
        client.upload_file(filepath, S3_BUCKET, s3_key, ExtraArgs={"ContentType": content_type})
        s3_url = f"{S3_ENDPOINT_URL}/{S3_BUCKET}/{s3_key}"
        log.info("S3 upload OK: %s (%.1f MB)", s3_key, os.path.getsize(filepath) / (1024 * 1024))
        return s3_url
    except Exception as e:
        log.warning("S3 upload echoue: %s", e)
        return None


DEFAULT_NEGATIVE_PROMPT = (
    "blurry, out of focus, low quality, distorted, watermark, "
    "logo, text, subtitle, banner, signature, username, "
    "compressed artifacts, jpeg artifacts, noise, grainy"
)


# --- Utilitaires ---


def get_disk_usage_mb(path: str = "/") -> float:
    """Retourne l'espace utilise en MB."""
    stat = shutil.disk_usage(path)
    return stat.used / (1024 * 1024)


def resolve_image(image_data: str) -> str:
    """URL ou base64 → chemin fichier temporaire."""
    if image_data.startswith(("http://", "https://")):
        log.info("Telechargement image: %s", image_data[:80])
        resp = requests.get(image_data, timeout=30)
        resp.raise_for_status()
        data = resp.content
        log.info("Image telechargee: %.0f KB", len(data) / 1024)
    else:
        data = base64.b64decode(image_data)
        log.info("Image base64 decodee: %.0f KB", len(data) / 1024)

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.write(data)
    tmp.close()
    return tmp.name


def compute_target_resolution(
    image_path: str,
    target_megapixels: float = 0.92,
) -> tuple[int, int]:
    """Calcule la resolution cible en preservant l'aspect ratio.

    Reproduit la logique ComfyUI (ResizeImageMaskNode scale 0.92M + align 32px).

    Returns:
        (height, width) alignes sur 32 pixels.
    """
    img = Image.open(image_path)
    w, h = img.size
    scale = math.sqrt(target_megapixels * 1_000_000 / (w * h))
    target_w = round(w * scale / 32) * 32
    target_h = round(h * scale / 32) * 32
    # Clamp minimum
    target_w = max(target_w, 32)
    target_h = max(target_h, 32)
    return target_h, target_w


def compute_input_hash(job_input: dict) -> str:
    """Hash deterministe de l'input pour dedup."""
    raw = json.dumps(job_input, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def lookup_cache(input_hash: str) -> list[dict] | None:
    """Cherche des outputs existants dans le cache volume pour ce hash."""
    cache_path = os.path.join(CACHE_DIR, "dedup", input_hash)
    if not os.path.isdir(cache_path):
        return None

    files = [
        f for f in os.listdir(cache_path)
        if os.path.isfile(os.path.join(cache_path, f))
    ]
    if not files:
        return None

    outputs = []
    for filename in files:
        filepath = os.path.join(cache_path, filename)
        ext = os.path.splitext(filename)[1].lower()
        file_type = "video" if ext in VIDEO_EXTENSIONS else "image"
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        s3_key = f"generated/videos/{filename}"
        s3_url = upload_to_s3(filepath, s3_key)
        entry = {
            "filename": filename,
            "content_type": file_type,
            "size_mb": round(size_mb, 2),
            "volume_path": filepath,
            "s3_key": s3_key,
        }
        if s3_url:
            entry["s3_url"] = s3_url
        outputs.append(entry)
    return outputs


def save_to_cache(input_hash: str, source_file: str) -> None:
    """Copie l'output dans le cache volume indexe par hash."""
    cache_path = os.path.join(CACHE_DIR, "dedup", input_hash)
    os.makedirs(cache_path, exist_ok=True)
    filename = os.path.basename(source_file)
    shutil.copy2(source_file, os.path.join(cache_path, filename))


def collect_output(source_file: str, job_id: str) -> dict:
    """Copie l'output sur le volume et retourne les metadonnees."""
    dest_dir = os.path.join(OUTPUT_VOLUME_DIR, job_id)
    os.makedirs(dest_dir, exist_ok=True)

    filename = os.path.basename(source_file)
    dest_path = os.path.join(dest_dir, filename)
    shutil.copy2(source_file, dest_path)

    ext = os.path.splitext(filename)[1].lower()
    file_type = "video" if ext in VIDEO_EXTENSIONS else "image"
    size_mb = os.path.getsize(dest_path) / (1024 * 1024)
    s3_key = f"generated/videos/{filename}"

    log.info("%s: %.1f MB (%s) -> %s/", filename, size_mb, file_type, dest_dir)

    s3_url = upload_to_s3(dest_path, s3_key)
    result = {
        "filename": filename,
        "content_type": file_type,
        "size_mb": round(size_mb, 2),
        "volume_path": dest_path,
        "s3_key": s3_key,
    }
    if s3_url:
        result["s3_url"] = s3_url
    return result


# --- Handler ---

pipeline: MedusaPipeline | None = None


def handler(job: dict) -> dict:
    """Handler RunPod pour generation I2V."""
    global pipeline

    job_id = job.get("id", f"unknown-{int(time.time())}")
    job_input = job.get("input", {})

    # --- Dedup cache EN PREMIER (filesystem only, pas besoin du pipeline) ---
    input_hash = compute_input_hash(job_input)
    cached = lookup_cache(input_hash)
    if cached:
        log.info("Cache hit (%s) - %d fichier(s), skip execution", input_hash, len(cached))
        return {"images": cached, "cached": True}

    # --- Parse input ---
    image_data = job_input.get("image")
    if not image_data:
        return {"error": "Le champ 'image' est requis (URL https ou base64)"}

    camera = job_input.get("camera", "dolly-in")
    if camera not in CAMERAS:
        return {"error": f"Camera inconnue: {camera}. Choix: {list(CAMERAS.keys())}"}

    seed = job_input.get("seed", random.randint(0, 2**32 - 1))
    num_frames = job_input.get("num_frames", 25)
    frame_rate = job_input.get("frame_rate", 24)
    image_strength = job_input.get("image_strength", 1.0)
    last_image_data = job_input.get("last_image")
    last_image_strength = job_input.get("last_image_strength", 1.0)
    prompt_override = job_input.get("prompt")
    negative_override = job_input.get("negative_prompt")

    # --- Lazy init pipeline (seulement si cache miss) ---
    if pipeline is None:
        log.info("Pipeline non init — cold start...")
        pipeline = init_pipeline()

    disk_before = get_disk_usage_mb()
    log.info("Job %s - Disque avant: %.0f MB", job_id, disk_before)

    # --- Resolve image(s) ---
    tmp_image = None
    tmp_last_image = None
    try:
        tmp_image = resolve_image(image_data)
        if last_image_data:
            tmp_last_image = resolve_image(last_image_data)

        # --- Resolution dynamique ---
        height, width = compute_target_resolution(tmp_image)
        log.info("Resolution cible: %dx%d (aspect ratio preserve, ~0.92M px)", width, height)

        # --- Camera LoRA ---
        lora_filename, _default_prompt = CAMERAS[camera]
        camera_lora_path = os.path.join(MODELS_DIR, "loras", lora_filename)

        # --- Generate ---
        output_dir = tempfile.mkdtemp(prefix="medusa_")
        output_filename = f"medusa_i2v_{job_id}.mp4"
        output_path = os.path.join(output_dir, output_filename)

        start_time = time.time()
        pipeline.generate(
            image_path=tmp_image,
            camera_lora_path=camera_lora_path,
            camera_key=camera,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            output_path=output_path,
            image_strength=image_strength,
            last_image_path=tmp_last_image,
            last_image_strength=last_image_strength,
            prompt_override=prompt_override,
            negative_override=negative_override,
        )
        elapsed = time.time() - start_time
        log.info("Generation terminee: %.1fs", elapsed)

        # --- Collect output ---
        output_meta = collect_output(output_path, job_id)

        # --- Save to dedup cache ---
        save_to_cache(input_hash, output_path)
        log.info("Cache sauvegarde (%s)", input_hash)

        # --- Cleanup ---
        shutil.rmtree(output_dir, ignore_errors=True)

        disk_after = get_disk_usage_mb()
        freed = disk_before - disk_after
        log.info("Disque apres: %.0f MB (libere: %.0f MB)", disk_after, freed)

        return {"images": [output_meta]}

    except Exception as e:
        log.error("Erreur generation: %s", e, exc_info=True)
        return {"error": str(e)}

    finally:
        # Cleanup images temporaires
        if tmp_image and os.path.isfile(tmp_image):
            os.unlink(tmp_image)
        if tmp_last_image and os.path.isfile(tmp_last_image):
            os.unlink(tmp_last_image)


# --- Init & Start ---

def init_pipeline() -> MedusaPipeline:
    """Initialise le pipeline au demarrage du worker."""
    log.info("Initialisation MedusaPipeline...")
    p = MedusaPipeline(models_dir=MODELS_DIR)

    # 1. Warmup embeddings (depuis cache disque)
    embeddings_cache_dir = os.path.join(CACHE_DIR, "embeddings")
    os.makedirs(embeddings_cache_dir, exist_ok=True)
    p.warmup_embeddings(embeddings_cache_dir)

    # 2. Build transformer avec camera par defaut (dolly-in)
    dolly_in_path = os.path.join(MODELS_DIR, "loras", CAMERAS["dolly-in"][0])
    p.get_transformer(dolly_in_path)

    # 3. Charger video encoder + video decoder (persistent)
    p.load_video_encoder()
    p.load_video_decoder()

    log.info("Pipeline pret.")
    return p


if __name__ == "__main__":
    # Pas d'init ici — lazy init dans le handler au premier cache miss
    runpod.serverless.start({"handler": handler})
