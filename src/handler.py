"""
Handler RunPod Serverless pour Medusa I2V (ltx-pipelines direct).

API simplifiee : image + camera_motion → video MP4.
Reprend les fonctionnalites cles de handler_wrapper.py :
  - Dedup cache par hash
  - Sauvegarde outputs sur network volume
  - Cleanup disque ephemere
  - Resolution dynamique (720p 1-stage ou 1080p 2-stage, aspect ratio preserve)
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
from concurrent.futures import ThreadPoolExecutor

import boto3
import requests
import runpod
from botocore.config import Config
from PIL import Image

from ltx_pipelines.utils.media_io import encode_video

from pipeline import MedusaPipeline
from prompts import CAMERA_PRESETS

logging.basicConfig(level=logging.INFO, format="[handler] %(message)s")
log = logging.getLogger("handler")

# --- Constantes ---

VIDEO_EXTENSIONS = {".mp4", ".webm", ".gif", ".webp"}
OUTPUT_VOLUME_DIR = os.environ.get("OUTPUT_VOLUME_DIR", "/runpod-volume/output")
CACHE_DIR = os.environ.get("CACHE_DIR", "/runpod-volume/cache")
VOLUME_ROOT = os.environ.get("VOLUME_ROOT", "/runpod-volume")
MODELS_DIR = os.environ.get("MODELS_DIR", "/runpod-volume/models")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "2"))


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


RESOLUTION_CONFIGS: dict[str, tuple[float, int]] = {
    "720p": (0.92, 32),
    "1080p": (2.0, 64),
}


def compute_target_resolution(
    image_path: str,
    resolution: str = "720p",
) -> tuple[int, int]:
    """Calcule la resolution cible en preservant l'aspect ratio.

    720p : ~0.92M px, align 32px (1-stage).
    1080p : ~2M px, align 64px (2-stage, half-res doit etre multiple de 32).

    Returns:
        (height, width) alignes sur le step d'alignement.
    """
    target_megapixels, align = RESOLUTION_CONFIGS[resolution]
    img = Image.open(image_path)
    w, h = img.size
    scale = math.sqrt(target_megapixels * 1_000_000 / (w * h))
    target_w = round(w * scale / align) * align
    target_h = round(h * scale / align) * align
    # Clamp minimum
    target_w = max(target_w, align)
    target_h = max(target_h, align)
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


# --- Post-processing pool (MP4 encode + S3 upload en parallele du GPU) ---

postprocess_pool = ThreadPoolExecutor(max_workers=3)


def postprocess_and_upload(
    frames: list,
    frame_rate: int,
    output_dir: str,
    output_filename: str,
    job_id: str,
    input_hash: str,
) -> dict:
    """Encode MP4 + copie volume + S3 upload + cache dedup (thread pool worker)."""
    output_path = os.path.join(output_dir, output_filename)
    os.makedirs(output_dir, exist_ok=True)

    encode_video(
        video=iter(frames),
        fps=frame_rate,
        audio=None,
        audio_sample_rate=None,
        output_path=output_path,
        video_chunks_number=1,
    )
    log.info("MP4 encode: %s", output_filename)

    output_meta = collect_output(output_path, job_id)
    save_to_cache(input_hash, output_path)
    log.info("Cache sauvegarde (%s)", input_hash)

    # Cleanup temp dir
    shutil.rmtree(output_dir, ignore_errors=True)
    return output_meta


# --- Prefetch pool (download images pendant que le GPU travaille) ---

prefetch_pool = ThreadPoolExecutor(max_workers=max(BATCH_SIZE, 2))


def resolve_and_preprocess(
    image_data: str,
    resolution: str = "720p",
) -> tuple[str, int, int]:
    """Download + resolve image + compute target resolution (thread prefetch)."""
    tmp_path = resolve_image(image_data)
    height, width = compute_target_resolution(tmp_path, resolution=resolution)
    return tmp_path, height, width


# --- Handler ---

pipeline: MedusaPipeline | None = None


def handler(job: dict) -> dict:
    """Handler RunPod pour generation I2V.

    Supporte:
      - image (singulier) : generation single item (retro-compatible)
      - images (liste) : generation batch avec overlapping
    """
    global pipeline

    job_id = job.get("id", f"unknown-{int(time.time())}")
    job_input = job.get("input", {})

    # --- Dedup cache EN PREMIER (filesystem only, pas besoin du pipeline) ---
    input_hash = compute_input_hash(job_input)
    cached = lookup_cache(input_hash)
    if cached:
        log.info("Cache hit (%s) - %d fichier(s), skip execution", input_hash, len(cached))
        return {"images": cached, "cached": True}

    # --- Parse input : support image (singulier) et images (pluriel) ---
    image_data_list = job_input.get("images")
    single_image = job_input.get("image")
    if image_data_list:
        is_batch = len(image_data_list) > 1
    elif single_image:
        image_data_list = [single_image]
        is_batch = False
    else:
        return {"error": "Le champ 'image' ou 'images' est requis (URL https ou base64)"}

    # --- camera_motion : preset connu → description, sinon texte libre ---
    camera_motion = job_input.get("camera_motion", job_input.get("camera", "static"))
    camera_motion_text = CAMERA_PRESETS.get(camera_motion, camera_motion)

    base_seed = job_input.get("seed", random.randint(0, 2**32 - 1))
    num_frames = job_input.get("num_frames", 25)
    frame_rate = job_input.get("frame_rate", 24)
    image_strength = job_input.get("image_strength", 1.0)
    last_image_data = job_input.get("last_image")
    last_image_strength = job_input.get("last_image_strength", 1.0)
    negative_override = job_input.get("negative_prompt")

    resolution = job_input.get("resolution", "720p")
    if resolution not in RESOLUTION_CONFIGS:
        return {"error": f"Resolution inconnue: {resolution}. Choix: {list(RESOLUTION_CONFIGS.keys())}"}

    # --- Pipeline deja init au startup (eager init) ---
    if pipeline is None:
        raise RuntimeError("Pipeline non initialise — le worker doit etre lance via __main__")

    disk_before = get_disk_usage_mb()
    total_images = len(image_data_list)
    log.info("Job %s - %d image(s), camera_motion=%s, disque avant: %.0f MB", job_id, total_images, camera_motion, disk_before)

    two_stage = resolution == "1080p"

    # Seeds uniques par item
    seeds = [base_seed + i for i in range(total_images)]

    tmp_images: list[str] = []
    tmp_last_image: str | None = None
    all_futures: list = []

    try:
        # Resolve last_image une seule fois (partage par tous les items)
        if last_image_data:
            tmp_last_image = resolve_image(last_image_data)

        if not is_batch:
            # ===== SINGLE IMAGE (retro-compatible) =====
            tmp_img = resolve_image(image_data_list[0])
            tmp_images.append(tmp_img)

            height, width = compute_target_resolution(tmp_img, resolution=resolution)
            log.info(
                "Resolution cible: %dx%d (%s, %s)",
                width, height, resolution, "2-stage" if two_stage else "1-stage",
            )

            start_time = time.time()
            frames = pipeline.generate_frames(
                image_path=tmp_img,
                prompt=camera_motion_text,
                seed=base_seed,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=frame_rate,
                image_strength=image_strength,
                last_image_path=tmp_last_image,
                last_image_strength=last_image_strength,
                negative_override=negative_override,
                two_stage=two_stage,
            )
            elapsed = time.time() - start_time
            log.info("Generation frames terminee: %.1fs", elapsed)

            output_dir = tempfile.mkdtemp(prefix="medusa_")
            output_filename = f"medusa_i2v_{job_id}.mp4"
            future = postprocess_pool.submit(
                postprocess_and_upload,
                frames, int(frame_rate),
                output_dir, output_filename,
                job_id, input_hash,
            )
            output_meta = future.result()

            disk_after = get_disk_usage_mb()
            log.info("Disque apres: %.0f MB (libere: %.0f MB)", disk_after, disk_before - disk_after)
            return {"images": [output_meta]}

        else:
            # ===== BATCH MODE (overlapping + prefetch) =====
            log.info("Batch mode: %d images, BATCH_SIZE=%d", total_images, BATCH_SIZE)

            # Split en sub-batches
            sub_batches = [
                image_data_list[i:i + BATCH_SIZE]
                for i in range(0, total_images, BATCH_SIZE)
            ]
            sub_seeds = [
                seeds[i:i + BATCH_SIZE]
                for i in range(0, total_images, BATCH_SIZE)
            ]

            next_batch_futures = None
            height: int | None = None
            width: int | None = None

            for batch_idx, (batch_urls, batch_seeds) in enumerate(zip(sub_batches, sub_seeds)):
                batch_size = len(batch_urls)

                # --- Recuperer images (prefetchees ou resolve maintenant) ---
                if next_batch_futures is not None:
                    current_images = [f.result() for f in next_batch_futures]
                else:
                    # Premier batch — resolve en parallele
                    with ThreadPoolExecutor(max_workers=batch_size) as pool:
                        current_images = list(pool.map(resolve_image, batch_urls))
                tmp_images.extend(current_images)

                # Resolution depuis la premiere image du premier batch
                if height is None:
                    height, width = compute_target_resolution(current_images[0], resolution=resolution)
                    log.info(
                        "Resolution cible: %dx%d (%s, %s)",
                        width, height, resolution, "2-stage" if two_stage else "1-stage",
                    )

                # --- Prefetch du batch suivant AVANT le denoising GPU ---
                if batch_idx + 1 < len(sub_batches):
                    next_batch_futures = [
                        prefetch_pool.submit(resolve_image, url)
                        for url in sub_batches[batch_idx + 1]
                    ]
                else:
                    next_batch_futures = None

                # --- Generation batch (GPU) ---
                start_time = time.time()

                if batch_size > 1:
                    items = []
                    for img_path, seed_i in zip(current_images, batch_seeds):
                        item: dict = {"image_path": img_path, "seed": seed_i}
                        if tmp_last_image:
                            item["last_image_path"] = tmp_last_image
                            item["last_image_strength"] = last_image_strength
                        items.append(item)

                    batch_frames = pipeline.generate_batch_frames(
                        items=items,
                        prompt=camera_motion_text,
                        height=height,
                        width=width,
                        num_frames=num_frames,
                        frame_rate=frame_rate,
                        image_strength=image_strength,
                        two_stage=two_stage,
                    )
                else:
                    # Sub-batch de 1 — utiliser generate_frames (single item)
                    frames = pipeline.generate_frames(
                        image_path=current_images[0],
                        prompt=camera_motion_text,
                        seed=batch_seeds[0],
                        height=height,
                        width=width,
                        num_frames=num_frames,
                        frame_rate=frame_rate,
                        image_strength=image_strength,
                        last_image_path=tmp_last_image,
                        last_image_strength=last_image_strength,
                        two_stage=two_stage,
                    )
                    batch_frames = [frames]

                elapsed = time.time() - start_time
                log.info(
                    "Sub-batch %d/%d: %.1fs (%d items)",
                    batch_idx + 1, len(sub_batches), elapsed, batch_size,
                )

                # --- Post-processing async (Opt 4) ---
                for item_idx, item_frames in enumerate(batch_frames):
                    global_idx = batch_idx * BATCH_SIZE + item_idx
                    output_dir = tempfile.mkdtemp(prefix="medusa_")
                    output_filename = f"medusa_i2v_{job_id}_{global_idx}.mp4"

                    fut = postprocess_pool.submit(
                        postprocess_and_upload,
                        item_frames, int(frame_rate),
                        output_dir, output_filename,
                        job_id, input_hash,
                    )
                    all_futures.append(fut)

            # --- Attendre tous les post-processing avant return ---
            results = [f.result() for f in all_futures]

            disk_after = get_disk_usage_mb()
            log.info("Disque apres: %.0f MB (libere: %.0f MB)", disk_after, disk_before - disk_after)
            return {"images": results}

    except Exception as e:
        log.error("Erreur generation: %s", e, exc_info=True)
        return {"error": str(e)}

    finally:
        # Cleanup images temporaires
        for tmp_img in tmp_images:
            if os.path.isfile(tmp_img):
                os.unlink(tmp_img)
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

    # 2. Build transformer (distilled LoRA fusionnee)
    p.get_transformer()

    # 3. Charger video encoder + video decoder + spatial upsampler (persistent)
    p.load_video_encoder()
    p.load_video_decoder()
    p.load_spatial_upsampler()

    log.info("Pipeline pret.")
    return p


if __name__ == "__main__":
    # Eager init — le pipeline est pret avant le premier job
    pipeline = init_pipeline()
    runpod.serverless.start({"handler": handler})
