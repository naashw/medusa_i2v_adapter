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
MAX_BATCH = int(os.environ.get("MAX_BATCH", "9"))


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


FIXED_RESOLUTIONS: dict[str, tuple[int, int]] = {
    "720p": (704, 1280),
    "1080p": (1088, 1920),
}

DYNAMIC_CONFIGS: dict[str, tuple[float, int]] = {
    "720p": (0.92, 32),
    "1080p": (2.0, 64),
}


def compute_target_resolution(
    resolution: str = "720p",
    image_path: str | None = None,
    dynamic: bool = False,
) -> tuple[int, int]:
    """Retourne la resolution cible (height, width).

    Par defaut : resolution fixe (720p=1280x704, 1080p=1920x1088).
    Si dynamic=True et image_path fourni : calcul par aspect ratio.
    """
    if dynamic and image_path:
        target_megapixels, align = DYNAMIC_CONFIGS[resolution]
        img = Image.open(image_path)
        w, h = img.size
        scale = math.sqrt(target_megapixels * 1_000_000 / (w * h))
        target_w = round(w * scale / align) * align
        target_h = round(h * scale / align) * align
        target_w = max(target_w, align)
        target_h = max(target_h, align)
        return target_h, target_w
    return FIXED_RESOLUTIONS[resolution]



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
) -> dict:
    """Encode MP4 + copie volume + S3 upload (thread pool worker)."""
    output_path = os.path.join(output_dir, output_filename)
    os.makedirs(output_dir, exist_ok=True)

    encode_video(
        video=iter(frames),
        fps=frame_rate,
        audio=None,
        output_path=output_path,
        video_chunks_number=1,
    )
    log.info("MP4 encode: %s", output_filename)

    output_meta = collect_output(output_path, job_id)

    # Cleanup temp dir
    shutil.rmtree(output_dir, ignore_errors=True)
    return output_meta


# --- Normalisation input ---


def normalize_items(job_input: dict) -> tuple[list[dict], str | None]:
    """Normalize 3 formats d'input (items[], images[], image) en liste uniforme.

    Returns (items, error_message). error_message is None on success.
    Chaque item normalise: id, image, camera_motion, seed, last_image,
    last_image_strength, _original_index.
    """
    raw_items = job_input.get("items")
    raw_images = job_input.get("images")
    raw_image = job_input.get("image")

    shared_camera = job_input.get("camera_motion", job_input.get("camera", "static"))
    shared_seed = job_input.get("seed")
    shared_last_image = job_input.get("last_image")
    shared_last_image_strength = job_input.get("last_image_strength", 1.0)

    if raw_items:
        # Format items[] — per-item params
        items = []
        for i, raw in enumerate(raw_items):
            if not isinstance(raw, dict):
                return [], f"items[{i}] doit etre un objet"
            image = raw.get("image")
            if not image:
                return [], f"items[{i}].image est requis"
            items.append({
                "id": raw.get("id"),
                "image": image,
                "camera_motion": raw.get("camera_motion", raw.get("camera", shared_camera)),
                "seed": raw.get("seed", random.randint(0, 2**32 - 1)),
                "last_image": raw.get("last_image"),
                "last_image_strength": raw.get("last_image_strength", shared_last_image_strength),
                "_original_index": i,
            })
        return items, None

    elif raw_images:
        # Format images[] — params partages du top-level
        base_seed = shared_seed if shared_seed is not None else random.randint(0, 2**32 - 1)
        items = []
        for i, img in enumerate(raw_images):
            items.append({
                "id": None,
                "image": img,
                "camera_motion": shared_camera,
                "seed": base_seed + i,
                "last_image": shared_last_image,
                "last_image_strength": shared_last_image_strength,
                "_original_index": i,
            })
        return items, None

    elif raw_image:
        # Format image — single item
        seed = shared_seed if shared_seed is not None else random.randint(0, 2**32 - 1)
        return [{
            "id": None,
            "image": raw_image,
            "camera_motion": shared_camera,
            "seed": seed,
            "last_image": shared_last_image,
            "last_image_strength": shared_last_image_strength,
            "_original_index": 0,
        }], None

    else:
        return [], "Le champ 'image', 'images' ou 'items' est requis (URL https ou base64)"


# --- Handler ---

pipeline: MedusaPipeline | None = None


def handler(job: dict) -> dict:
    """Handler RunPod pour generation I2V.

    Supporte 3 formats d'input:
      - items[] : batch multi-client (per-item camera_motion, seed, last_image)
      - images[] : batch avec params partages (retro-compatible)
      - image : single item (retro-compatible)

    Les items sont regroupes par camera_motion (= meme prompt = memes embeddings).
    Chaque groupe est traite via generate_batch_frames() ou generate_frames().
    """
    global pipeline

    job_id = job.get("id", f"unknown-{int(time.time())}")
    job_input = job.get("input", {})

    log.info("Job %s — input: %s", job_id, json.dumps(job_input, default=str))

    # --- Normalize input (3 formats → liste uniforme) ---
    normalized, error = normalize_items(job_input)
    if error:
        return {"error": error}

    # --- Params partages (top-level) ---
    resolution = job_input.get("resolution", "720p")
    if resolution not in FIXED_RESOLUTIONS:
        return {"error": f"Resolution inconnue: {resolution}. Choix: {list(FIXED_RESOLUTIONS.keys())}"}
    dynamic_resolution = job_input.get("dynamic_resolution", False)
    num_frames = job_input.get("num_frames", 25)
    frame_rate = job_input.get("frame_rate", 24)
    image_strength = job_input.get("image_strength", 1.0)
    negative_override = job_input.get("negative_prompt")
    two_stage = resolution == "1080p"

    # --- Pipeline deja init au startup (eager init) ---
    if pipeline is None:
        raise RuntimeError("Pipeline non initialise — le worker doit etre lance via __main__")

    disk_before = get_disk_usage_mb()
    total_items = len(normalized)
    log.info("Job %s - %d item(s), disque avant: %.0f MB", job_id, total_items, disk_before)

    tmp_files: list[str] = []

    try:
        # --- Download ALL images en parallele (image + last_image) ---
        all_urls: list[str] = []
        url_map: list[tuple[int, str]] = []  # (item_index, field_name)
        for i, item in enumerate(normalized):
            all_urls.append(item["image"])
            url_map.append((i, "_image_path"))
            if item.get("last_image"):
                all_urls.append(item["last_image"])
                url_map.append((i, "_last_image_path"))

        with ThreadPoolExecutor(max_workers=min(len(all_urls), 8)) as dl_pool:
            paths = list(dl_pool.map(resolve_image, all_urls))

        for (item_idx, field), path in zip(url_map, paths):
            normalized[item_idx][field] = path
            tmp_files.append(path)

        # --- Resolution cible (une seule fois, premiere image) ---
        height, width = compute_target_resolution(
            resolution=resolution, image_path=normalized[0]["_image_path"], dynamic=dynamic_resolution,
        )
        log.info(
            "Resolution cible: %dx%d (%s, %s%s)",
            width, height, resolution, "2-stage" if two_stage else "1-stage",
            ", dynamic" if dynamic_resolution else "",
        )

        # --- Groupement par prompt (camera_motion → texte) ---
        groups: dict[str, list[dict]] = {}
        for item in normalized:
            prompt_text = CAMERA_PRESETS.get(item["camera_motion"], item["camera_motion"])
            item["_prompt_text"] = prompt_text
            groups.setdefault(prompt_text, []).append(item)

        log.info(
            "Groupes par prompt: %d groupe(s) — %s",
            len(groups), {k[:30]: len(v) for k, v in groups.items()},
        )

        # --- Generation par groupe ---
        results_by_index: dict[int, tuple] = {}  # index → (future, id)

        for prompt_text, group_items in groups.items():
            sub_batches = [
                group_items[i:i + MAX_BATCH]
                for i in range(0, len(group_items), MAX_BATCH)
            ]

            for batch_idx, sub_batch in enumerate(sub_batches):
                start_time = time.time()

                # Padding : toujours envoyer MAX_BATCH items au transformer pour eviter
                # les recompilations Dynamo (shape fixe dans le compile cache)
                real_count = len(sub_batch)
                if real_count < MAX_BATCH:
                    sub_batch_padded = sub_batch + [sub_batch[-1]] * (MAX_BATCH - real_count)
                else:
                    sub_batch_padded = sub_batch

                pipeline_items = []
                for item in sub_batch_padded:
                    pi: dict = {"image_path": item["_image_path"], "seed": item["seed"]}
                    if item.get("_last_image_path"):
                        pi["last_image_path"] = item["_last_image_path"]
                        pi["last_image_strength"] = item.get("last_image_strength", 1.0)
                    pipeline_items.append(pi)

                # Callback : soumettre le MP4 encode + S3 upload des que
                # le VAE decode un item, pendant que le VAE decode les suivants
                def on_decoded(batch_idx: int, frames: list) -> None:
                    if batch_idx >= real_count:
                        return  # Ignorer items padding
                    item = sub_batch[batch_idx]
                    orig_idx = item["_original_index"]
                    output_dir = tempfile.mkdtemp(prefix="medusa_")

                    if item.get("id"):
                        output_filename = f"medusa_i2v_{item['id']}.mp4"
                    elif total_items > 1:
                        output_filename = f"medusa_i2v_{job_id}_{orig_idx}.mp4"
                    else:
                        output_filename = f"medusa_i2v_{job_id}.mp4"

                    fut = postprocess_pool.submit(
                        postprocess_and_upload,
                        frames, int(frame_rate),
                        output_dir, output_filename,
                        job_id,
                    )
                    results_by_index[orig_idx] = (fut, item.get("id"))

                pipeline.generate_batch_frames(
                    items=pipeline_items,
                    prompt=prompt_text,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    frame_rate=frame_rate,
                    image_strength=image_strength,
                    two_stage=two_stage,
                    on_item_decoded=on_decoded,
                )

                elapsed = time.time() - start_time
                log.info(
                    "Groupe '%s' sub-batch %d/%d: %.1fs (%d item(s), %d padded)",
                    prompt_text[:30], batch_idx + 1, len(sub_batches), elapsed,
                    real_count, MAX_BATCH - real_count,
                )

        # --- Reordonner les resultats par _original_index ---
        ordered_results = []
        for idx in range(total_items):
            fut, item_id = results_by_index[idx]
            result = fut.result()
            if item_id is not None:
                result["id"] = item_id
            ordered_results.append(result)

        disk_after = get_disk_usage_mb()
        log.info("Disque apres: %.0f MB (libere: %.0f MB)", disk_after, disk_before - disk_after)
        return {"images": ordered_results}

    except Exception as e:
        log.error("Erreur generation: %s", e, exc_info=True)
        return {"error": str(e)}

    finally:
        # Cleanup toutes les images temporaires
        for tmp in tmp_files:
            if os.path.isfile(tmp):
                os.unlink(tmp)


# --- Init & Start ---

def init_pipeline() -> MedusaPipeline:
    """Initialise le pipeline au demarrage du worker."""
    log.info("Initialisation MedusaPipeline...")
    p = MedusaPipeline(models_dir=MODELS_DIR)

    # 1. Warmup embeddings (depuis cache disque)
    embeddings_cache_dir = os.path.join(CACHE_DIR, "embeddings")
    os.makedirs(embeddings_cache_dir, exist_ok=True)
    p.warmup_embeddings(embeddings_cache_dir)

    # 2. Build transformer distilled (FP8 cast)
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
