"""Fast MP4 encoder — preset veryfast, sans audio, sans tqdm."""

from __future__ import annotations

import gc
import logging
import os
import threading
import time
from collections.abc import Iterator

import av
import torch

log = logging.getLogger("video_encoder")

ENCODE_PRESET = os.environ.get("ENCODE_PRESET", "veryfast")
ENCODE_CRF = os.environ.get("ENCODE_CRF", "23")

# PyAV/libx264 n'est pas thread-safe pour encode/cleanup concurrent
_encode_lock = threading.Lock()


def encode_video_fast(
    video: torch.Tensor | Iterator[torch.Tensor],
    fps: int,
    output_path: str,
) -> None:
    """Encode frames → H264 MP4 avec preset configurable.

    Args:
        video: Tensor (N, H, W, 3) uint8 ou Iterator de chunks.
        fps: Framerate de sortie.
        output_path: Chemin du fichier MP4 de sortie.
    """
    t0 = time.perf_counter()

    if isinstance(video, torch.Tensor):
        video = iter([video])

    first_chunk = next(video)
    _, height, width, _ = first_chunk.shape

    with _encode_lock:
        stream = None
        container = av.open(output_path, mode="w")
        try:
            stream = container.add_stream(
                "libx264",
                rate=int(fps),
                options={"preset": ENCODE_PRESET, "crf": ENCODE_CRF},
            )
            stream.width = width
            stream.height = height
            stream.pix_fmt = "yuv420p"

            chunk_np = first_chunk.to("cpu").numpy()
            for frame_array in chunk_np:
                frame = av.VideoFrame.from_ndarray(frame_array, format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)

            for chunk in video:
                chunk_np = chunk.to("cpu").numpy()
                for frame_array in chunk_np:
                    frame = av.VideoFrame.from_ndarray(frame_array, format="rgb24")
                    for packet in stream.encode(frame):
                        container.mux(packet)

            # Flush encoder
            for packet in stream.encode():
                container.mux(packet)
        finally:
            container.close()
            # Casser le cycle stream↔container pour eviter que le GC cyclique
            # ne detruise les objets C de PyAV dans un ordre arbitraire en idle
            # (cause: abort dans avpriv_slicethread_free/pthread_cond_destroy)
            del stream, container
            gc.collect()

    elapsed = time.perf_counter() - t0
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    log.info(
        "Encoded %s — %dx%d, preset=%s, crf=%s, %.1f MB in %.2fs",
        os.path.basename(output_path), width, height,
        ENCODE_PRESET, ENCODE_CRF, size_mb, elapsed,
    )
