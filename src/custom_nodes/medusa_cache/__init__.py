"""
Cache d'embeddings CLIP persistant pour Medusa I2V.

Remplace CLIPTextEncode avec un cache .pt sur disque :
- Cache hit : charge le tensor depuis .pt (~0s)
- Cache miss : encode avec CLIP, sauvegarde .pt local + volume

Le cache local (NVMe) est ephemere mais rapide.
Le cache volume (NFS) persiste entre les cold starts.
"""

import hashlib
import os

import torch

LOCAL_CACHE_DIR = "/ComfyUI/cache/embeddings"
VOLUME_CACHE_DIR = os.environ.get(
    "EMBEDDING_CACHE_DIR", "/runpod-volume/cache/embeddings"
)


class CachedCLIPTextEncode:
    """CLIPTextEncode avec cache .pt persistant."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "clip": ("CLIP",),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "medusa"

    def encode(self, clip, text):
        cache_key = hashlib.sha256(text.encode()).hexdigest()[:16]

        # Cache lookup : local (NVMe) d'abord, puis volume (NFS)
        for cache_dir in [LOCAL_CACHE_DIR, VOLUME_CACHE_DIR]:
            cache_path = os.path.join(cache_dir, f"{cache_key}.pt")
            if os.path.isfile(cache_path):
                try:
                    conditioning = torch.load(
                        cache_path, map_location="cpu", weights_only=False
                    )
                    print(f"[medusa] Embedding cache hit: {cache_key}")
                    return (conditioning,)
                except Exception as e:
                    print(f"[medusa] Cache corrompu {cache_path}: {e}")

        # Cache miss : encodage CLIP
        print(f"[medusa] Embedding cache miss: {cache_key}, encoding...")
        tokens = clip.tokenize(text)
        output = clip.encode_from_tokens(
            tokens, return_pooled=True, return_dict=True
        )
        cond = output.pop("cond")
        conditioning = [[cond, output]]

        # Sauvegarde local + volume
        for cache_dir in [LOCAL_CACHE_DIR, VOLUME_CACHE_DIR]:
            cache_path = os.path.join(cache_dir, f"{cache_key}.pt")
            try:
                os.makedirs(cache_dir, exist_ok=True)
                torch.save(conditioning, cache_path)
                print(f"[medusa] Embedding sauvegarde: {cache_path}")
            except OSError as e:
                print(f"[medusa] Erreur sauvegarde {cache_path}: {e}")

        return (conditioning,)


NODE_CLASS_MAPPINGS = {
    "CachedCLIPTextEncode": CachedCLIPTextEncode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CachedCLIPTextEncode": "CLIP Text Encode (Cached)",
}
