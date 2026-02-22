"""Constantes partagees entre pipeline.py et warmup_embeddings.py."""

DEFAULT_NEGATIVE_PROMPT = (
    "blurry, out of focus, low quality, distorted, watermark, "
    "logo, text, subtitle, banner, signature, username, "
    "compressed artifacts, jpeg artifacts, noise, grainy"
)

CAMERA_PROMPTS: dict[str, str] = {
    "dolly-in": "A steady dolly-in camera movement, smooth forward motion, cinematic.",
    "dolly-out": "A steady dolly-out camera movement, smooth backward motion, cinematic.",
    "dolly-left": "A steady dolly-left camera movement, smooth lateral motion to the left, cinematic.",
    "dolly-right": "A steady dolly-right camera movement, smooth lateral motion to the right, cinematic.",
    "jib-down": "A steady jib-down camera movement, smooth downward motion, cinematic.",
    "jib-up": "A steady jib-up camera movement, smooth upward motion, cinematic.",
    "static": "A static camera, no movement, cinematic.",
}
