"""Constantes partagees entre pipeline.py et warmup_embeddings.py."""

DEFAULT_NEGATIVE_PROMPT = (
    "blurry, out of focus, low quality, distorted, watermark, "
    "logo, text, subtitle, banner, signature, username, "
    "compressed artifacts, jpeg artifacts, noise, grainy"
)

CAMERA_PRESETS: dict[str, str] = {
    "dolly-in": "The camera slowly moves forward into the scene",
    "dolly-out": "The camera slowly pulls back from the scene",
    "dolly-left": "The camera smoothly translates to the left",
    "dolly-right": "The camera smoothly translates to the right",
    "jib-up": "The camera rises vertically",
    "jib-down": "The camera descends vertically",
    "static": "Static camera, no movement",
}
