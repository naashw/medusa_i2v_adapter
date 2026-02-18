"""
MedusaPipeline — ltx-pipelines direct inference for LTX-2 19B I2V.

Remplace ComfyUI par un appel Python direct a ltx-core / ltx-pipelines.
Gestion du lifecycle des modeles entre jobs :
  - Video encoder : persistent en VRAM (~1GB)
  - Transformer   : cache par camera LoRA, reste en VRAM entre jobs
  - Text encoder  : charge sur CPU au warmup, embeddings caches sur disque
  - Video decoder : charge/decharge par job (libere VRAM)
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from pathlib import Path

import torch

from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.guiders import MultiModalGuider, MultiModalGuiderParams
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.protocols import DiffusionStepProtocol
from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_core.model.video_vae import TilingConfig
from ltx_core.model.video_vae import decode_video as vae_decode_video
from ltx_core.quantization import QuantizationPolicy
from ltx_core.text_encoders.gemma import encode_text
from ltx_core.types import LatentState, VideoPixelShape
from ltx_pipelines.utils import ModelLedger
from ltx_pipelines.utils.constants import DISTILLED_SIGMA_VALUES
from ltx_pipelines.utils.helpers import (
    cleanup_memory,
    denoise_audio_video,
    euler_denoising_loop,
    image_conditionings_by_replacing_latent,
    multi_modal_guider_denoising_func,
)
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.types import PipelineComponents

log = logging.getLogger("medusa")

# --- Monkey-patch : fallback CPU pour _fuse_delta_with_cast_fp8 ---
# La version installee utilise calculate_weight_float8 (kernel Triton, CUDA-only).
# Ce patch ajoute le path CPU present dans les versions recentes de ltx-core.
import ltx_core.loader.fuse_loras as _fuse_loras_mod  # noqa: E402

_orig_fuse_cast = _fuse_loras_mod._fuse_delta_with_cast_fp8


def _cpu_safe_fuse_delta_with_cast_fp8(
    deltas: torch.Tensor,
    weight: torch.Tensor,
    key: str,
    target_dtype: torch.dtype,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if str(device).startswith("cuda"):
        return _orig_fuse_cast(deltas, weight, key, target_dtype, device)
    # CPU path : dequant FP8→BF16, add delta, recast
    deltas.add_(weight.to(dtype=deltas.dtype, device=device))
    return {key: deltas.to(dtype=target_dtype)}


_fuse_loras_mod._fuse_delta_with_cast_fp8 = _cpu_safe_fuse_delta_with_cast_fp8

# LoRA strengths (matches current ComfyUI workflow)
DISTILLED_LORA_STRENGTH = 0.7
I2V_ADAPTER_STRENGTH = 0.8
CAMERA_LORA_STRENGTH = 1.0

# Audio skip : audio calcule seulement au step 0 / 8
AUDIO_SKIP_STEP = 99

# Negative prompt par defaut
DEFAULT_NEGATIVE_PROMPT = (
    "blurry, out of focus, low quality, distorted, watermark, "
    "logo, text, subtitle, banner, signature, username, "
    "compressed artifacts, jpeg artifacts, noise, grainy"
)

# Prompts camera standardises
CAMERA_PROMPTS: dict[str, str] = {
    "dolly-in": "A steady dolly-in camera movement, smooth forward motion, cinematic.",
    "dolly-out": "A steady dolly-out camera movement, smooth backward motion, cinematic.",
    "dolly-left": "A steady dolly-left camera movement, smooth lateral motion to the left, cinematic.",
    "dolly-right": "A steady dolly-right camera movement, smooth lateral motion to the right, cinematic.",
    "jib-down": "A steady jib-down camera movement, smooth downward motion, cinematic.",
    "jib-up": "A steady jib-up camera movement, smooth upward motion, cinematic.",
    "static": "A static camera, no movement, cinematic.",
}


class MedusaPipeline:
    """Pipeline I2V utilisant ltx-pipelines avec audio skip et transformer cache."""

    def __init__(self, models_dir: str, device: torch.device | None = None) -> None:
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.dtype = torch.bfloat16
        self.models_dir = models_dir

        # Paths modeles
        self._checkpoint_path = os.path.join(models_dir, "checkpoints", "ltx-2-19b-dev-fp8.safetensors")
        self._gemma_root = os.path.join(models_dir, "text_encoders", "gemma-3-12b-it")
        self._distilled_lora = os.path.join(models_dir, "loras", "ltx-2-19b-distilled-lora-384.safetensors")
        self._i2v_adapter = os.path.join(models_dir, "loras", "LTX-2-Image2Vid-Adapter.safetensors")

        # Pipeline components (patchifiers + scale factors)
        self._components = PipelineComponents(dtype=self.dtype, device=self.device)

        # Base LoRAs (distilled + i2v adapter, utilisees pour chaque transformer build)
        self._base_loras = [
            LoraPathStrengthAndSDOps(self._distilled_lora, DISTILLED_LORA_STRENGTH, LTXV_LORA_COMFY_RENAMING_MAP),
            LoraPathStrengthAndSDOps(self._i2v_adapter, I2V_ADAPTER_STRENGTH, LTXV_LORA_COMFY_RENAMING_MAP),
        ]

        # Base ModelLedger SANS LoRAs — uniquement pour video_encoder et video_decoder (VAE)
        self._base_ledger = ModelLedger(
            dtype=self.dtype,
            device=self.device,
            checkpoint_path=self._checkpoint_path,
            quantization=QuantizationPolicy.fp8_cast(),
        )

        # Video encoder (charge apres warmup embeddings pour eviter OOM)
        self._video_encoder: torch.nn.Module | None = None

        # Cache transformer (par camera LoRA)
        self._transformer: torch.nn.Module | None = None
        self._current_camera_lora: str | None = None

        # Embeddings cache (prompt -> (video_ctx, audio_ctx))
        self._embeddings_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

        # Sigmas distilled (8 steps)
        self._sigmas = torch.tensor(
            DISTILLED_SIGMA_VALUES, dtype=torch.float32, device=self.device
        )

    def warmup_embeddings(self, cache_dir: str) -> None:
        """Encode tous les prompts camera + negative, sauvegarde sur disque.

        Charge le text encoder Gemma 3 12B sur CPU (~24GB RAM),
        encode les prompts, sauvegarde le cache, puis libere la memoire.
        """
        cache_path = os.path.join(cache_dir, "embeddings_cache.pt")

        # Si cache existe deja, charger directement
        if os.path.isfile(cache_path):
            log.info("Chargement embeddings depuis cache: %s", cache_path)
            self._load_embeddings_cache(cache_path)
            return

        log.info("Pas de cache embeddings — generation avec text encoder sur CPU...")

        # Creer un ledger CPU dedie au text encoder
        cpu_ledger = ModelLedger(
            dtype=self.dtype,
            device=torch.device("cpu"),
            checkpoint_path=self._checkpoint_path,
            gemma_root_path=self._gemma_root,
        )
        text_encoder = cpu_ledger.text_encoder()

        # Encoder tous les prompts : 7 cameras + 1 negative
        all_prompts = list(CAMERA_PROMPTS.values()) + [DEFAULT_NEGATIVE_PROMPT]
        all_keys = list(CAMERA_PROMPTS.keys()) + ["_negative"]

        log.info("Encoding %d prompts sur CPU...", len(all_prompts))
        results = encode_text(text_encoder, prompts=all_prompts)

        # Construire le cache
        cache_data: dict[str, dict[str, torch.Tensor]] = {}
        for key, (v_ctx, a_ctx) in zip(all_keys, results):
            cache_data[key] = {"video": v_ctx.cpu(), "audio": a_ctx.cpu()}
            self._embeddings_cache[key] = (v_ctx.to(self.device), a_ctx.to(self.device))

        # Sauvegarder sur disque
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(cache_data, cache_path)
        log.info("Embeddings sauvegardes: %s (%d prompts)", cache_path, len(cache_data))

        # Liberer le text encoder
        del text_encoder
        del cpu_ledger
        cleanup_memory()

    def load_video_encoder(self) -> None:
        """Charge le video encoder en VRAM (~1GB). Appeler apres warmup_embeddings."""
        log.info("Chargement video encoder (persistent)...")
        self._video_encoder = self._base_ledger.video_encoder()

    def _load_embeddings_cache(self, cache_path: str) -> None:
        """Charge les embeddings depuis un fichier .pt."""
        cache_data = torch.load(cache_path, map_location="cpu", weights_only=True)
        for key, tensors in cache_data.items():
            self._embeddings_cache[key] = (
                tensors["video"].to(self.device),
                tensors["audio"].to(self.device),
            )
        log.info("Embeddings charges: %d prompts", len(self._embeddings_cache))

    def _get_transformer(self, camera_lora_path: str) -> torch.nn.Module:
        """Retourne le transformer avec le bon camera LoRA, cache en VRAM.

        Build sur CPU (RAM) pour eviter OOM VRAM pendant le merge LoRA,
        puis transfert du modele merge (~20GB FP8) vers GPU.
        Le monkey-patch _cpu_safe_fuse_delta_with_cast_fp8 gere les ops FP8 sur CPU.
        """
        if camera_lora_path == self._current_camera_lora and self._transformer is not None:
            return self._transformer

        log.info("Chargement transformer avec LoRA: %s", os.path.basename(camera_lora_path))

        # Toutes les LoRAs : distilled + i2v + camera
        all_loras = [
            *self._base_loras,
            LoraPathStrengthAndSDOps(camera_lora_path, CAMERA_LORA_STRENGTH, LTXV_LORA_COMFY_RENAMING_MAP),
        ]

        # Build sur CPU avec fp8_cast (le monkey-patch ajoute le fallback CPU)
        cpu_ledger = ModelLedger(
            dtype=self.dtype,
            device=torch.device("cpu"),
            checkpoint_path=self._checkpoint_path,
            loras=all_loras,
            quantization=QuantizationPolicy.fp8_cast(),
        )

        # Liberer l'ancien transformer
        if self._transformer is not None:
            del self._transformer
            cleanup_memory()

        log.info("Build transformer + merge LoRAs sur CPU...")
        transformer = cpu_ledger.transformer()

        log.info("Transfert vers GPU...")
        self._transformer = transformer.to(self.device)
        self._current_camera_lora = camera_lora_path

        # Liberer le ledger CPU
        del cpu_ledger
        cleanup_memory()

        log.info("Transformer pret en VRAM.")
        return self._transformer

    @torch.inference_mode()
    def generate(
        self,
        image_path: str,
        camera_lora_path: str,
        camera_key: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        output_path: str,
        image_strength: float = 1.0,
        prompt_override: str | None = None,
        negative_override: str | None = None,
    ) -> None:
        """Genere une video I2V et sauvegarde en MP4.

        Args:
            image_path: Chemin vers l'image source (fichier local).
            camera_lora_path: Chemin vers le safetensors du camera LoRA.
            camera_key: Cle camera (dolly-in, etc.) pour lookup embeddings.
            seed: Seed pour la generation.
            height: Hauteur output (multiple de 32).
            width: Largeur output (multiple de 32).
            num_frames: Nombre de frames (k*8+1).
            frame_rate: FPS output.
            output_path: Chemin de sortie MP4.
            image_strength: Force du conditioning image (0-1).
            prompt_override: Prompt custom (sinon utilise le cache camera).
            negative_override: Negative prompt custom.
        """
        generator = torch.Generator(device=self.device).manual_seed(seed)

        # 1. Embeddings depuis cache
        if prompt_override:
            v_context_p, a_context_p = self._encode_prompt(prompt_override)
        elif camera_key in self._embeddings_cache:
            v_context_p, a_context_p = self._embeddings_cache[camera_key]
        else:
            raise ValueError(f"Camera '{camera_key}' non trouvee dans le cache embeddings")

        neg_key = "_negative"
        if negative_override:
            v_context_n, a_context_n = self._encode_prompt(negative_override)
        elif neg_key in self._embeddings_cache:
            v_context_n, a_context_n = self._embeddings_cache[neg_key]
        else:
            raise ValueError("Negative prompt non trouve dans le cache embeddings")

        # 2. Transformer avec bon camera LoRA
        transformer = self._get_transformer(camera_lora_path)

        # 3. Setup denoising
        noiser = GaussianNoiser(generator=generator)
        stepper = EulerDiffusionStep()

        # 4. Image conditioning
        conditionings = image_conditionings_by_replacing_latent(
            images=[(image_path, 0, image_strength)],
            height=height,
            width=width,
            video_encoder=self._video_encoder,
            dtype=self.dtype,
            device=self.device,
        )

        # 5. Guiders : video (toujours actif) + audio (skip 87.5%)
        video_guider_params = MultiModalGuiderParams(
            cfg_scale=1.0,
            stg_scale=0.0,
            modality_scale=1.0,
            skip_step=0,
        )
        audio_guider_params = MultiModalGuiderParams(
            cfg_scale=1.0,
            stg_scale=0.0,
            modality_scale=1.0,
            skip_step=AUDIO_SKIP_STEP,
        )

        # Denoising loop closure
        def denoising_loop(
            sigmas: torch.Tensor,
            video_state: LatentState,
            audio_state: LatentState,
            stepper_arg: DiffusionStepProtocol,
        ) -> tuple[LatentState, LatentState]:
            return euler_denoising_loop(
                sigmas=sigmas,
                video_state=video_state,
                audio_state=audio_state,
                stepper=stepper_arg,
                denoise_fn=multi_modal_guider_denoising_func(
                    video_guider=MultiModalGuider(
                        params=video_guider_params,
                        negative_context=v_context_n,
                    ),
                    audio_guider=MultiModalGuider(
                        params=audio_guider_params,
                        negative_context=a_context_n,
                    ),
                    v_context=v_context_p,
                    a_context=a_context_p,
                    transformer=transformer,
                ),
            )

        # 6. Denoise
        output_shape = VideoPixelShape(
            batch=1, frames=num_frames, width=width, height=height, fps=frame_rate,
        )

        video_state, _audio_state = denoise_audio_video(
            output_shape=output_shape,
            conditionings=conditionings,
            noiser=noiser,
            sigmas=self._sigmas,
            stepper=stepper,
            denoising_loop_fn=denoising_loop,
            components=self._components,
            dtype=self.dtype,
            device=self.device,
        )

        # 7. VAE decode (tiled pour eviter OOM 720p)
        log.info("VAE decode tiled...")
        video_decoder = self._base_ledger.video_decoder()
        decoded_video: Iterator[torch.Tensor] = vae_decode_video(
            video_state.latent,
            video_decoder,
            TilingConfig.default(),
            generator,
        )

        # 8. Encode MP4 (sans audio)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        encode_video(
            video=decoded_video,
            fps=int(frame_rate),
            audio=None,
            audio_sample_rate=None,
            output_path=output_path,
            video_chunks_number=1,
        )
        log.info("Video sauvegardee: %s", output_path)

        # Liberer le video decoder
        del video_decoder
        cleanup_memory()

    def _encode_prompt(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode un prompt custom via le text encoder (cold path, rarement utilise)."""
        log.warning("Encoding prompt custom sur CPU (lent): %s", prompt[:50])
        cpu_ledger = ModelLedger(
            dtype=self.dtype,
            device=torch.device("cpu"),
            checkpoint_path=self._checkpoint_path,
            gemma_root_path=self._gemma_root,
        )
        text_encoder = cpu_ledger.text_encoder()
        results = encode_text(text_encoder, prompts=[prompt])
        v_ctx, a_ctx = results[0]
        del text_encoder
        del cpu_ledger
        cleanup_memory()
        return v_ctx.to(self.device), a_ctx.to(self.device)
