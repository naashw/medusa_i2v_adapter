"""
MedusaPipeline — ltx-pipelines direct inference for LTX-2 19B I2V.

Remplace ComfyUI par un appel Python direct a ltx-core / ltx-pipelines.
Gestion du lifecycle des modeles entre jobs :
  - Video encoder  : persistent en VRAM (~1GB)
  - Transformer    : base (distilled + I2V) persistent en VRAM, camera = delta applique en place
  - Camera LoRAs   : lazy-load en RAM CPU a la demande, delta applique/annule en ~2-5s
  - Text encoder   : charge sur CPU au warmup, embeddings caches sur disque
  - Video decoder  : charge/decharge par job (libere VRAM)
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator

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

# Imports pour le mecanisme de delta camera (Phase 2)
from ltx_core.loader.fuse_loras import _prepare_deltas, _fuse_delta_with_cast_fp8
from ltx_core.loader.primitives import LoraStateDictWithStrength, StateDict
from ltx_core.loader.sft_loader import SafetensorsModelStateDictLoader

log = logging.getLogger("medusa")

# --- Monkey-patch : strip .weight_scale keys pour forcer le path cast FP8 ---
# Avec fp8_cast(), les poids sont charges en format cast (non transposes),
# mais les cles .weight_scale du checkpoint ne sont pas supprimees.
# Leur presence fait croire a apply_loras que les poids sont en format scaled
# (transposes), ce qui provoque un shape mismatch lors du merge LoRA.
import ltx_core.loader.fuse_loras as _fuse_loras_mod  # noqa: E402

_orig_apply_loras = _fuse_loras_mod.apply_loras


def _apply_loras_strip_scales(
    model_sd: "StateDict",
    lora_sd_and_strengths: list,
    dtype: torch.dtype | None = None,
    destination_sd: "StateDict | None" = None,
) -> "StateDict":
    scale_keys = [k for k in model_sd.sd if k.endswith(".weight_scale")]
    if scale_keys:
        log.info("Stripping %d .weight_scale keys (force cast FP8 path)", len(scale_keys))
        for k in scale_keys:
            del model_sd.sd[k]
    return _orig_apply_loras(model_sd, lora_sd_and_strengths, dtype, destination_sd)


_fuse_loras_mod.apply_loras = _apply_loras_strip_scales

# Patch aussi le binding local dans single_gpu_model_builder
# (from ... import apply_loras cree une reference locale non affectee par le patch module)
import ltx_core.loader.single_gpu_model_builder as _builder_mod  # noqa: E402

_builder_mod.apply_loras = _apply_loras_strip_scales

# --- Monkey-patch : charger les LoRAs sur CPU pour eviter l'OOM VRAM ---
# Modele FP8 ~23GB + LoRAs BF16 ~12.1GB > 24GB VRAM.
# _prepare_deltas (fuse_loras.py) fait .to(device) cle-par-cle → streaming GPU automatique.
_orig_sgmb_build = _builder_mod.SingleGPUModelBuilder.build


def _patched_sgmb_build(self, device=None, dtype=None):
    """Charge les LoRAs sur CPU plutot que GPU pour eviter l'OOM pendant le build."""
    if not self.loras:
        return _orig_sgmb_build(self, device=device, dtype=dtype)

    device_arg = torch.device("cuda") if device is None else device
    lora_paths = {lora.path for lora in self.loras}
    orig_load_sd = self.load_sd

    def _cpu_load_sd(paths, **kwargs):
        if len(paths) == 1 and paths[0] in lora_paths:
            kwargs["device"] = torch.device("cpu")
            log.info("LoRA charge sur CPU (streaming): %s", os.path.basename(paths[0]))
        return orig_load_sd(paths, **kwargs)

    # SingleGPUModelBuilder est un @dataclass(frozen=True) — utiliser object.__setattr__
    object.__setattr__(self, "load_sd", _cpu_load_sd)
    try:
        return _orig_sgmb_build(self, device=device_arg, dtype=dtype)
    finally:
        object.__setattr__(self, "load_sd", orig_load_sd)


_builder_mod.SingleGPUModelBuilder.build = _patched_sgmb_build

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

        # Transformer de base en VRAM (distilled + I2V fusionnes, sans camera LoRA)
        # Build une seule fois au cold start, camera delta applique/annule en place.
        self._base_transformer: torch.nn.Module | None = None
        self._current_camera_lora: str | None = None

        # Camera LoRAs preloadees en RAM CPU (key = lora_path absolu)
        self._camera_loras_ram: dict[str, StateDict] = {}

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

    def _build_base_transformer(self) -> None:
        """Build transformer avec distilled + I2V fusionnes (sans camera LoRA). Une seule fois."""
        log.info("Build transformer de base (distilled + I2V, sans camera)...")

        gpu_ledger = ModelLedger(
            dtype=self.dtype,
            device=self.device,
            checkpoint_path=self._checkpoint_path,
            loras=self._base_loras,
            quantization=QuantizationPolicy.fp8_cast(),
        )

        if self._base_transformer is not None:
            del self._base_transformer
            cleanup_memory()

        self._base_transformer = gpu_ledger.transformer()
        del gpu_ledger
        cleanup_memory()
        log.info("Transformer de base pret en VRAM.")

    def _preload_camera_loras(self, camera_loras_paths: dict[str, str]) -> None:
        """Charge toutes les camera LoRAs en RAM CPU.

        Args:
            camera_loras_paths: dict mapping camera_key -> lora_file_path absolu.
        """
        loader = SafetensorsModelStateDictLoader()
        for camera_key, path in camera_loras_paths.items():
            log.info("Preload camera LoRA en RAM: %s", os.path.basename(path))
            sd = loader.load([path], sd_ops=LTXV_LORA_COMFY_RENAMING_MAP, device=torch.device("cpu"))
            self._camera_loras_ram[path] = sd
        log.info("Camera LoRAs preloadees en RAM: %d", len(self._camera_loras_ram))

    def _lazy_load_camera_lora(self, path: str) -> None:
        """Charge une camera LoRA en RAM CPU a la demande (lazy).

        SafetensorsStateDictLoader utilise safe_open avec copy=False → mmap natif.
        """
        if path in self._camera_loras_ram:
            return
        log.info("Lazy-load camera LoRA: %s", os.path.basename(path))
        loader = SafetensorsModelStateDictLoader()
        sd = loader.load([path], sd_ops=LTXV_LORA_COMFY_RENAMING_MAP, device=torch.device("cpu"))
        self._camera_loras_ram[path] = sd

    def _apply_camera_delta(self, camera_path: str, sign: int = 1) -> None:
        """Applique ou annule un camera LoRA delta sur le transformer de base.

        Args:
            camera_path: Chemin absolu du fichier camera LoRA.
            sign: +1 pour appliquer, -1 pour annuler.
        """
        lora_sd = self._camera_loras_ram[camera_path]
        strength = CAMERA_LORA_STRENGTH * sign
        lora_item = [LoraStateDictWithStrength(lora_sd, strength)]
        model = self._base_transformer

        for name, param in model.named_parameters():
            if not name.endswith(".weight"):
                continue
            delta = _prepare_deltas(lora_item, name, torch.bfloat16, param.device)
            if delta is None:
                continue
            if param.data.dtype == torch.float8_e4m3fn:
                result = _fuse_delta_with_cast_fp8(
                    delta, param.data, name, torch.float8_e4m3fn, param.device
                )
                param.data = result[name]
            else:
                param.data = (param.data.to(torch.bfloat16) + delta).to(param.data.dtype)

    def _get_transformer(self, camera_lora_path: str) -> torch.nn.Module:
        """Retourne le transformer avec le bon camera delta applique.

        Si la camera LoRA est preloadee en RAM, le switch se fait en ~2-5s
        (undo delta precedent + apply nouveau delta).
        Sinon, fallback sur rebuild complet via _get_transformer_rebuild.
        """
        if camera_lora_path == self._current_camera_lora and self._base_transformer is not None:
            return self._base_transformer

        # Lazy-load si pas encore en RAM
        self._lazy_load_camera_lora(camera_lora_path)

        # Undo camera precedente
        if self._current_camera_lora is not None and self._current_camera_lora in self._camera_loras_ram:
            log.info("Undo camera delta: %s", os.path.basename(self._current_camera_lora))
            self._apply_camera_delta(self._current_camera_lora, sign=-1)

        # Apply nouvelle camera
        log.info("Apply camera delta: %s", os.path.basename(camera_lora_path))
        self._apply_camera_delta(camera_lora_path, sign=+1)
        self._current_camera_lora = camera_lora_path
        return self._base_transformer

    def _get_transformer_rebuild(self, camera_lora_path: str) -> torch.nn.Module:
        """Fallback : rebuild complet du transformer avec la camera LoRA donnee.

        Utilise uniquement si la camera LoRA n'est pas preloadee en RAM.
        Build + merge LoRAs directement en VRAM via destination_sd (in-place).
        DummyRegistry (defaut) permet le merge sans doubler la memoire.
        """
        log.info("Rebuild transformer avec LoRA: %s", os.path.basename(camera_lora_path))

        all_loras = [
            *self._base_loras,
            LoraPathStrengthAndSDOps(camera_lora_path, CAMERA_LORA_STRENGTH, LTXV_LORA_COMFY_RENAMING_MAP),
        ]

        gpu_ledger = ModelLedger(
            dtype=self.dtype,
            device=self.device,
            checkpoint_path=self._checkpoint_path,
            loras=all_loras,
            quantization=QuantizationPolicy.fp8_cast(),
        )

        if self._base_transformer is not None:
            del self._base_transformer
            cleanup_memory()

        log.info("Build transformer + merge LoRAs sur GPU...")
        self._base_transformer = gpu_ledger.transformer()
        self._current_camera_lora = camera_lora_path

        del gpu_ledger
        cleanup_memory()

        log.info("Transformer pret en VRAM.")
        return self._base_transformer

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
