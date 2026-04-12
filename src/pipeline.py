"""
MedusaPipeline — ltx-pipelines direct inference for LTX-2.3 22B Distilled.

Remplace ComfyUI par un appel Python direct a ltx-core / ltx-pipelines.
Checkpoint distilled BF16 + QuantizationPolicy.fp8_cast() → stockage FP8, compute BF16.
Gestion du lifecycle des modeles entre jobs :
  - Video encoder  : persistent en VRAM (~1GB)
  - Video decoder  : persistent en VRAM (~2GB)
  - Transformer    : distilled persistante, FP8 cast (~19-20GB VRAM)
  - Text encoder   : charge sur GPU a la demande, embeddings caches sur disque
"""

from __future__ import annotations

import dataclasses
import gc
import hashlib
import logging
import math
import os
import time
import warnings
from collections.abc import Callable, Iterator

import json

import safetensors
import torch
import torch.nn.functional as F

# Supprimer les warnings internes torch.compile/dynamo (pas de bug, juste du bruit)
warnings.filterwarnings("ignore", message=".*lru_cache.*", module=r"torch\._dynamo")
warnings.filterwarnings("ignore", message=".*To copy construct from a tensor.*", module=r"torch\.")
from safetensors import safe_open
from safetensors.torch import save_file as save_safetensors

from ltx_core.components.diffusion_steps import EulerDiffusionStep, Res2sDiffusionStep
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.patchifiers import AudioPatchifier, VideoLatentPatchifier
from ltx_core.components.protocols import DiffusionStepProtocol
from ltx_core.conditioning import (
    ConditioningItem,
    VideoConditionByReferenceLatent,
)
from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP
from ltx_core.loader.sd_ops import SDOps
from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
from ltx_core.model.transformer import LTXModelConfigurator, LTXV_MODEL_COMFY_RENAMING_MAP, X0Model
from ltx_core.model.upsampler import LatentUpsamplerConfigurator, upsample_video
from ltx_core.model.video_vae import (
    TilingConfig,
    VideoDecoderConfigurator,
    VideoEncoderConfigurator,
    VAE_DECODER_COMFY_KEYS_FILTER,
    VAE_ENCODER_COMFY_KEYS_FILTER,
)
from ltx_core.quantization import QuantizationPolicy
from ltx_core.tools import AudioLatentTools, VideoLatentTools
from ltx_core.types import AudioLatentShape, LatentState, VideoLatentShape, VideoPixelShape

from ltx_pipelines.utils.constants import DISTILLED_SIGMA_VALUES, STAGE_2_DISTILLED_SIGMA_VALUES
from ltx_pipelines.utils.args import ImageConditioningInput
from ltx_pipelines.utils.helpers import (
    cleanup_memory,
    create_noised_state,
    image_conditionings_by_replacing_latent,
    image_conditionings_by_adding_guiding_latent,
    modality_from_latent_state,
)
from ltx_pipelines.utils.samplers import euler_denoising_loop

from camera_path import interpolate_camera_path, quat_to_matrix
from prompts import DEFAULT_NEGATIVE_PROMPT

log = logging.getLogger("medusa")

# Log SDPA backend (cuDNN Fused Flash Attention natif sur H100, pas de dep externe)
if torch.cuda.is_available():
    log.info("SDPA backend: cuDNN attention (natif PyTorch, H100 sm_90)")


# Version du cache transformer (incrementer pour invalider tous les caches existants)
CACHE_VERSION = "v5"


class _MedusaDenoiser:
    """Denoiser sans guidance (CFG=1, STG=0) compatible protocol Denoiser LTX-2.

    Encapsule les embeddings video/audio et appelle le transformer directement.
    Logs VRAM et timing au step 0, warning si recompilation detectee.
    """

    def __init__(self, v_context: torch.Tensor, a_context: torch.Tensor) -> None:
        self.v_context = v_context
        self.a_context = a_context

    def __call__(
        self,
        transformer: torch.nn.Module,
        video_state: LatentState | None,
        audio_state: LatentState | None,
        sigmas: torch.Tensor,
        step_index: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        t0 = time.perf_counter()
        sigma = sigmas[step_index]
        vid_mod = modality_from_latent_state(video_state, self.v_context, sigma, enabled=True)
        aud_mod = modality_from_latent_state(audio_state, self.a_context, sigma, enabled=False)
        torch.compiler.cudagraph_mark_step_begin()
        result = transformer(video=vid_mod, audio=aud_mod, perturbations=None)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        if step_index == 0:
            alloc = torch.cuda.memory_allocated() / 2**30
            rsvd = torch.cuda.memory_reserved() / 2**30
            log.info("step %d: %.2fs (sigma=%.4f) VRAM %.2fGB alloc %.2fGB rsvd",
                     step_index, dt, sigma.item(), alloc, rsvd)
        else:
            log.debug("step %d: %.2fs (sigma=%.4f)", step_index, dt, sigma.item())
        if dt > 10.0:
            log.warning("step %d took %.1fs — possible Dynamo recompilation", step_index, dt)
        return result


class _MedusaBatchDenoiser:
    """Denoiser batch — expand sigma pour batch_size > 1.

    Logs VRAM, timing et Inductor cache counters au step 0.
    """

    def __init__(self, v_context: torch.Tensor, a_context: torch.Tensor, batch_size: int) -> None:
        self.v_context = v_context
        self.a_context = a_context
        self.batch_size = batch_size

    def __call__(
        self,
        transformer: torch.nn.Module,
        video_state: LatentState | None,
        audio_state: LatentState | None,
        sigmas: torch.Tensor,
        step_index: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        t0 = time.perf_counter()
        sigma = sigmas[step_index]
        vid_mod = modality_from_latent_state(video_state, self.v_context, sigma, enabled=True)
        aud_mod = modality_from_latent_state(audio_state, self.a_context, sigma, enabled=False)
        sigma_b = sigma.unsqueeze(0).expand(self.batch_size)
        vid_mod = dataclasses.replace(vid_mod, sigma=sigma_b)
        aud_mod = dataclasses.replace(aud_mod, sigma=sigma_b)
        torch.compiler.cudagraph_mark_step_begin()
        result = transformer(video=vid_mod, audio=aud_mod, perturbations=None)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        if step_index == 0:
            alloc = torch.cuda.memory_allocated() / 2**30
            rsvd = torch.cuda.memory_reserved() / 2**30
            log.info("batch step %d: %.2fs (sigma=%.4f) VRAM %.2fGB alloc %.2fGB rsvd",
                     step_index, dt, sigma.item(), alloc, rsvd)
            try:
                from torch._dynamo.utils import counters
                cache_counters = {k: v for k, v in counters["inductor"].items()
                                  if "cache" in k.lower() or "autograd" in k.lower()}
                if cache_counters:
                    log.info("Inductor cache counters: %s", cache_counters)
            except Exception:
                pass
        else:
            log.debug("batch step %d: %.2fs (sigma=%.4f)", step_index, dt, sigma.item())
        if dt > 10.0:
            log.warning("batch step %d took %.1fs — possible Dynamo recompilation", step_index, dt)
        return result


class MedusaPipeline:
    """Pipeline I2V utilisant ltx-pipelines avec LTX-2.3 22B Distilled + FP8 Cast."""

    def __init__(self, models_dir: str, device: torch.device | None = None) -> None:
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.dtype = torch.bfloat16
        self.models_dir = models_dir

        # Paths modeles LTX-2.3
        self._checkpoint_path = os.path.join(models_dir, "checkpoints", "ltx-2.3-22b-distilled.safetensors")
        self._gemma_root = os.path.join(models_dir, "text_encoders", "gemma-3-12b-it")
        self._upsampler_path = os.path.join(models_dir, "upscalers", "ltx-2.3-spatial-upscaler-x2-1.0.safetensors")
        self._temporal_upsampler_path = os.path.join(models_dir, "upscalers", "ltx-2.3-temporal-upscaler-x2-1.0.safetensors")

        # Video encoder (persistent en VRAM)
        self._video_encoder: torch.nn.Module | None = None

        # Video decoder (persistent en VRAM)
        self._video_decoder: torch.nn.Module | None = None

        # Spatial upsampler x2 (persistent en VRAM, ~1GB)
        self._spatial_upsampler: torch.nn.Module | None = None

        # Transformer distilled en VRAM (FP8 cast via QuantizationPolicy)
        self._transformer: torch.nn.Module | None = None

        # --- LoRA config ---
        self._lora_dir = os.path.join(models_dir, "loras")

        # --- Depth IC-LoRA config ---
        self._depth_lora_enabled = os.environ.get("DEPTH_LORA", "1") == "1"
        self._depth_lora_strength = float(os.environ.get("DEPTH_LORA_STRENGTH", "1.0"))
        self._depth_lora_name = "depth"
        self._depth_lora_file = "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"
        self._depth_model: torch.nn.Module | None = None
        self._depth_downscale_factor: int = 1  # lu depuis metadata LoRA au chargement
        self._camera_speed_ms_default = float(os.environ.get("CAMERA_SPEED_MS", "0.5"))

        # Embeddings cache (prompt/key -> (video_ctx, audio_ctx))
        self._embeddings_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._embeddings_cache_dir: str | None = None

        # Charger les compile artifacts (Mega Cache) AVANT tout torch.compile()
        self._load_compile_artifacts()

        # Sigmas distilled (8 steps stage 1, 3 steps stage 2)
        self._sigmas = torch.tensor(
            DISTILLED_SIGMA_VALUES, dtype=torch.float32, device=self.device
        )
        self._stage2_sigmas = torch.tensor(
            STAGE_2_DISTILLED_SIGMA_VALUES, dtype=torch.float32, device=self.device
        )

    def _encode_prompts_with_gemma(self, prompts: list[str]) -> list:
        """Encode prompts via PromptEncoder block (load Gemma → encode → free)."""
        from ltx_pipelines.utils.blocks import PromptEncoder

        encoder = PromptEncoder(
            checkpoint_path=self._checkpoint_path,
            gemma_root=self._gemma_root,
            dtype=self.dtype,
            device=self.device,
        )
        return encoder(prompts)

    def warmup_embeddings(self, cache_dir: str) -> None:
        """Charge les embeddings depuis cache disque (genere par warmup_embeddings.py).

        Si le cache n'existe pas, genere les embeddings avec Gemma sur GPU.
        Charge aussi les prompts custom caches (prompt_*.pt).
        """
        self._embeddings_cache_dir = cache_dir
        cache_path = os.path.join(cache_dir, "embeddings_cache.pt")

        # Si cache existe deja, charger directement
        if os.path.isfile(cache_path):
            log.info("Chargement embeddings depuis cache: %s", cache_path)
            self._load_embeddings_cache(cache_path)

            # Charger aussi les prompts custom caches
            self._load_custom_embeddings(cache_dir)
            return

        log.info("Pas de cache embeddings — generation avec text encoder sur GPU...")

        # Encoder uniquement le prompt negative
        all_prompts = [DEFAULT_NEGATIVE_PROMPT]
        all_keys = ["_negative"]

        log.info("Encoding %d prompts...", len(all_prompts))
        results = self._encode_prompts_with_gemma(all_prompts)

        # Construire le cache
        cache_data: dict[str, dict[str, torch.Tensor]] = {}
        for key, output in zip(all_keys, results):
            v_ctx, a_ctx = output.video_encoding, output.audio_encoding
            cache_data[key] = {"video": v_ctx.cpu(), "audio": a_ctx.cpu()}
            self._embeddings_cache[key] = (v_ctx.to(self.device), a_ctx.to(self.device))

        # Sauvegarder sur disque
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(cache_data, cache_path)
        log.info("Embeddings sauvegardes: %s (%d prompts)", cache_path, len(cache_data))

    def _build_model(
        self,
        configurator: type,
        sd_ops: SDOps | None = None,
        path: str | None = None,
    ) -> torch.nn.Module:
        """Build un modele avec SingleGPUModelBuilder (persistent en VRAM)."""
        builder = SingleGPUModelBuilder(
            model_path=path or self._checkpoint_path,
            model_class_configurator=configurator,
            model_sd_ops=sd_ops,
        )
        return builder.build(device=self.device, dtype=self.dtype).eval()

    def load_video_encoder(self) -> None:
        """Charge le video encoder en VRAM (~1GB). Appeler apres warmup_embeddings."""
        log.info("Chargement video encoder (persistent)...")
        self._video_encoder = self._build_model(VideoEncoderConfigurator, VAE_ENCODER_COMFY_KEYS_FILTER)
        self._log_vram("apres video encoder")

    def load_video_decoder(self) -> None:
        """Charge le video decoder en VRAM (~2GB). Persistent entre jobs."""
        log.info("Chargement video decoder (persistent)...")
        self._video_decoder = self._build_model(VideoDecoderConfigurator, VAE_DECODER_COMFY_KEYS_FILTER)
        if os.environ.get("VAE_COMPILE", "1") == "1":
            vae_dynamic = os.environ.get("VAE_DYNAMIC_COMPILE", os.environ.get("DYNAMIC_COMPILE", "0")) == "1"
            vae_fullgraph = os.environ.get("VAE_FULLGRAPH", os.environ.get("FULLGRAPH", "1")) == "1"
            vae_mode = os.environ.get("VAE_COMPILE_MODE", os.environ.get("COMPILE_MODE", "default"))
            log.info("torch.compile video decoder (mode=%s, fullgraph=%s, dynamic=%s)...", vae_mode, vae_fullgraph, vae_dynamic)
            self._video_decoder = torch.compile(
                self._video_decoder, mode=vae_mode, fullgraph=vae_fullgraph, dynamic=vae_dynamic,
            )
        self._log_vram("apres video decoder")

    def load_spatial_upsampler(self) -> None:
        """Charge le spatial upsampler x2 en VRAM (~1GB). Persistent entre jobs."""
        log.info("Chargement spatial upsampler x2 (persistent)...")
        self._spatial_upsampler = self._build_model(
            LatentUpsamplerConfigurator, path=self._upsampler_path,
        )
        self._log_vram("apres spatial upsampler")

    def _load_embeddings_cache(self, cache_path: str) -> None:
        """Charge les embeddings depuis un fichier .pt.

        Si le cache contient des cles legacy (presets camera), le supprime et regenere.
        """
        cache_data = torch.load(cache_path, map_location="cpu", weights_only=True)

        # Check version schéma : si legacy presets detectes, invalider le cache
        legacy_keys = {"dolly-in", "dolly-out", "dolly-left", "dolly-right", "jib-up", "jib-down", "static"}
        has_legacy = any(k in cache_data for k in legacy_keys)

        if has_legacy:
            log.warning("Cache embeddings contient presets camera legacy — suppression et regeneration")
            try:
                os.unlink(cache_path)
            except OSError:
                pass
            # Regenerer a chaud
            self.warmup_embeddings(os.path.dirname(cache_path))
            return

        for key, tensors in cache_data.items():
            self._embeddings_cache[key] = (
                tensors["video"].to(self.device),
                tensors["audio"].to(self.device),
            )
        log.info("Embeddings charges: %d prompts", len(self._embeddings_cache))

    def _load_custom_embeddings(self, cache_dir: str) -> None:
        """Charge les embeddings custom caches (prompt_*.pt) depuis le disque."""
        count = 0
        for filename in os.listdir(cache_dir):
            if not filename.startswith("prompt_") or not filename.endswith(".pt"):
                continue
            prompt_hash = filename[len("prompt_"):-len(".pt")]
            if prompt_hash in self._embeddings_cache:
                continue
            filepath = os.path.join(cache_dir, filename)
            try:
                data = torch.load(filepath, map_location="cpu", weights_only=True)
                self._embeddings_cache[prompt_hash] = (
                    data["video"].to(self.device), data["audio"].to(self.device),
                )
                count += 1
            except Exception as e:
                log.warning("Skip custom embedding %s: %s", filename, e)
        if count:
            log.info("Custom embeddings charges: %d prompt(s)", count)

    @staticmethod
    def _log_vram(label: str) -> None:
        """Log l'utilisation VRAM courante (alloue + reserve)."""
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 2**30
            rsvd = torch.cuda.memory_reserved() / 2**30
            log.info("VRAM [%s]: %.2fGB alloc, %.2fGB reserved", label, alloc, rsvd)

    @staticmethod
    def _make_video_tools(shape: VideoPixelShape) -> VideoLatentTools:
        """Cree les outils latent video (patchifier + shape) pour un output_shape donne."""
        return VideoLatentTools(
            patchifier=VideoLatentPatchifier(patch_size=1),
            target_shape=VideoLatentShape.from_pixel_shape(shape),
            fps=shape.fps,
        )

    @staticmethod
    def _make_audio_tools(shape: VideoPixelShape) -> AudioLatentTools:
        """Cree les outils latent audio pour un output_shape donne."""
        return AudioLatentTools(
            patchifier=AudioPatchifier(patch_size=1),
            target_shape=AudioLatentShape.from_video_pixel_shape(shape),
        )

    @staticmethod
    def _compile_artifacts_path() -> str:
        """Chemin du blob Mega Cache sur le volume."""
        build_hash = os.environ.get("BUILD_HASH", "unknown")
        return os.path.join(
            os.environ.get("WORKSPACE", "/runpod-volume"),
            "cache", "compile_artifacts", f"{build_hash}.bin",
        )

    def _load_compile_artifacts(self) -> None:
        """Charge les compile artifacts (Mega Cache) depuis le volume si disponibles.

        DOIT etre appele AVANT tout torch.compile() dans le process.
        """
        artifacts_path = self._compile_artifacts_path()
        if os.path.isfile(artifacts_path):
            try:
                with open(artifacts_path, "rb") as f:
                    info = torch.compiler.load_cache_artifacts(f.read())
                sz = os.path.getsize(artifacts_path) / (1024 * 1024)
                log.info("Compile artifacts loaded: %s (%.0f MB, info=%s)", artifacts_path, sz, info)
            except Exception as e:
                log.warning("load_cache_artifacts failed (will recompile): %s", e)
        else:
            log.info("No compile artifacts found at %s (cold start)", artifacts_path)

    def save_compile_artifacts(self) -> None:
        """Sauvegarde les artifacts torch.compile sur le volume (Mega Cache)."""
        artifacts_path = self._compile_artifacts_path()
        if os.path.isfile(artifacts_path):
            return  # deja sauvegarde
        try:
            result = torch.compiler.save_cache_artifacts()
            if result is None:
                log.warning("save_cache_artifacts returned None (no compilation done?)")
                return
            artifact_bytes, info = result
            os.makedirs(os.path.dirname(artifacts_path), exist_ok=True)
            tmp = artifacts_path + ".tmp"
            with open(tmp, "wb") as f:
                f.write(artifact_bytes)
            os.replace(tmp, artifacts_path)
            sz = len(artifact_bytes) / (1024 * 1024)
            log.info("Compile artifacts saved: %s (%.0f MB, info=%s)", artifacts_path, sz, info)
        except Exception as e:
            log.warning("save_compile_artifacts failed: %s", e)

    def _create_dummy_depth_conditioning(
        self, stage1_h: int, stage1_w: int, num_frames: int,
    ) -> ConditioningItem:
        """Cree un dummy depth conditioning pour le warmup compile.

        Meme shape que create_depth_conditioning() mais sans DA3METRIC —
        juste un tensor zero de la bonne dimension, VAE-encode, pour que
        Dynamo compile le graph avec reference latents.
        """
        scale = self._depth_downscale_factor
        ref_h = (stage1_h // scale // 32) * 32
        ref_w = (stage1_w // scale // 32) * 32
        # Dummy depth video : [1, 3, F, ref_h, ref_w] meme dtype que le VAE encoder
        dummy_video = torch.zeros(
            1, 3, num_frames, ref_h, ref_w,
            dtype=self.dtype, device=self.device,
        )
        encoded = self._video_encoder(dummy_video)
        return VideoConditionByReferenceLatent(
            latent=encoded,
            downscale_factor=scale,
            strength=self._depth_lora_strength,
        )

    @torch.inference_mode()
    def warmup_compile(self, num_frames: int = 25, frame_rate: float = 24) -> None:
        """Pre-compile le transformer pour toutes les shapes (720p + 1080p 2-stage, landscape + portrait).

        Lance un forward dummy par resolution pour declencher la compilation
        Dynamo de toutes les variantes attention/shapes. Elimine les recompilations
        couteuses (~30s) du premier job reel.

        Pour les shapes s1, un second forward avec depth conditioning est effectue
        pour couvrir le chemin IC-LoRA (reference latents → sequence d'attention elargie).
        """
        from PIL import Image

        log.info("Warmup compile: pre-compilation transformer...")
        t0_total = time.perf_counter()

        transformer = self.get_transformer()
        depth_enabled = self._depth_lora_enabled and self._depth_model is not None

        # Image dummy (blank, sera remplacee par le conditioning latent)
        tmp_path = "/tmp/warmup_compile_dummy.png"
        Image.new("RGB", (64, 64), (128, 128, 128)).save(tmp_path)

        # Prompt stub fixe pour warmup (pas de preset)
        warmup_prompt = "A test scene for compilation warmup"
        v_ctx, a_ctx = self.encode_prompt(warmup_prompt)

        gen = torch.Generator(device=self.device).manual_seed(0)
        noiser = GaussianNoiser(generator=gen)

        # Configs couvrant toutes les shapes (landscape + portrait).
        # audio.enabled=False partout (on ne genere pas d'audio) → un seul chemin Dynamo par shape.
        # Portrait = landscape transpose (memes megapixels, sequences de tokens differentes)
        # Note : 540p (544×960) = meme shape que 1080p-s1, pas de warmup supplementaire
        # is_s1 = True pour les shapes Stage 1 qui recoivent du depth conditioning
        configs: list[tuple[str, int, int, torch.Tensor, bool]] = [
            # Tier 2 — Standard 720p 2-stage
            ("720p-s1",              352,  640,  self._sigmas,         True),
            ("720p-s2",              704,  1280, self._stage2_sigmas,  False),
            # Tier 3 — Production 1080p 2-stage
            ("1080p-s1",             544,  960,  self._sigmas,         True),
            ("1080p-s2",             1088, 1920, self._stage2_sigmas,  False),
            # Portrait 9:16
            ("720p-portrait-s1",     640,  352,  self._sigmas,         True),
            ("720p-portrait-s2",     1280, 704,  self._stage2_sigmas,  False),
            ("1080p-portrait-s1",    960,  544,  self._sigmas,         True),
            ("1080p-portrait-s2",    1920, 1088, self._stage2_sigmas,  False),
        ]

        for label, h, w, sigmas, is_s1 in configs:
            t0 = time.perf_counter()

            images = [ImageConditioningInput(path=tmp_path, frame_idx=0, strength=1.0)]
            conds = image_conditionings_by_replacing_latent(
                images=images, height=h, width=w,
                video_encoder=self._video_encoder,
                dtype=self.dtype, device=self.device,
            )

            shape = VideoPixelShape(
                batch=1, frames=num_frames, width=w, height=h, fps=frame_rate,
            )

            v_tools = self._make_video_tools(shape)
            a_tools = self._make_audio_tools(shape)
            vs = create_noised_state(v_tools, conds, noiser, self.dtype, self.device)
            as_ = create_noised_state(a_tools, [], noiser, self.dtype, self.device)

            sigma = sigmas[0]
            vid_mod = modality_from_latent_state(vs, v_ctx, sigma, enabled=True)
            aud_mod = modality_from_latent_state(as_, a_ctx, sigma, enabled=False)

            sigma_b = sigma.unsqueeze(0)
            vid_mod = dataclasses.replace(vid_mod, sigma=sigma_b)
            aud_mod = dataclasses.replace(aud_mod, sigma=sigma_b)

            torch.compiler.cudagraph_mark_step_begin()
            transformer(video=vid_mod, audio=aud_mod, perturbations=None)
            torch.cuda.synchronize()

            dt = time.perf_counter() - t0
            log.info("Warmup %s (%dx%d): %.1fs", label, w, h, dt)

            del vs, as_, vid_mod, aud_mod, conds
            torch.cuda.empty_cache()

            # Second pass avec depth conditioning pour les shapes s1
            if is_s1 and depth_enabled:
                t0_d = time.perf_counter()

                depth_cond = self._create_dummy_depth_conditioning(h, w, num_frames)

                images_d = [ImageConditioningInput(path=tmp_path, frame_idx=0, strength=1.0)]
                conds_d = image_conditionings_by_replacing_latent(
                    images=images_d, height=h, width=w,
                    video_encoder=self._video_encoder,
                    dtype=self.dtype, device=self.device,
                )
                conds_d = conds_d + [depth_cond]

                vs_d = create_noised_state(v_tools, conds_d, noiser, self.dtype, self.device)
                as_d = create_noised_state(a_tools, [], noiser, self.dtype, self.device)

                vid_mod_d = modality_from_latent_state(vs_d, v_ctx, sigma, enabled=True)
                aud_mod_d = modality_from_latent_state(as_d, a_ctx, sigma, enabled=False)
                vid_mod_d = dataclasses.replace(vid_mod_d, sigma=sigma_b)
                aud_mod_d = dataclasses.replace(aud_mod_d, sigma=sigma_b)

                torch.compiler.cudagraph_mark_step_begin()
                transformer(video=vid_mod_d, audio=aud_mod_d, perturbations=None)
                torch.cuda.synchronize()

                dt_d = time.perf_counter() - t0_d
                log.info("Warmup %s+depth (%dx%d): %.1fs", label, w, h, dt_d)

                del vs_d, as_d, vid_mod_d, aud_mod_d, conds_d, depth_cond
                torch.cuda.empty_cache()

        try:
            os.unlink(tmp_path)
        except OSError:
            pass

        dt_total = time.perf_counter() - t0_total
        log.info("Warmup compile termine: %.1fs total", dt_total)

    @staticmethod
    def _log_inductor_cache_diagnostic() -> None:
        """Diagnostic compact du cache Inductor."""
        cache_dir = os.environ.get("TORCHINDUCTOR_CACHE_DIR", "")
        build_hash = os.environ.get("BUILD_HASH", "unknown")
        is_debug = os.environ.get("LOG_LEVEL", "info").lower() == "debug"

        # Counts aotautograd + fxgraph
        aot_count = 0
        aot_dir = os.path.join(cache_dir, "aotautograd") if cache_dir else ""
        if aot_dir and os.path.isdir(aot_dir):
            aot_count = len(os.listdir(aot_dir))

        fx_dirs, fx_files = 0, 0
        fxgraph_dir = os.path.join(cache_dir, "fxgraph") if cache_dir else ""
        if fxgraph_dir and os.path.isdir(fxgraph_dir):
            for d in os.listdir(fxgraph_dir):
                dp = os.path.join(fxgraph_dir, d)
                if os.path.isdir(dp):
                    fx_dirs += 1
                    fx_files += sum(1 for f in os.listdir(dp) if os.path.isfile(os.path.join(dp, f)))

        # Compile artifacts (Mega Cache)
        artifacts_path = os.path.join(
            os.environ.get("WORKSPACE", "/runpod-volume"),
            "cache", "compile_artifacts", f"{build_hash}.bin",
        )
        if os.path.isfile(artifacts_path):
            art_sz = os.path.getsize(artifacts_path) / (1024 * 1024)
            art_info = f"{artifacts_path} (exists, {art_sz:.0f} MB)"
        else:
            art_info = f"{artifacts_path} (missing)"

        # Inductor config hash
        cfg_hash = "?"
        try:
            from torch._inductor import config as ind_config
            cfg = ind_config.save_config_portable()
            cfg_str = json.dumps(cfg, sort_keys=True, default=str)
            cfg_hash = hashlib.sha256(cfg_str.encode()).hexdigest()[:16]
        except Exception:
            pass

        # CUDA info
        cuda_info = ""
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            cuda_info = f"{props.name}, sm_{props.major}.{props.minor}, torch={torch.__version__}"

        ic = torch._inductor.config
        log.info("=== INDUCTOR CACHE ===")
        log.info("  cache_dir=%s", cache_dir or "(default)")
        log.info(
            "  fx_graph_cache=%s, autograd_cache=%s, compile_threads=%s, config_hash=%s",
            getattr(ic, "fx_graph_cache", "?"),
            os.environ.get("TORCHINDUCTOR_AUTOGRAD_CACHE", "0"),
            getattr(ic, "compile_threads", "?"),
            cfg_hash,
        )
        log.info("  aotautograd: %d entries | fxgraph: %d dirs (%d files)", aot_count, fx_dirs, fx_files)
        log.info("  artifacts: %s", art_info)
        if cuda_info:
            log.info("  CUDA: %s", cuda_info)
        log.info("=== FIN CACHE ===")

        # Mode debug : inductor config non-default + fxgraph listing detaille
        if is_debug:
            try:
                cfg = ind_config.save_config_portable()
                non_default = {k: v for k, v in sorted(cfg.items())
                               if v is not None and v != "" and v is not False and v != 0}
                for k, v in non_default.items():
                    log.debug("  inductor_config %s=%s", k, v)
            except Exception:
                pass
            if fxgraph_dir and os.path.isdir(fxgraph_dir):
                for d in os.listdir(fxgraph_dir):
                    dp = os.path.join(fxgraph_dir, d)
                    if os.path.isdir(dp):
                        files = [f for f in os.listdir(dp) if os.path.isfile(os.path.join(dp, f))]
                        total_sz = sum(os.path.getsize(os.path.join(dp, f)) for f in files)
                        log.debug("    fxgraph/%s: %d files, %.1f KB", d, len(files), total_sz / 1024)

    # --- Depth estimation (DA3) + IC-LoRA conditioning ---

    def load_depth_model(self) -> None:
        """Charge DA3METRIC-LARGE sur GPU (~1.64GB VRAM)."""
        if not self._depth_lora_enabled:
            return
        da3_path = os.path.join(self.models_dir, "da3-metric")
        if not os.path.isdir(da3_path):
            log.warning("DA3METRIC model introuvable: %s — depth disabled", da3_path)
            self._depth_lora_enabled = False
            return

        from depth_anything_3.api import DepthAnything3

        t0 = time.perf_counter()
        self._depth_model = DepthAnything3.from_pretrained(da3_path)
        self._depth_model = self._depth_model.to(device=self.device)
        dt = time.perf_counter() - t0
        log.info("DA3METRIC-LARGE charge en %.1fs (VRAM ~1.64GB)", dt)

        # Lire reference_downscale_factor depuis metadata du LoRA
        lora_path = os.path.join(self._lora_dir, self._depth_lora_file)
        if os.path.isfile(lora_path):
            self._depth_downscale_factor = self._read_lora_downscale_factor(lora_path)
            log.info("IC-LoRA downscale_factor=%d", self._depth_downscale_factor)

    @staticmethod
    def _read_lora_downscale_factor(lora_path: str) -> int:
        """Lit reference_downscale_factor depuis les metadata du safetensors."""
        try:
            with safe_open(lora_path, framework="pt") as f:
                metadata = f.metadata() or {}
                return int(metadata.get("reference_downscale_factor", 1))
        except Exception as e:
            log.warning("Echec lecture metadata LoRA '%s': %s", lora_path, e)
            return 1

    def estimate_depth_metric(self, image_path: str) -> tuple[torch.Tensor, torch.Tensor]:
        """DA3METRIC inference → (depth_meters [H,W], sky_mask [H,W]) sur GPU.

        Convertit la prediction brute en metres via les intrinsics estimes.
        """
        prediction = self._depth_model.inference([image_path])

        # Depth brute [H, W] float32
        depth_raw = prediction.depth[0]
        if not isinstance(depth_raw, torch.Tensor):
            depth_raw = torch.from_numpy(depth_raw)
        depth_raw = depth_raw.to(device=self.device, dtype=torch.float32)

        # Intrinsics [1, 3, 3] → focale moyenne (fallback: max(H,W))
        intrinsics = prediction.intrinsics
        if intrinsics is not None:
            if not isinstance(intrinsics, torch.Tensor):
                intrinsics = torch.from_numpy(intrinsics)
            intrinsics = intrinsics.to(device=self.device, dtype=torch.float32)
            focal = (intrinsics[0, 0, 0] + intrinsics[0, 1, 1]) / 2.0
        else:
            focal = float(max(depth_raw.shape[-2], depth_raw.shape[-1]))
            log.warning("DA3METRIC: intrinsics=None, fallback focal=%.0f px", focal)

        # Conversion en metres : depth_meters = focal * depth_raw / 300.0
        depth_meters = focal * depth_raw / 300.0

        # Sky mask (fallback: pas de masque)
        sky_mask = getattr(prediction, "sky_mask", None)
        if sky_mask is None:
            sky_mask = torch.zeros_like(depth_raw, dtype=torch.bool)
        else:
            if not isinstance(sky_mask, torch.Tensor):
                sky_mask = torch.from_numpy(sky_mask)
            sky_mask = sky_mask.to(device=self.device, dtype=torch.bool)

        log.debug(
            "DA3METRIC: depth %.1f-%.1f m, focal %.0f px, sky %.1f%%",
            depth_meters.min(), depth_meters.max(), focal,
            sky_mask.float().mean() * 100,
        )
        return depth_meters, sky_mask

    @staticmethod
    def _warp_depth_generic(
        depth: torch.Tensor,
        translation: torch.Tensor,
        rotation_quat: torch.Tensor,
        focal_px: float,
        sky_mask: torch.Tensor,
    ) -> torch.Tensor:
        """6-DOF depth warp via unproject→transform→reproject + forward splatting.

        Convention: translation et rotation_quat sont fournis en repere monde
        Y-up, right-handed, Z-forward (CLAUDE.md). L'unprojection pixel utilise
        Y-down (convention image standard), donc on convertit en interne via
        F = diag(1, -1, 1):
          - translation_pix = F @ translation
          - R_pix = F @ R_user @ F

        Pour une translation Z pure (dolly), F @ R @ F = R et le Y-flip de
        translation n'a aucun effet → compat totale avec l'ancien comportement.

        Args:
            depth: [H, W] profondeur metrique en metres (GPU)
            translation: [3] deplacement camera en metres [x, y, z] (Y-up)
            rotation_quat: [4] quaternion unit [w, x, y, z] (Y-up)
            focal_px: focal length en pixels, doit correspondre a depth.shape[1]
            sky_mask: [H, W] bool (True = ciel, pas warpe)

        Returns:
            [H, W] depth warpee en metres
        """
        H, W = depth.shape
        device = depth.device

        # Convention flip : user (Y-up) → pixel (Y-down)
        # F = diag(1, -1, 1), involution : F @ F = I
        flip = torch.tensor([1.0, -1.0, 1.0], device=device, dtype=torch.float32)
        translation = translation * flip  # [tx, -ty, tz]

        # Quaternion to rotation matrix puis F @ R @ F → convention pixel
        R_user = quat_to_matrix(rotation_quat.unsqueeze(0))[0]  # [3, 3]
        R = R_user * flip.unsqueeze(0) * flip.unsqueeze(1)  # F @ R_user @ F

        cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
        f = focal_px

        # Pixel grids
        v_grid, u_grid = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing="ij",
        )

        # Unproject 2D → 3D
        X = (u_grid - cx) * depth / f
        Y = (v_grid - cy) * depth / f
        Z = depth
        P = torch.stack([X, Y, Z], dim=-1)  # [H, W, 3]

        # Transform inverse : P' = R^T @ (P - translation) pour chaque point colonne.
        # Avec P_trans[H,W,3] (dernier axe = vecteur), appliquer R^T au vecteur colonne
        # equivaut a post-multiplier par R : (R^T @ p_col)[d] = sum_c R[c,d] * p[c] = (p_row @ R)[d]
        P_trans = P - translation.unsqueeze(0).unsqueeze(0)  # [H, W, 3]
        P_prime = P_trans @ R  # [H, W, 3] @ [3, 3] -> [H, W, 3]

        # Reproject 3D → 2D
        u_new = cx + f * P_prime[..., 0] / torch.clamp(P_prime[..., 2], min=1e-6)
        v_new = cy + f * P_prime[..., 1] / torch.clamp(P_prime[..., 2], min=1e-6)
        d_new = P_prime[..., 2]

        # Foreground: pas ciel ET pas derriere camera
        fg = ~sky_mask & (d_new > 0.01)

        # Valid: foreground ET dans les limites
        valid = fg & (u_new >= 0) & (u_new < W) & (v_new >= 0) & (v_new < H)

        # Z-buffer splatting (scatter_reduce amin = closest)
        u_idx = torch.clamp(torch.round(u_new[valid]).long(), 0, W - 1)
        v_idx = torch.clamp(torch.round(v_new[valid]).long(), 0, H - 1)
        flat_idx = v_idx * W + u_idx
        flat_d = d_new[valid]

        output = torch.full((H * W,), float("inf"), device=device, dtype=torch.float32)
        output.scatter_reduce_(0, flat_idx, flat_d, reduce="amin")
        output = output.view(H, W)

        # Fallback global (utilise si image 100% ciel ou trous non comblables)
        fallback_val = depth[fg].max() if fg.any() else depth.max()

        # Hole filling via morphological dilation (weighted avg of neighbors)
        holes = output.isinf()
        if holes.any():
            clean = output.clone()
            clean[holes] = 0.0
            mask = (~holes).float()
            clean_4d = clean.unsqueeze(0).unsqueeze(0)
            mask_4d = mask.unsqueeze(0).unsqueeze(0)

            for _ in range(5):
                sum_v = F.avg_pool2d(clean_4d, 3, 1, 1) * 9
                sum_m = F.avg_pool2d(mask_4d, 3, 1, 1) * 9
                avg = sum_v / torch.clamp(sum_m, min=1)
                fill_mask = (mask_4d == 0) & (sum_m > 0)
                clean_4d = torch.where(fill_mask, avg, clean_4d)
                mask_4d = torch.where(fill_mask, torch.ones_like(mask_4d), mask_4d)
                if mask_4d.all():
                    break

            output = clean_4d.squeeze(0).squeeze(0)

            # Fallback: pixels still empty → max foreground
            still_empty = output == 0.0
            if still_empty.any():
                output[still_empty] = fallback_val

        # Sky: non-warped, set to max depth
        if sky_mask.any():
            if (~sky_mask).any():
                output[sky_mask] = output[~sky_mask].max()
            else:
                output[sky_mask] = fallback_val

        return output

    @staticmethod
    def _warp_depth_dolly(
        depth: torch.Tensor,
        delta: float,
        sky_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward splatting parallax warp pour dolly-in.

        Chaque pixel se deplace du centre proportionnellement a sa profondeur :
        scale = d / (d - delta). Les proches bougent plus que les lointains.
        La focale s'annule dans la derivation (pas besoin d'intrinsics).

        Args:
            depth: [H, W] profondeur metrique en metres (GPU)
            delta: deplacement camera en metres (>0 = dolly-in)
            sky_mask: [H, W] bool (True = ciel, pas warpe)

        Returns:
            [H, W] depth warpee en metres
        """
        if delta < 1e-6:
            return depth.clone()

        H, W = depth.shape
        device = depth.device
        cx, cy = (W - 1) / 2.0, (H - 1) / 2.0

        # Grilles de coordonnees pixels
        v_grid, u_grid = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing="ij",
        )

        # Foreground : pas ciel ET pas derriere la camera apres deplacement
        fg = ~sky_mask & (depth > delta + 0.01)

        # Scale de parallaxe par pixel (proches > 1, lointains ≈ 1)
        scale = torch.ones(H, W, device=device, dtype=torch.float32)
        scale[fg] = depth[fg] / (depth[fg] - delta)

        # Nouvelles positions pixels (forward warp)
        u_new = cx + (u_grid - cx) * scale
        v_new = cy + (v_grid - cy) * scale

        # Depth apres avancee camera
        d_new = torch.clamp(depth - delta, min=0.01)

        # Filtrer : foreground + dans les limites de l'image
        valid = fg & (u_new >= 0) & (u_new < W) & (v_new >= 0) & (v_new < H)

        # Z-buffer splatting (scatter_reduce amin = garder le plus proche)
        u_idx = torch.clamp(torch.round(u_new[valid]).long(), 0, W - 1)
        v_idx = torch.clamp(torch.round(v_new[valid]).long(), 0, H - 1)
        flat_idx = v_idx * W + u_idx
        flat_d = d_new[valid]

        output = torch.full((H * W,), float("inf"), device=device, dtype=torch.float32)
        output.scatter_reduce_(0, flat_idx, flat_d, reduce="amin")
        output = output.view(H, W)

        # Remplir les trous par dilatation morphologique (avg pondere des voisins)
        # Les trous du splatting forment un pattern regulier que le modele hallucine
        holes = output.isinf()
        if holes.any():
            clean = output.clone()
            clean[holes] = 0.0
            mask = (~holes).float()
            clean_4d = clean.unsqueeze(0).unsqueeze(0)
            mask_4d = mask.unsqueeze(0).unsqueeze(0)

            for _ in range(5):
                sum_v = F.avg_pool2d(clean_4d, 3, 1, 1) * 9
                sum_m = F.avg_pool2d(mask_4d, 3, 1, 1) * 9
                avg = sum_v / torch.clamp(sum_m, min=1)
                fill_mask = (mask_4d == 0) & (sum_m > 0)
                clean_4d = torch.where(fill_mask, avg, clean_4d)
                mask_4d = torch.where(fill_mask, torch.ones_like(mask_4d), mask_4d)
                if mask_4d.all():
                    break

            output = clean_4d.squeeze(0).squeeze(0)

            # Fallback : pixels encore vides (gaps > 5px) → max foreground
            still_empty = output == 0.0
            if still_empty.any():
                fill_val = depth[fg].max() if fg.any() else depth.max()
                output[still_empty] = fill_val

        # Sky → max depth (pixels ciel non warpes)
        if sky_mask.any():
            output[sky_mask] = output[~sky_mask].max() if (~sky_mask).any() else fill_val

        return output

    def render_depth_sequence(
        self,
        depth_meters: torch.Tensor,
        sky_mask: torch.Tensor,
        num_frames: int,
        camera_path: list[dict] | None,
        interpolation: str,
        camera_speed_ms: float,
        focal_px: float,
        frame_rate: float,
        target_h: int,
        target_w: int,
    ) -> torch.Tensor:
        """Genere une sequence depth par 6-DOF warp (translation + rotation).

        Args:
            depth_meters: [H, W] profondeur metrique en metres
            sky_mask: [H, W] bool (True = pixels ciel)
            num_frames: nombre de frames a generer
            camera_path: list[dict] keyframes | None (fallback dolly-in si None)
            interpolation: "linear" | "cubic" mode interpolation
            camera_speed_ms: vitesse camera en m/s (fallback si camera_path None)
            focal_px: focal length en pixels
            frame_rate: fps
            target_h, target_w: resolution de sortie

        Returns:
            [1, 3, F, target_h, target_w] tensor normalise [-1, 1]
        """
        # Fallback backward compat : si pas de camera_path, generer dolly-in
        if camera_path is None or len(camera_path) == 0:
            duration_s = (num_frames - 1) / frame_rate
            max_z = camera_speed_ms * duration_s
            camera_path = [
                {"t": 0.0, "translation": [0.0, 0.0, 0.0], "rotation_quat": [1.0, 0.0, 0.0, 0.0]},
                {"t": 1.0, "translation": [0.0, 0.0, max_z], "rotation_quat": [1.0, 0.0, 0.0, 0.0]},
            ]
            interpolation = "linear"

        has_sky = sky_mask.any()

        frames = []
        for i in range(num_frames):
            t = i / max(num_frames - 1, 1)

            # Interpolate camera pose (CPU tensors → move to depth device)
            trans, quat = interpolate_camera_path(camera_path, t, interpolation)
            trans = trans.to(depth_meters.device)
            quat = quat.to(depth_meters.device)

            # 6-DOF depth warp
            depth_frame = self._warp_depth_generic(depth_meters, trans, quat, focal_px, sky_mask)

            # Per-frame normalisation [0,1] sur foreground uniquement
            fg_vals = depth_frame[~sky_mask] if has_sky else depth_frame
            if fg_vals.numel() > 0:
                f_min, f_max = fg_vals.min(), fg_vals.max()
            else:
                f_min, f_max = depth_frame.min(), depth_frame.max()
            f_range = f_max - f_min

            if f_range > 1e-6:
                depth_norm = (depth_frame - f_min) / f_range
            else:
                depth_norm = torch.zeros_like(depth_frame)

            # Ciel → 1.0 (profondeur maximale)
            if has_sky:
                depth_norm = torch.where(sky_mask, torch.ones_like(depth_norm), depth_norm)

            depth_norm = torch.clamp(depth_norm, 0.0, 1.0)
            frames.append(depth_norm)

        # Stack [F, H, W]
        depth_video = torch.stack(frames)

        # Resize vers resolution cible
        depth_video = depth_video.unsqueeze(1)  # [F, 1, H, W]
        depth_video = F.interpolate(
            depth_video, size=(target_h, target_w),
            mode="bilinear", align_corners=False,
        )

        # [0,1] → [-1,1] et expand 3 channels RGB
        depth_video = depth_video * 2.0 - 1.0
        depth_video = depth_video.expand(-1, 3, -1, -1)  # [F, 3, H, W]

        # Reshape [1, 3, F, H, W] (batch=1, format video)
        depth_video = depth_video.permute(1, 0, 2, 3).unsqueeze(0)
        return depth_video.to(dtype=self.dtype)

    def create_depth_conditioning(
        self,
        image_path: str,
        stage1_h: int,
        stage1_w: int,
        num_frames: int,
        frame_rate: float,
        camera_speed_ms: float,
        camera_path: list[dict] | None = None,
        interpolation: str = "linear",
        fov_degrees: float = 60.0,
    ) -> ConditioningItem:
        """Cree le conditioning IC-LoRA depth pour Stage 1.

        Pipeline : DA3METRIC → depth metres + sky mask → 6-DOF warp N frames →
        VAE encode → VideoConditionByReferenceLatent
        """
        t0 = time.perf_counter()

        # 1. Depth estimation metrique
        depth_meters, sky_mask = self.estimate_depth_metric(image_path)

        # 2. Compute focal length from FOV
        # IMPORTANT: focal_px doit correspondre a la resolution sur laquelle
        # l'unprojection/reprojection est faite, soit depth_meters.shape.
        # Utiliser ref_w ici introduirait un facteur focal_wrong/focal_correct
        # sur les translations X/Y et rotations (invisible pour dolly-in Z pur).
        scale = self._depth_downscale_factor
        ref_h = (stage1_h // scale // 32) * 32  # multiples de 32 (target resize)
        ref_w = (stage1_w // scale // 32) * 32
        depth_w = depth_meters.shape[1]
        focal_px = depth_w / (2 * math.tan(math.radians(fov_degrees) / 2))

        # 3. Render depth sequence (6-DOF warp avec camera_path optionnel)
        depth_video = self.render_depth_sequence(
            depth_meters, sky_mask, num_frames,
            camera_path=camera_path,
            interpolation=interpolation,
            camera_speed_ms=camera_speed_ms,
            focal_px=focal_px,
            frame_rate=frame_rate,
            target_h=ref_h, target_w=ref_w,
        )  # [1, 3, F, ref_h, ref_w]

        # 4. VAE encode
        encoded = self._video_encoder(depth_video)

        # 5. Wrap en conditioning IC-LoRA
        cond = VideoConditionByReferenceLatent(
            latent=encoded,
            downscale_factor=scale,
            strength=self._depth_lora_strength,
        )

        dt = time.perf_counter() - t0
        log.info(
            "Depth conditioning: DA3METRIC+render+VAE en %.2fs "
            "(%dx%d, %d frames, scale=%d, speed=%.2f m/s, fov=%.1f)",
            dt, ref_w, ref_h, num_frames, scale, camera_speed_ms, fov_degrees,
        )
        return cond

    def _cache_fingerprint(self, lora_path: str | None = None, lora_strength: float = 1.0) -> str:
        """Hash d'invalidation pour le cache transformer (avec LoRA fuse).

        Combine CACHE_VERSION, taille checkpoint, et LoRA path/strength.
        Retourne un hash court (16 chars).
        """
        h = hashlib.sha256()
        h.update(CACHE_VERSION.encode())
        h.update(str(os.path.getsize(self._checkpoint_path)).encode())
        if lora_path is not None:
            h.update(str(os.path.getsize(lora_path)).encode())
            h.update(str(lora_strength).encode())
        return h.hexdigest()[:16]

    def _save_transformer_cache(self, cache_path: str) -> None:
        """Sauvegarde le transformer fusionne (distilled) en format checkpoint.

        Extrait le state_dict du LTXModel interne, re-ajoute le prefixe
        `model.diffusion_model.` pour compatibilite avec SingleGPUModelBuilder,
        et copie les metadonnees du checkpoint original.
        Ecriture atomique via .tmp + os.replace().
        """
        raw_sd = self._transformer.velocity_model.state_dict()

        # Re-ajouter le prefixe checkpoint (inverse du renaming COMFY fait au chargement)
        sd = {f"model.diffusion_model.{k}": v.cpu() for k, v in raw_sd.items()}

        # Copier les metadonnees du checkpoint original (config architecture, etc.)
        metadata: dict[str, str] = {}
        with safetensors.safe_open(self._checkpoint_path, framework="pt") as f:
            orig_meta = f.metadata()
            if orig_meta:
                metadata.update(orig_meta)

        # Ecriture atomique
        tmp_path = cache_path + ".tmp"
        try:
            save_safetensors(sd, tmp_path, metadata=metadata)
            os.replace(tmp_path, cache_path)
            size_gb = os.path.getsize(cache_path) / 2**30
            log.info("Cache transformer sauvegarde: %s (%.1fGB)", cache_path, size_gb)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def _build_transformer_from_checkpoint(
        self,
        checkpoint_path: str,
        lora_path: str | None = None,
        lora_strength: float = 1.0,
    ) -> torch.nn.Module:
        """Build le transformer avec SingleGPUModelBuilder + FP8 quantization.

        Si lora_path est fourni, le LoRA est fuse pendant le build via le
        kernel Triton natif de ltx-core (apply_loras).
        """
        quantization = QuantizationPolicy.fp8_cast()
        base_sd_ops = LTXV_MODEL_COMFY_RENAMING_MAP
        sd_ops = SDOps(
            name=f"chain_{base_sd_ops.name}+{quantization.sd_ops.name}",
            mapping=(*base_sd_ops.mapping, *quantization.sd_ops.mapping),
        )
        builder = SingleGPUModelBuilder(
            model_path=checkpoint_path,
            model_class_configurator=LTXModelConfigurator,
            model_sd_ops=sd_ops,
            module_ops=quantization.module_ops,
        )
        if lora_path is not None:
            builder = builder.lora(lora_path, strength=lora_strength, sd_ops=LTXV_LORA_COMFY_RENAMING_MAP)
        transformer = X0Model(builder.build(device=self.device, dtype=self.dtype))
        return transformer.to(self.device)

    def _load_transformer_cache(self, cache_path: str) -> torch.nn.Module:
        """Charge le transformer pre-fusionne depuis le cache."""
        log.info("Chargement transformer depuis cache: %s", cache_path)
        return self._build_transformer_from_checkpoint(cache_path)

    def get_transformer(self) -> torch.nn.Module:
        """Retourne le transformer distilled avec FP8 Cast + LoRA fuse.

        Checkpoint distilled BF16 + QuantizationPolicy.fp8_cast() → stockage FP8, compute BF16.
        Si DEPTH_LORA=1, le IC-LoRA est fuse nativement par le builder (kernel Triton).
        Le cache contient le transformer avec LoRA deja fuse.
        """
        if self._transformer is not None:
            return self._transformer

        # Determiner LoRA path
        lora_path = None
        lora_strength = self._depth_lora_strength
        if self._depth_lora_enabled:
            lp = os.path.join(self._lora_dir, self._depth_lora_file)
            if os.path.isfile(lp):
                lora_path = lp

        cache_dir = os.path.join(os.path.dirname(self.models_dir), "cache", "transformer")
        os.makedirs(cache_dir, exist_ok=True)
        fingerprint = self._cache_fingerprint(lora_path=lora_path, lora_strength=lora_strength)
        cache_filename = f"transformer_{fingerprint}.safetensors"
        cache_path = os.path.join(cache_dir, cache_filename)
        cache_enabled = os.environ.get("TRANSFORMER_CACHE", "1") == "1"

        # Purger les anciens caches transformer (fingerprint different)
        for old_file in os.listdir(cache_dir):
            if old_file.startswith("transformer_") and old_file.endswith(".safetensors") and old_file != cache_filename:
                old_path = os.path.join(cache_dir, old_file)
                try:
                    size_gb = os.path.getsize(old_path) / 2**30
                    os.unlink(old_path)
                    log.info("Purge ancien cache transformer: %s (%.1fGB)", old_file, size_gb)
                except OSError as e:
                    log.warning("Echec purge %s: %s", old_file, e)

        if cache_enabled and os.path.isfile(cache_path):
            try:
                # Cache FP8 avec LoRA deja fuse — pas besoin de re-fuser
                self._transformer = self._load_transformer_cache(cache_path)
            except Exception as e:
                log.warning("Cache transformer invalide, rebuild: %s", e)
                try:
                    os.unlink(cache_path)
                except OSError:
                    pass

        if self._transformer is None:
            log.info("Build transformer distilled (FP8 cast%s)...",
                     " + LoRA depth" if lora_path else "")
            self._transformer = self._build_transformer_from_checkpoint(
                self._checkpoint_path,
                lora_path=lora_path,
                lora_strength=lora_strength,
            )

            if cache_enabled:
                self._save_transformer_cache(cache_path)

        # torch.compile sur les blocs transformer
        if os.environ.get("TORCH_COMPILE", "1") == "1":
            torch._inductor.config.fx_graph_cache = True
            # compile_threads DOIT etre fixe — sinon varie selon cpu_count() du worker
            # RunPod et change le hash du cache key (save_config_portable inclut cette valeur)
            torch._inductor.config.compile_threads = 12
            # cache_dir lu depuis env var TORCHINDUCTOR_CACHE_DIR (set dans start.sh)

            # === DIAGNOSTIC CACHE INDUCTOR ===
            self._log_inductor_cache_diagnostic()

            log.info(
                "Inductor config: fx_graph_cache=%s, autograd_cache=%s, cache_dir=%s",
                torch._inductor.config.fx_graph_cache,
                os.environ.get("TORCHINDUCTOR_AUTOGRAD_CACHE", "0"),
                os.environ.get("TORCHINDUCTOR_CACHE_DIR", "(default)"),
            )

            # Activer les logs Inductor seulement en mode debug
            if os.environ.get("LOG_LEVEL", "info").lower() == "debug":
                try:
                    torch._logging.set_logs(inductor=logging.DEBUG)
                    log.info("Inductor debug logging active (cache hit/miss visible)")
                except Exception as e:
                    log.warning("Inductor debug logging failed: %s", e)
            dynamic_compile = os.environ.get("DYNAMIC_COMPILE", "0") == "1"
            torch._dynamo.config.automatic_dynamic_shapes = dynamic_compile
            torch._dynamo.config.allow_unspec_int_on_nn_module = dynamic_compile
            torch._dynamo.config.force_parameter_static_shapes = not dynamic_compile
            torch._dynamo.config.cache_size_limit = 32
            torch._dynamo.config.recompile_limit = 48
            compile_mode = os.environ.get("COMPILE_MODE", "default")
            valid_modes = {"default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"}
            if compile_mode not in valid_modes:
                log.warning("COMPILE_MODE=%s invalide, fallback default", compile_mode)
                compile_mode = "default"
            fullgraph = os.environ.get("FULLGRAPH", "1") == "1"
            blocks = self._transformer.velocity_model.transformer_blocks
            log.info("torch.compile regional: %d blocs (mode=%s, fullgraph=%s, dynamic=%s)...",
                     len(blocks), compile_mode, fullgraph, dynamic_compile)
            for i, block in enumerate(blocks):
                blocks[i] = torch.compile(
                    block,
                    mode=compile_mode,
                    fullgraph=fullgraph,
                    dynamic=dynamic_compile,
                )

        self._log_vram("apres base transformer")
        return self._transformer

    def encode_prompt(self, prompt: str, negative: str | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode un prompt (cache hit ou Gemma GPU on-demand).

        Lookup steps :
        1. Cache RAM (par hash du prompt)
        2. Cache volume (fichier prompt_{hash}.pt)
        3. Sinon Gemma GPU → encode → cache RAM + volume → cleanup VRAM
        """
        # 1. Check RAM cache (par hash)
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]
        if prompt_hash in self._embeddings_cache:
            log.info("Prompt cache hit (RAM): %s", prompt[:40])
            return self._embeddings_cache[prompt_hash]

        # 3. Check volume cache (fichier individuel, peut venir d'un autre worker)
        if self._embeddings_cache_dir:
            cache_path = os.path.join(self._embeddings_cache_dir, f"prompt_{prompt_hash}.pt")
            if os.path.isfile(cache_path):
                try:
                    data = torch.load(cache_path, map_location=self.device, weights_only=True)
                    v_gpu, a_gpu = data["video"], data["audio"]
                    self._embeddings_cache[prompt_hash] = (v_gpu, a_gpu)
                    log.info("Prompt cache hit (volume): %s", prompt[:40])
                    return v_gpu, a_gpu
                except Exception as e:
                    log.warning("Cache volume corrompu %s: %s", cache_path, e)

        # 4. On-demand : charger Gemma GPU et encoder
        log.info("Encoding prompt custom via Gemma GPU: %s", prompt[:80])

        prompts_to_encode = [prompt]
        if negative and negative != DEFAULT_NEGATIVE_PROMPT:
            prompts_to_encode.append(negative)

        results = self._encode_prompts_with_gemma(prompts_to_encode)
        v_ctx, a_ctx = results[0].video_encoding, results[0].audio_encoding
        log.info("Gemma cleanup done")

        # Sauvegarder en cache RAM (GPU) + volume (CPU)
        v_gpu, a_gpu = v_ctx.to(self.device), a_ctx.to(self.device)
        self._embeddings_cache[prompt_hash] = (v_gpu, a_gpu)
        if self._embeddings_cache_dir:
            cache_path = os.path.join(self._embeddings_cache_dir, f"prompt_{prompt_hash}.pt")
            torch.save({"video": v_ctx.cpu(), "audio": a_ctx.cpu()}, cache_path)
            log.info("Prompt cached (RAM + volume): %s", prompt[:40])

        return v_gpu, a_gpu

    def _get_negative_embeddings(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Retourne les embeddings du negative prompt (toujours pre-cache)."""
        if "_negative" in self._embeddings_cache:
            return self._embeddings_cache["_negative"]
        return self.encode_prompt(DEFAULT_NEGATIVE_PROMPT)

    @torch.inference_mode()
    def generate_batch_frames(
        self,
        items: list[dict],
        prompt: str,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        image_strength: float = 1.0,
        two_stage: bool = False,
        on_item_decoded: Callable[[int, list[torch.Tensor]], None] | None = None,
    ) -> list[list[torch.Tensor]]:
        """Genere les frames pour un batch d'items (meme prompt, images/seeds differents).

        Chaque item dict doit avoir:
            image_path: str | None (optionnel pour t2v)
            seed: int
            mode: str (optionnel, default "i2v_depth": t2v, i2v, i2v_depth, flf2v)
            use_depth: bool (optionnel, gere le depth conditioning par item)
        Optionnel:
            last_image_path: str
            last_image_strength: float (defaut 1.0)

        Le denoising tourne en batch=N sur un seul forward transformer.
        Le VAE decode est sequentiel par item (limitation VideoDecoder).

        Returns:
            Liste de listes de frames CPU, une par item.
        """
        batch_size = len(items)
        log.info(
            "Batch generate: %d items, %dx%d, %s",
            batch_size, width, height, "2-stage" if two_stage else "1-stage",
        )

        # 1. Embeddings (partages pour tout le batch, meme prompt)
        v_context_p, a_context_p = self.encode_prompt(prompt)

        # Expand context au batch size
        if v_context_p.shape[0] == 1 and batch_size > 1:
            v_ctx_batch = v_context_p.expand(batch_size, -1, -1)
            a_ctx_batch = a_context_p.expand(batch_size, -1, -1)
        else:
            v_ctx_batch = v_context_p
            a_ctx_batch = a_context_p

        # 2. Transformer
        self._log_vram("debut batch generate")
        transformer = self.get_transformer()

        # 3. Setup
        use_res2s = os.environ.get("SAMPLER", "euler") == "res2s"
        stepper = Res2sDiffusionStep() if use_res2s else EulerDiffusionStep()
        generators = [
            torch.Generator(device=self.device).manual_seed(item["seed"])
            for item in items
        ]

        # --- Helper : creer N etats individuels puis concatener en batch ---
        def create_batched_states(
            h: int,
            w: int,
            initial_video_latents: torch.Tensor | None = None,
            initial_audio_latents: torch.Tensor | None = None,
            noise_scale: float = 1.0,
            extra_conditionings: list[ConditioningItem] | None = None,
        ) -> tuple[LatentState, LatentState, tuple[VideoLatentTools, AudioLatentTools]]:
            video_states: list[LatentState] = []
            audio_states: list[LatentState] = []

            shape_ref = VideoPixelShape(batch=1, frames=num_frames, width=w, height=h, fps=frame_rate)
            v_tools = self._make_video_tools(shape_ref)
            a_tools = self._make_audio_tools(shape_ref)

            for i, item in enumerate(items):
                noiser_i = GaussianNoiser(generator=generators[i])

                # Conditionings per item (batch=1) — premier frame optionnel
                images_i = []
                if item.get("image_path"):
                    images_i.append(ImageConditioningInput(path=item["image_path"], frame_idx=0, strength=image_strength))

                conds_i = image_conditionings_by_replacing_latent(
                    images=images_i,
                    height=h,
                    width=w,
                    video_encoder=self._video_encoder,
                    dtype=self.dtype,
                    device=self.device,
                )

                # Last frame optionnel (FLF2V) — ajouté avec guiding latent
                last_img = item.get("last_image_path")
                if last_img:
                    last_idx = (num_frames - 1) // 8
                    last_str = item.get("last_image_strength", 1.0)
                    last_conds = image_conditionings_by_adding_guiding_latent(
                        images=[ImageConditioningInput(path=last_img, frame_idx=last_idx, strength=last_str)],
                        height=h,
                        width=w,
                        video_encoder=self._video_encoder,
                        dtype=self.dtype,
                        device=self.device,
                    )
                    conds_i = conds_i + last_conds

                # IC-LoRA depth conditioning — gating per-item
                if extra_conditionings and item.get("use_depth", False):
                    conds_i = conds_i + [extra_conditionings[i]]

                init_vid_i = initial_video_latents[i:i+1] if initial_video_latents is not None else None
                init_aud_i = initial_audio_latents[i:i+1] if initial_audio_latents is not None else None

                vs_i = create_noised_state(
                    v_tools, conds_i, noiser_i, self.dtype, self.device,
                    noise_scale=noise_scale, initial_latent=init_vid_i,
                )
                as_i = create_noised_state(
                    a_tools, [], noiser_i, self.dtype, self.device,
                    noise_scale=noise_scale, initial_latent=init_aud_i,
                )

                video_states.append(vs_i)
                audio_states.append(as_i)

            # Concatener le long de batch dim
            batched_video = LatentState(
                latent=torch.cat([s.latent for s in video_states], dim=0),
                denoise_mask=torch.cat([s.denoise_mask for s in video_states], dim=0),
                positions=torch.cat([s.positions for s in video_states], dim=0),
                clean_latent=torch.cat([s.clean_latent for s in video_states], dim=0),
            )
            batched_audio = LatentState(
                latent=torch.cat([s.latent for s in audio_states], dim=0),
                denoise_mask=torch.cat([s.denoise_mask for s in audio_states], dim=0),
                positions=torch.cat([s.positions for s in audio_states], dim=0),
                clean_latent=torch.cat([s.clean_latent for s in audio_states], dim=0),
            )
            return batched_video, batched_audio, (v_tools, a_tools)

        # Denoiser batch (expand sigma pour batch_size)
        denoiser = _MedusaBatchDenoiser(v_ctx_batch, a_ctx_batch, batch_size)

        if two_stage:
            # --- 2-stage batch pipeline ---
            half_h, half_w = height // 2, width // 2

            # IC-LoRA depth conditioning per-item (Stage 1 only) — gating par item
            depth_conds: list[ConditioningItem | None] | None = None
            if self._depth_lora_enabled and self._depth_model is not None:
                t0_depth = time.perf_counter()
                depth_conds = []
                for item in items:
                    use_depth = item.get("use_depth", False)
                    if use_depth and item.get("image_path"):
                        camera_speed = item.get("camera_speed_ms", self._camera_speed_ms_default)
                        camera_path = item.get("camera_path")
                        interpolation = item.get("interpolation", "linear")
                        fov_degrees = item.get("fov_degrees", 60.0)
                        depth_conds.append(self.create_depth_conditioning(
                            item["image_path"], half_h, half_w, num_frames,
                            frame_rate=frame_rate, camera_speed_ms=camera_speed,
                            camera_path=camera_path,
                            interpolation=interpolation,
                            fov_degrees=fov_degrees,
                        ))
                    else:
                        depth_conds.append(None)
                log.info("Depth conditioning: %d items (%.1fs)", sum(1 for c in depth_conds if c is not None), time.perf_counter() - t0_depth)

            # Stage 1 — half-res batch denoise
            log.info("Batch Stage 1: denoise %dx%d (half-res, 8 steps)...", half_w, half_h)
            bv_s1, ba_s1, (vtools_s1, atools_s1) = create_batched_states(
                half_h, half_w, extra_conditionings=depth_conds,
            )

            bv_s1, ba_s1 = euler_denoising_loop(
                self._sigmas, bv_s1, ba_s1, stepper, transformer, denoiser,
            )
            bv_s1 = vtools_s1.clear_conditioning(bv_s1)
            bv_s1 = vtools_s1.unpatchify(bv_s1)
            ba_s1 = atools_s1.clear_conditioning(ba_s1)
            ba_s1 = atools_s1.unpatchify(ba_s1)
            self._log_vram("apres batch stage 1")
            torch.cuda.empty_cache()

            # Upscale batch
            log.info("Batch upscale latent x2 (%d items)...", batch_size)
            upscaled_batch = upsample_video(
                latent=bv_s1.latent,
                video_encoder=self._video_encoder,
                upsampler=self._spatial_upsampler,
            )
            self._log_vram("apres batch upscale")
            torch.cuda.empty_cache()

            # Stage 2 — full-res batch refine
            log.info("Batch Stage 2: refine %dx%d (full-res, 3 steps)...", width, height)

            bv_s2, ba_s2, (vtools_s2, atools_s2) = create_batched_states(
                height, width,
                initial_video_latents=upscaled_batch,
                initial_audio_latents=ba_s1.latent,
                noise_scale=self._stage2_sigmas[0].item(),
            )

            bv_s2, ba_s2 = euler_denoising_loop(
                self._stage2_sigmas, bv_s2, ba_s2, stepper, transformer, denoiser,
            )
            bv_s2 = vtools_s2.clear_conditioning(bv_s2)
            video_state = vtools_s2.unpatchify(bv_s2)
            self._log_vram("apres batch stage 2")

            torch.cuda.empty_cache()

        else:
            # --- 1-stage batch pipeline ---
            # IC-LoRA depth conditioning per-item — gating par item
            depth_conds_1s: list[ConditioningItem | None] | None = None
            if self._depth_lora_enabled and self._depth_model is not None:
                t0_depth = time.perf_counter()
                depth_conds_1s = []
                for item in items:
                    use_depth = item.get("use_depth", False)
                    if use_depth and item.get("image_path"):
                        camera_speed = item.get("camera_speed_ms", self._camera_speed_ms_default)
                        camera_path = item.get("camera_path")
                        interpolation = item.get("interpolation", "linear")
                        fov_degrees = item.get("fov_degrees", 60.0)
                        depth_conds_1s.append(self.create_depth_conditioning(
                            item["image_path"], height, width, num_frames,
                            frame_rate=frame_rate, camera_speed_ms=camera_speed,
                            camera_path=camera_path,
                            interpolation=interpolation,
                            fov_degrees=fov_degrees,
                        ))
                    else:
                        depth_conds_1s.append(None)
                log.info("Depth conditioning: %d items (%.1fs)", sum(1 for c in depth_conds_1s if c is not None), time.perf_counter() - t0_depth)

            log.info("Batch 1-stage: denoise %dx%d (8 steps)...", width, height)
            batched_video, batched_audio, (vtools, atools) = create_batched_states(
                height, width, extra_conditionings=depth_conds_1s,
            )

            batched_video, batched_audio = euler_denoising_loop(
                self._sigmas, batched_video, batched_audio, stepper, transformer, denoiser,
            )
            batched_video = vtools.clear_conditioning(batched_video)
            video_state = vtools.unpatchify(batched_video)

        # VAE decode sequentiel par item
        torch.cuda.empty_cache()
        tiling = TilingConfig.default() if os.environ.get("VAE_TILING", "0") == "1" else None
        log.info("Batch VAE decode (%d items, sequentiel)...", batch_size)
        all_frames: list[list[torch.Tensor]] = []
        for i in range(batch_size):
            t0 = time.perf_counter()
            item_latent = video_state.latent[i:i+1].contiguous()
            decoded = self._video_decoder.decode_video(item_latent, tiling, generators[i])
            item_frames = [chunk.cpu() for chunk in decoded]
            dt = time.perf_counter() - t0
            if dt > 5.0:
                log.warning("VAE decode item %d/%d: %.1fs (possible recompilation)", i, batch_size, dt)
            all_frames.append(item_frames)
            if on_item_decoded is not None:
                on_item_decoded(i, item_frames)

        torch.cuda.empty_cache()
        log.info("Batch frames generees: %d items", len(all_frames))
        return all_frames
