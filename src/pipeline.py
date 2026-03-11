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
import os
import time
import warnings
from collections.abc import Iterator

import safetensors
import torch

# Supprimer les warnings internes torch.compile/dynamo (pas de bug, juste du bruit)
warnings.filterwarnings("ignore", message=".*lru_cache.*", module=r"torch\._dynamo")
warnings.filterwarnings("ignore", message=".*To copy construct from a tensor.*", module=r"torch\.")
from safetensors.torch import load_file as load_safetensors
from safetensors.torch import save_file as save_safetensors

from ltx_core.components.diffusion_steps import EulerDiffusionStep, Res2sDiffusionStep
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.protocols import DiffusionStepProtocol
from ltx_core.model.upsampler import upsample_video
from ltx_core.model.video_vae import TilingConfig, decode_video as vae_decode_video

from ltx_core.types import LatentState, VideoPixelShape
from ltx_pipelines.utils import ModelLedger
from ltx_pipelines.utils.constants import DISTILLED_SIGMA_VALUES, STAGE_2_DISTILLED_SIGMA_VALUES
from ltx_pipelines.utils.args import ImageConditioningInput
from ltx_pipelines.utils.helpers import (
    cleanup_memory,
    denoise_audio_video,
    encode_prompts,
    image_conditionings_by_replacing_latent,
    modality_from_latent_state,
    noise_audio_state,
    noise_video_state,
    simple_denoising_func,
)
from ltx_pipelines.utils.samplers import euler_denoising_loop
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.types import PipelineComponents
from ltx_core.model.transformer.attention import Attention, PytorchAttention
from ltx_core.quantization import QuantizationPolicy

from prompts import CAMERA_PRESETS, DEFAULT_NEGATIVE_PROMPT

log = logging.getLogger("medusa")


class SageAttentionCallable:
    """SageAttention2++ (H100 sm_90), fallback SDPA si mask present."""

    def __init__(self) -> None:
        from sageattention import sageattn

        self._sageattn = sageattn
        self._fallback = PytorchAttention()

    def __call__(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is not None:
            return self._fallback(q, k, v, heads, mask)

        b, _, dim_head = q.shape
        dim_head //= heads
        q = q.to(torch.bfloat16)
        k = k.to(torch.bfloat16)
        v = v.to(torch.bfloat16)
        q, k, v = (t.view(b, -1, heads, dim_head).transpose(1, 2) for t in (q, k, v))
        out = self._sageattn(q, k, v, tensor_layout="HND", is_causal=False)
        out = out.transpose(1, 2).reshape(b, -1, heads * dim_head)
        return out


# Version du cache transformer (incrementer pour invalider tous les caches existants)
CACHE_VERSION = "v4"


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

        # Pipeline components (patchifiers + scale factors)
        self._components = PipelineComponents(dtype=self.dtype, device=self.device)

        # Base ModelLedger SANS quantization — pour video_encoder, video_decoder (VAE) et spatial upsampler
        self._base_ledger = ModelLedger(
            dtype=self.dtype,
            device=self.device,
            checkpoint_path=self._checkpoint_path,
            spatial_upsampler_path=self._upsampler_path,
        )

        # Video encoder (persistent en VRAM)
        self._video_encoder: torch.nn.Module | None = None

        # Video decoder (persistent en VRAM)
        self._video_decoder: torch.nn.Module | None = None

        # Spatial upsampler x2 (persistent en VRAM, ~1GB)
        self._spatial_upsampler: torch.nn.Module | None = None

        # Transformer distilled en VRAM (FP8 cast via QuantizationPolicy)
        self._transformer: torch.nn.Module | None = None

        # Embeddings cache (prompt/key -> (video_ctx, audio_ctx))
        self._embeddings_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

        # Sigmas distilled (8 steps stage 1, 3 steps stage 2)
        self._sigmas = torch.tensor(
            DISTILLED_SIGMA_VALUES, dtype=torch.float32, device=self.device
        )
        self._stage2_sigmas = torch.tensor(
            STAGE_2_DISTILLED_SIGMA_VALUES, dtype=torch.float32, device=self.device
        )

    def warmup_embeddings(self, cache_dir: str) -> None:
        """Charge les embeddings depuis cache disque (genere par warmup_embeddings.py).

        Si le cache n'existe pas, genere les embeddings avec Gemma sur GPU.
        """
        cache_path = os.path.join(cache_dir, "embeddings_cache.pt")

        # Si cache existe deja, charger directement
        if os.path.isfile(cache_path):
            log.info("Chargement embeddings depuis cache: %s", cache_path)
            self._load_embeddings_cache(cache_path)
            return

        log.info("Pas de cache embeddings — generation avec text encoder sur GPU...")

        # Creer un ledger dedie au text encoder
        te_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        cpu_ledger = ModelLedger(
            dtype=self.dtype,
            device=te_device,
            checkpoint_path=self._checkpoint_path,
            gemma_root_path=self._gemma_root,
        )

        # Encoder tous les prompts : 7 presets camera + 1 negative
        all_prompts = list(CAMERA_PRESETS.values()) + [DEFAULT_NEGATIVE_PROMPT]
        all_keys = list(CAMERA_PRESETS.keys()) + ["_negative"]

        log.info("Encoding %d prompts...", len(all_prompts))
        results = encode_prompts(all_prompts, cpu_ledger)

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

        del cpu_ledger
        cleanup_memory()

    def load_video_encoder(self) -> None:
        """Charge le video encoder en VRAM (~1GB). Appeler apres warmup_embeddings."""
        log.info("Chargement video encoder (persistent)...")
        self._video_encoder = self._base_ledger.video_encoder()
        self._log_vram("apres video encoder")

    def load_video_decoder(self) -> None:
        """Charge le video decoder en VRAM (~2GB). Persistent entre jobs."""
        log.info("Chargement video decoder (persistent)...")
        self._video_decoder = self._base_ledger.video_decoder()
        if os.environ.get("VAE_COMPILE", "1") == "1":
            log.info("torch.compile video decoder (mode=default, dynamic=True)...")
            self._video_decoder = torch.compile(
                self._video_decoder, mode="default", fullgraph=False, dynamic=True,
            )
        self._log_vram("apres video decoder")

    def load_spatial_upsampler(self) -> None:
        """Charge le spatial upsampler x2 en VRAM (~1GB). Persistent entre jobs."""
        log.info("Chargement spatial upsampler x2 (persistent)...")
        self._spatial_upsampler = self._base_ledger.spatial_upsampler()
        self._log_vram("apres spatial upsampler")

    def _load_embeddings_cache(self, cache_path: str) -> None:
        """Charge les embeddings depuis un fichier .pt."""
        cache_data = torch.load(cache_path, map_location="cpu", weights_only=True)
        for key, tensors in cache_data.items():
            self._embeddings_cache[key] = (
                tensors["video"].to(self.device),
                tensors["audio"].to(self.device),
            )
        log.info("Embeddings charges: %d prompts", len(self._embeddings_cache))

    @staticmethod
    def _log_vram(label: str) -> None:
        """Log l'utilisation VRAM courante (alloue + reserve)."""
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 2**30
            rsvd = torch.cuda.memory_reserved() / 2**30
            log.info("VRAM [%s]: %.2fGB alloc, %.2fGB reserved", label, alloc, rsvd)

    @staticmethod
    def _log_cache_stats(label: str) -> None:
        """Log le nombre de fichiers dans les caches Inductor et Triton."""
        import subprocess
        for name, env_key in [("inductor", "TORCHINDUCTOR_CACHE_DIR"), ("triton", "TRITON_CACHE_DIR")]:
            cache_dir = os.environ.get(env_key)
            if not cache_dir:
                continue
            result = subprocess.run(
                ["find", cache_dir, "-type", "f"],
                capture_output=True, text=True, timeout=5,
            )
            count = len(result.stdout.strip().split("\n")) if result.stdout.strip() else 0
            log.info("Cache %s [%s]: %d fichiers (%s)", name, label, count, cache_dir)

    def _get_orig_module(self) -> torch.nn.Module:
        """Unwrap torch.compile OptimizedModule si besoin."""
        mod = self._transformer
        if hasattr(mod, "_orig_mod"):
            return mod._orig_mod
        return mod

    @staticmethod
    def _patch_sage_attention(transformer: torch.nn.Module) -> int:
        """Remplace attention_function par SageAttention2++ sur les modules Attention."""
        sage_attn = SageAttentionCallable()
        patched = 0
        for module in transformer.modules():
            if isinstance(module, Attention):
                module.attention_function = sage_attn
                patched += 1
        return patched

    def _cache_fingerprint(self) -> str:
        """Hash d'invalidation pour le cache transformer pre-fusionne.

        Combine CACHE_VERSION et taille du checkpoint. Retourne un hash court (16 chars).
        """
        h = hashlib.sha256()
        h.update(CACHE_VERSION.encode())
        h.update(str(os.path.getsize(self._checkpoint_path)).encode())
        return h.hexdigest()[:16]

    def _save_transformer_cache(self, cache_path: str) -> None:
        """Sauvegarde le transformer fusionne (distilled) en format checkpoint.

        Extrait le state_dict du LTXModel interne, re-ajoute le prefixe
        `model.diffusion_model.` pour compatibilite avec ModelLedger,
        et copie les metadonnees du checkpoint original.
        Ecriture atomique via .tmp + os.replace().
        """
        module = self._get_orig_module()
        raw_sd = module.velocity_model.state_dict()

        # Re-ajouter le prefixe checkpoint (inverse du renaming fait par ModelLedger)
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

    def _load_transformer_cache(self, cache_path: str) -> torch.nn.Module:
        """Charge le transformer pre-fusionne depuis le cache.

        Utilise ModelLedger sans LoRAs (le fichier contient deja les poids fusionnes).
        Retourne un X0Model pret a l'emploi.
        """
        log.info("Chargement transformer depuis cache: %s", cache_path)
        ledger = ModelLedger(
            dtype=self.dtype,
            device=self.device,
            checkpoint_path=cache_path,
            loras=[],
            quantization=QuantizationPolicy.fp8_cast(),
        )
        transformer = ledger.transformer()
        del ledger
        return transformer

    def get_transformer(self) -> torch.nn.Module:
        """Retourne le transformer distilled avec FP8 Cast.

        Checkpoint distilled BF16 + QuantizationPolicy.fp8_cast() → stockage FP8, compute BF16.
        Compatible torch.compile + SageAttention.
        """
        if self._transformer is not None:
            return self._transformer

        cache_dir = os.path.join(os.path.dirname(self.models_dir), "cache", "transformer")
        os.makedirs(cache_dir, exist_ok=True)
        fingerprint = self._cache_fingerprint()
        cache_path = os.path.join(cache_dir, f"transformer_{fingerprint}.safetensors")
        cache_enabled = os.environ.get("TRANSFORMER_CACHE", "1") == "1"

        if cache_enabled and os.path.isfile(cache_path):
            try:
                self._transformer = self._load_transformer_cache(cache_path)
            except Exception as e:
                log.warning("Cache transformer invalide, rebuild: %s", e)
                try:
                    os.unlink(cache_path)
                except OSError:
                    pass

        if self._transformer is None:
            log.info("Build transformer distilled (FP8 cast)...")
            ledger = ModelLedger(
                dtype=self.dtype,
                device=self.device,
                checkpoint_path=self._checkpoint_path,
                quantization=QuantizationPolicy.fp8_cast(),
            )
            self._transformer = ledger.transformer()
            del ledger

            if cache_enabled:
                self._save_transformer_cache(cache_path)

        # SageAttention2++ (remplace SDPA sur les modules Attention du transformer)
        sage_active = False
        if os.environ.get("SAGE_ATTENTION", "1") == "1":
            try:
                patched = self._patch_sage_attention(self._transformer)
                log.info("SageAttention2++ active: %d modules patches", patched)
                sage_active = patched > 0
            except ImportError:
                log.warning("SageAttention non installe, fallback SDPA")
            except Exception as e:
                log.warning("SageAttention init echoue, fallback SDPA: %s", e)

        # torch.compile — dynamic=True pour eviter les recompilations entre stages
        # SA incompatible CUDA graphs → max-autotune-no-cudagraphs (autotuning Triton sans CUDA graphs).
        if os.environ.get("TORCH_COMPILE", "1") == "1":
            torch._inductor.config.fx_graph_cache = True
            log.info(
                "Inductor config: fx_graph_cache=%s, autograd_cache=%s, cache_dir=%s",
                torch._inductor.config.fx_graph_cache,
                os.environ.get("TORCHINDUCTOR_AUTOGRAD_CACHE", "0"),
                os.environ.get("TORCHINDUCTOR_CACHE_DIR", "(default)"),
            )
            torch._dynamo.config.automatic_dynamic_shapes = True
            torch._dynamo.config.allow_unspec_int_on_nn_module = True
            torch._dynamo.config.cache_size_limit = 32
            torch._dynamo.config.recompile_limit = 16
            if sage_active:
                compile_mode = "max-autotune-no-cudagraphs"
            else:
                compile_mode = os.environ.get("COMPILE_MODE", "reduce-overhead")
                valid_modes = {"default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"}
                if compile_mode not in valid_modes:
                    log.warning("COMPILE_MODE=%s invalide, fallback reduce-overhead", compile_mode)
                    compile_mode = "reduce-overhead"
            log.info("torch.compile transformer (mode=%s, sage=%s, dynamic=True)...", compile_mode, sage_active)
            self._transformer = torch.compile(
                self._transformer,
                mode=compile_mode,
                fullgraph=False,
                dynamic=True,
            )

        self._log_vram("apres base transformer")
        return self._transformer

    def encode_prompt(self, prompt: str, negative: str | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode un prompt (cache hit ou Gemma GPU on-demand).

        1. Verifie si le prompt est un preset connu dans le cache
        2. Sinon charge Gemma GPU → encode → cleanup VRAM → retourne
        """
        # Chercher dans le cache par valeur (preset text → key)
        for key, preset_text in CAMERA_PRESETS.items():
            if prompt == preset_text and key in self._embeddings_cache:
                return self._embeddings_cache[key]

        # Chercher par key directe (ex: "dolly-in")
        if prompt in self._embeddings_cache:
            return self._embeddings_cache[prompt]

        # On-demand : charger Gemma GPU et encoder
        log.info("Encoding prompt custom via Gemma GPU: %s", prompt[:80])
        te_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        te_ledger = ModelLedger(
            dtype=self.dtype,
            device=te_device,
            checkpoint_path=self._checkpoint_path,
            gemma_root_path=self._gemma_root,
        )

        prompts_to_encode = [prompt]
        if negative and negative != DEFAULT_NEGATIVE_PROMPT:
            prompts_to_encode.append(negative)

        results = encode_prompts(prompts_to_encode, te_ledger)
        v_ctx, a_ctx = results[0].video_encoding, results[0].audio_encoding

        del te_ledger
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log.info("Gemma cleanup done")

        return v_ctx.to(self.device), a_ctx.to(self.device)

    def _get_negative_embeddings(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Retourne les embeddings du negative prompt (toujours pre-cache)."""
        if "_negative" in self._embeddings_cache:
            return self._embeddings_cache["_negative"]
        return self.encode_prompt(DEFAULT_NEGATIVE_PROMPT)

    @torch.inference_mode()
    def generate_frames(
        self,
        image_path: str,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        image_strength: float = 1.0,
        last_image_path: str | None = None,
        last_image_strength: float = 1.0,
        negative_override: str | None = None,
        two_stage: bool = False,
    ) -> list[torch.Tensor]:
        """Genere les frames I2V sans encoder en MP4.

        Retourne les frames decodees sur CPU pour post-processing async.
        """
        generator = torch.Generator(device=self.device).manual_seed(seed)

        # 1. Embeddings (cache hit si preset, Gemma on-demand sinon)
        v_context_p, a_context_p = self.encode_prompt(prompt)

        # 2. Transformer (deja build au startup)
        self._log_vram("debut generate")
        transformer = self.get_transformer()

        # 3. Setup denoising
        noiser = GaussianNoiser(generator=generator)
        use_res2s = os.environ.get("SAMPLER", "euler") == "res2s"
        stepper = Res2sDiffusionStep() if use_res2s else EulerDiffusionStep()

        # 4. Image list
        images = [ImageConditioningInput(path=image_path, frame_idx=0, strength=image_strength)]
        if last_image_path is not None:
            last_latent_idx = (num_frames - 1) // 8
            images.append(ImageConditioningInput(path=last_image_path, frame_idx=last_latent_idx, strength=last_image_strength))

        # 5. Denoising loop — audio disabled (single CUDA graph, no guidance)
        # CFG=1.0, STG=0.0 → guiders are no-ops, audio not used for video generation.
        # Keeping audio.enabled constant across all steps avoids Dynamo recompilation.
        def denoising_loop_guided(
            sigmas: torch.Tensor,
            video_state: LatentState,
            audio_state: LatentState,
            stepper_arg: DiffusionStepProtocol,
        ) -> tuple[LatentState, LatentState]:

            def denoise_step(
                video_state: LatentState,
                audio_state: LatentState,
                sigmas: torch.Tensor,
                step_index: int,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                t0 = time.perf_counter()
                sigma = sigmas[step_index]
                vid_mod = modality_from_latent_state(
                    video_state, v_context_p, sigma, enabled=True,
                )
                aud_mod = modality_from_latent_state(
                    audio_state, a_context_p, sigma, enabled=False,
                )
                denoised_video, denoised_audio = transformer(
                    video=vid_mod, audio=aud_mod, perturbations=None,
                )
                torch.cuda.synchronize()
                dt = time.perf_counter() - t0
                if step_index == 0:
                    alloc = torch.cuda.memory_allocated() / 2**30
                    rsvd = torch.cuda.memory_reserved() / 2**30
                    log.info("step %d: %.2fs (sigma=%.4f) VRAM %.2fGB alloc %.2fGB rsvd",
                             step_index, dt, sigma.item(), alloc, rsvd)
                    self._log_cache_stats("apres step 0")
                else:
                    log.debug("step %d: %.2fs (sigma=%.4f)", step_index, dt, sigma.item())
                if dt > 10.0:
                    log.warning("step %d took %.1fs — possible Dynamo recompilation", step_index, dt)
                return denoised_video, denoised_audio

            return euler_denoising_loop(
                sigmas=sigmas,
                video_state=video_state,
                audio_state=audio_state,
                stepper=stepper_arg,
                denoise_fn=denoise_step,
            )

        if two_stage:
            # --- 2-stage pipeline (1080p) ---
            half_h = height // 2
            half_w = width // 2

            # Stage 1 — denoise at half-res (8 steps)
            log.info("Stage 1: denoise %dx%d (half-res, 8 steps)...", half_w, half_h)
            conditionings_s1 = image_conditionings_by_replacing_latent(
                images=images,
                height=half_h,
                width=half_w,
                video_encoder=self._video_encoder,
                dtype=self.dtype,
                device=self.device,
            )

            output_shape_s1 = VideoPixelShape(
                batch=1, frames=num_frames, width=half_w, height=half_h, fps=frame_rate,
            )

            video_state_s1, audio_state_s1 = denoise_audio_video(
                output_shape=output_shape_s1,
                conditionings=conditionings_s1,
                noiser=noiser,
                sigmas=self._sigmas,
                stepper=stepper,
                denoising_loop_fn=denoising_loop_guided,
                components=self._components,
                dtype=self.dtype,
                device=self.device,
            )
            self._log_vram("apres stage 1")
            torch.cuda.empty_cache()

            # Spatial upscale x2 in latent space
            log.info("Upscale latent x2...")
            upscaled_latent = upsample_video(
                latent=video_state_s1.latent[:1],
                video_encoder=self._video_encoder,
                upsampler=self._spatial_upsampler,
            )
            self._log_vram("apres upscale")
            torch.cuda.empty_cache()

            # Stage 2 — refine at full-res (3 steps)
            log.info("Stage 2: refine %dx%d (full-res, 3 steps)...", width, height)
            conditionings_s2 = image_conditionings_by_replacing_latent(
                images=images,
                height=height,
                width=width,
                video_encoder=self._video_encoder,
                dtype=self.dtype,
                device=self.device,
            )

            def denoising_loop_s2(
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
                    denoise_fn=simple_denoising_func(
                        video_context=v_context_p,
                        audio_context=a_context_p,
                        transformer=transformer,
                    ),
                )

            output_shape_s2 = VideoPixelShape(
                batch=1, frames=num_frames, width=width, height=height, fps=frame_rate,
            )

            video_state, _audio_state = denoise_audio_video(
                output_shape=output_shape_s2,
                conditionings=conditionings_s2,
                noiser=noiser,
                sigmas=self._stage2_sigmas,
                stepper=stepper,
                denoising_loop_fn=denoising_loop_s2,
                components=self._components,
                dtype=self.dtype,
                device=self.device,
                noise_scale=self._stage2_sigmas[0].item(),
                initial_video_latent=upscaled_latent,
                initial_audio_latent=audio_state_s1.latent,
            )
            self._log_vram("apres stage 2")

        else:
            # --- 1-stage pipeline (720p) ---
            log.info("1-stage: denoise %dx%d (8 steps)...", width, height)
            conditionings = image_conditionings_by_replacing_latent(
                images=images,
                height=height,
                width=width,
                video_encoder=self._video_encoder,
                dtype=self.dtype,
                device=self.device,
            )

            output_shape = VideoPixelShape(
                batch=1, frames=num_frames, width=width, height=height, fps=frame_rate,
            )

            video_state, _audio_state = denoise_audio_video(
                output_shape=output_shape,
                conditionings=conditionings,
                noiser=noiser,
                sigmas=self._sigmas,
                stepper=stepper,
                denoising_loop_fn=denoising_loop_guided,
                components=self._components,
                dtype=self.dtype,
                device=self.device,
            )

        torch.cuda.empty_cache()

        # 7. VAE decode (tiling optionnel pour 1080p)
        tiling = TilingConfig.default() if os.environ.get("VAE_TILING", "0") == "1" else None
        log.info("VAE decode%s...", " (tiled)" if tiling else "")
        decoded_video: Iterator[torch.Tensor] = vae_decode_video(
            video_state.latent,
            self._video_decoder,
            tiling,
            generator,
        )

        # 8. Materialiser frames sur CPU pour post-processing async
        frames = [chunk.cpu() for chunk in decoded_video]
        log.info("Frames generees: %d chunks", len(frames))
        return frames

    @torch.inference_mode()
    def generate(
        self,
        image_path: str,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        output_path: str,
        image_strength: float = 1.0,
        last_image_path: str | None = None,
        last_image_strength: float = 1.0,
        negative_override: str | None = None,
        two_stage: bool = False,
    ) -> None:
        """Genere une video I2V et sauvegarde en MP4 (retro-compatible).

        Delegue a generate_frames() puis encode le MP4.
        """
        frames = self.generate_frames(
            image_path=image_path,
            prompt=prompt,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            image_strength=image_strength,
            last_image_path=last_image_path,
            last_image_strength=last_image_strength,
            negative_override=negative_override,
            two_stage=two_stage,
        )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        encode_video(
            video=iter(frames),
            fps=int(frame_rate),
            audio=None,
            output_path=output_path,
            video_chunks_number=1,
        )
        log.info("Video sauvegardee: %s", output_path)

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
    ) -> list[list[torch.Tensor]]:
        """Genere les frames pour un batch d'items (meme prompt, images/seeds differents).

        Chaque item dict doit avoir:
            image_path: str (fichier local)
            seed: int
        Optionnel:
            last_image_path: str
            last_image_strength: float (defaut 1.0)

        Le denoising tourne en batch=N sur un seul forward transformer.
        Le VAE decode est sequentiel par item (limitation vae_decode_video).

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
        ) -> tuple[LatentState, LatentState, tuple]:
            video_states: list[LatentState] = []
            audio_states: list[LatentState] = []
            tools_ref = None

            for i, item in enumerate(items):
                noiser_i = GaussianNoiser(generator=generators[i])

                # Conditionings per item (batch=1)
                images_i = [ImageConditioningInput(path=item["image_path"], frame_idx=0, strength=image_strength)]
                last_img = item.get("last_image_path")
                if last_img:
                    last_idx = (num_frames - 1) // 8
                    last_str = item.get("last_image_strength", 1.0)
                    images_i.append(ImageConditioningInput(path=last_img, frame_idx=last_idx, strength=last_str))

                conds_i = image_conditionings_by_replacing_latent(
                    images=images_i,
                    height=h,
                    width=w,
                    video_encoder=self._video_encoder,
                    dtype=self.dtype,
                    device=self.device,
                )

                shape_i = VideoPixelShape(
                    batch=1, frames=num_frames, width=w, height=h, fps=frame_rate,
                )

                init_vid_i = initial_video_latents[i:i+1] if initial_video_latents is not None else None
                init_aud_i = initial_audio_latents[i:i+1] if initial_audio_latents is not None else None

                vs_i, vt_i = noise_video_state(
                    output_shape=shape_i,
                    noiser=noiser_i,
                    conditionings=conds_i,
                    components=self._components,
                    dtype=self.dtype,
                    device=self.device,
                    noise_scale=noise_scale,
                    initial_latent=init_vid_i,
                )
                as_i, at_i = noise_audio_state(
                    output_shape=shape_i,
                    noiser=noiser_i,
                    conditionings=[],
                    components=self._components,
                    dtype=self.dtype,
                    device=self.device,
                    noise_scale=noise_scale,
                    initial_latent=init_aud_i,
                )

                video_states.append(vs_i)
                audio_states.append(as_i)
                if tools_ref is None:
                    tools_ref = (vt_i, at_i)

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
            return batched_video, batched_audio, tools_ref

        # --- Denoising loop avec context batche ---
        def denoising_loop_guided(
            sigmas: torch.Tensor,
            video_state: LatentState,
            audio_state: LatentState,
            stepper_arg: DiffusionStepProtocol,
        ) -> tuple[LatentState, LatentState]:

            def denoise_step(
                video_state: LatentState,
                audio_state: LatentState,
                sigmas: torch.Tensor,
                step_index: int,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                t0 = time.perf_counter()
                sigma = sigmas[step_index]
                vid_mod = modality_from_latent_state(
                    video_state, v_ctx_batch, sigma, enabled=True,
                )
                aud_mod = modality_from_latent_state(
                    audio_state, a_ctx_batch, sigma, enabled=False,
                )
                # Fix: expand scalar sigma to (batch_size,) for prompt_adaln
                # in transformer._prepare_timestep (needs batch dim, not scalar)
                sigma_b = sigma.unsqueeze(0).expand(batch_size)
                vid_mod = dataclasses.replace(vid_mod, sigma=sigma_b)
                aud_mod = dataclasses.replace(aud_mod, sigma=sigma_b)
                denoised_video, denoised_audio = transformer(
                    video=vid_mod, audio=aud_mod, perturbations=None,
                )
                torch.cuda.synchronize()
                dt = time.perf_counter() - t0
                if step_index == 0:
                    alloc = torch.cuda.memory_allocated() / 2**30
                    rsvd = torch.cuda.memory_reserved() / 2**30
                    log.info(
                        "batch step %d: %.2fs (sigma=%.4f) VRAM %.2fGB alloc %.2fGB rsvd",
                        step_index, dt, sigma.item(), alloc, rsvd,
                    )
                    self._log_cache_stats("apres batch step 0")
                else:
                    log.debug("batch step %d: %.2fs (sigma=%.4f)", step_index, dt, sigma.item())
                if dt > 10.0:
                    log.warning("batch step %d took %.1fs — possible Dynamo recompilation", step_index, dt)
                return denoised_video, denoised_audio

            return euler_denoising_loop(
                sigmas=sigmas,
                video_state=video_state,
                audio_state=audio_state,
                stepper=stepper_arg,
                denoise_fn=denoise_step,
            )

        if two_stage:
            # --- 2-stage batch pipeline ---
            half_h, half_w = height // 2, width // 2

            # Stage 1 — half-res batch denoise
            log.info("Batch Stage 1: denoise %dx%d (half-res, 8 steps)...", half_w, half_h)
            bv_s1, ba_s1, (vtools_s1, atools_s1) = create_batched_states(half_h, half_w)

            bv_s1, ba_s1 = denoising_loop_guided(self._sigmas, bv_s1, ba_s1, stepper)
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

            def denoising_loop_s2(
                sigmas: torch.Tensor,
                video_state: LatentState,
                audio_state: LatentState,
                stepper_arg: DiffusionStepProtocol,
            ) -> tuple[LatentState, LatentState]:
                def denoise_step_s2(
                    video_state: LatentState,
                    audio_state: LatentState,
                    sigmas: torch.Tensor,
                    step_index: int,
                ) -> tuple[torch.Tensor, torch.Tensor]:
                    sigma = sigmas[step_index]
                    vid_mod = modality_from_latent_state(video_state, v_ctx_batch, sigma)
                    aud_mod = modality_from_latent_state(audio_state, a_ctx_batch, sigma)
                    # Fix: expand scalar sigma to (batch_size,) for prompt_adaln
                    sigma_b = sigma.unsqueeze(0).expand(batch_size)
                    vid_mod = dataclasses.replace(vid_mod, sigma=sigma_b)
                    aud_mod = dataclasses.replace(aud_mod, sigma=sigma_b)
                    return transformer(video=vid_mod, audio=aud_mod, perturbations=None)

                return euler_denoising_loop(
                    sigmas=sigmas,
                    video_state=video_state,
                    audio_state=audio_state,
                    stepper=stepper_arg,
                    denoise_fn=denoise_step_s2,
                )

            bv_s2, ba_s2, (vtools_s2, atools_s2) = create_batched_states(
                height, width,
                initial_video_latents=upscaled_batch,
                initial_audio_latents=ba_s1.latent,
                noise_scale=self._stage2_sigmas[0].item(),
            )

            bv_s2, ba_s2 = denoising_loop_s2(self._stage2_sigmas, bv_s2, ba_s2, stepper)
            bv_s2 = vtools_s2.clear_conditioning(bv_s2)
            video_state = vtools_s2.unpatchify(bv_s2)
            self._log_vram("apres batch stage 2")

            torch.cuda.empty_cache()

        else:
            # --- 1-stage batch pipeline ---
            log.info("Batch 1-stage: denoise %dx%d (8 steps)...", width, height)
            batched_video, batched_audio, (vtools, atools) = create_batched_states(height, width)

            batched_video, batched_audio = denoising_loop_guided(
                self._sigmas, batched_video, batched_audio, stepper,
            )
            batched_video = vtools.clear_conditioning(batched_video)
            video_state = vtools.unpatchify(batched_video)

        # VAE decode sequentiel par item (vae_decode_video ne supporte pas le batching)
        torch.cuda.empty_cache()
        tiling = TilingConfig.default() if os.environ.get("VAE_TILING", "0") == "1" else None
        log.info("Batch VAE decode (%d items, sequentiel)...", batch_size)
        all_frames: list[list[torch.Tensor]] = []
        for i in range(batch_size):
            item_latent = video_state.latent[i:i+1]
            decoded = vae_decode_video(item_latent, self._video_decoder, tiling, generators[i])
            item_frames = [chunk.cpu() for chunk in decoded]
            all_frames.append(item_frames)

        torch.cuda.empty_cache()
        log.info("Batch frames generees: %d items", len(all_frames))
        return all_frames
