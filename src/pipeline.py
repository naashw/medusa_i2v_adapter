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
from collections.abc import Callable, Iterator

import json
import tempfile

import numpy as np
import safetensors
import torch
from PIL import Image

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
)
from ltx_pipelines.utils.samplers import euler_denoising_loop
from video_encoder import encode_video_fast
from ltx_pipelines.utils.types import PipelineComponents
from ltx_core.quantization import QuantizationPolicy

from prompts import CAMERA_PRESETS, DEFAULT_NEGATIVE_PROMPT

log = logging.getLogger("medusa")

# Log SDPA backend (cuDNN Fused Flash Attention natif sur H100, pas de dep externe)
if torch.cuda.is_available():
    log.info("SDPA backend: cuDNN attention (natif PyTorch, H100 sm_90)")


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

        # --- Camera LoRA config ---
        self._camera_lora_enabled = os.environ.get("CAMERA_LORA", "1") == "1"
        self._camera_lora_strength = float(os.environ.get("CAMERA_LORA_STRENGTH", "0.8"))
        self._lora_dir = os.path.join(models_dir, "loras")

        # Registry : camera_motion -> lora filename (extensible)
        self._camera_lora_registry: dict[str, str] = {
            "dolly-in": "ltx-2-19b-lora-camera-control-dolly-in.safetensors",
        }

        # --- Depth IC-LoRA config ---
        self._depth_lora_enabled = os.environ.get("DEPTH_LORA", "1") == "1"
        self._depth_lora_strength = float(os.environ.get("DEPTH_LORA_STRENGTH", "0.8"))
        self._depth_lora_name = "depth"
        self._depth_lora_file = "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"
        self._depth_model: torch.nn.Module | None = None

        # Runtime state
        self._active_lora: str | None = None
        self._lora_deltas: dict[str, dict[str, torch.Tensor]] = {}

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
            vae_dynamic = os.environ.get("DYNAMIC_COMPILE", "0") == "1"
            log.info("torch.compile video decoder (mode=default, dynamic=%s)...", vae_dynamic)
            self._video_decoder = torch.compile(
                self._video_decoder, mode="default", fullgraph=False, dynamic=vae_dynamic,
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

    @torch.inference_mode()
    def warmup_compile(self, num_frames: int = 25, frame_rate: float = 24) -> None:
        """Pre-compile le transformer pour toutes les shapes (720p + 1080p 2-stage, landscape + portrait).

        Lance un forward dummy par resolution pour declencher la compilation
        Dynamo de toutes les variantes attention/shapes. Elimine les recompilations
        couteuses (~30s) du premier job reel.
        """
        from PIL import Image

        log.info("Warmup compile: pre-compilation transformer...")
        t0_total = time.perf_counter()

        transformer = self.get_transformer()

        # Image dummy (blank, sera remplacee par le conditioning latent)
        tmp_path = "/tmp/warmup_compile_dummy.png"
        Image.new("RGB", (64, 64), (128, 128, 128)).save(tmp_path)

        # Embeddings depuis le cache (premier preset disponible)
        first_key = next(iter(self._embeddings_cache))
        v_ctx, a_ctx = self._embeddings_cache[first_key]

        gen = torch.Generator(device=self.device).manual_seed(0)
        noiser = GaussianNoiser(generator=gen)

        # Configs couvrant toutes les shapes (landscape + portrait).
        # audio.enabled=False partout (on ne genere pas d'audio) → un seul chemin Dynamo par shape.
        # Portrait = landscape transpose (memes megapixels, sequences de tokens differentes)
        # Note : 540p (544×960) = meme shape que 1080p-s1, pas de warmup supplementaire
        configs: list[tuple[str, int, int, torch.Tensor]] = [
            # Tier 2 — Standard 720p 2-stage
            ("720p-s1",              352,  640,  self._sigmas),
            ("720p-s2",              704,  1280, self._stage2_sigmas),
            # Tier 3 — Production 1080p 2-stage
            ("1080p-s1",             544,  960,  self._sigmas),
            ("1080p-s2",             1088, 1920, self._stage2_sigmas),
            # Portrait 9:16
            ("720p-portrait-s1",     640,  352,  self._sigmas),
            ("720p-portrait-s2",     1280, 704,  self._stage2_sigmas),
            ("1080p-portrait-s1",    960,  544,  self._sigmas),
            ("1080p-portrait-s2",    1920, 1088, self._stage2_sigmas),
        ]

        for label, h, w, sigmas in configs:
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

            vs, _ = noise_video_state(
                output_shape=shape, noiser=noiser, conditionings=conds,
                components=self._components, dtype=self.dtype, device=self.device,
            )
            as_, _ = noise_audio_state(
                output_shape=shape, noiser=noiser, conditionings=[],
                components=self._components, dtype=self.dtype, device=self.device,
            )

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

    def _get_orig_module(self) -> torch.nn.Module:
        """Unwrap torch.compile OptimizedModule si besoin."""
        mod = self._transformer
        if hasattr(mod, "_orig_mod"):
            return mod._orig_mod
        return mod

    # ------------------------------------------------------------------
    # Camera LoRA : load / fuse / unfuse / ensure
    # ------------------------------------------------------------------

    def _load_lora_deltas(self, name: str) -> None:
        """Pre-calcule les deltas LoRA (B @ A * strength) pour un camera motion.

        Charge le safetensors, renomme les cles si necessaire, et stocke
        les deltas pre-calcules sur CPU pour fuse/unfuse rapide.
        """
        filename = self._camera_lora_registry.get(name)
        if not filename:
            return
        lora_path = os.path.join(self._lora_dir, filename)
        if not os.path.isfile(lora_path):
            log.warning("LoRA '%s' introuvable: %s", name, lora_path)
            return

        log.info(
            "Chargement LoRA '%s': %s (strength=%.2f)",
            name, filename, self._camera_lora_strength,
        )
        raw = load_safetensors(lora_path, device=str(self.device))

        # Log les cles pour diagnostic
        sample_keys = list(raw.keys())[:5]
        log.info("LoRA '%s' sample keys: %s", name, sample_keys)

        # Grouper les paires lora_A/lora_B par cle de base
        # Convention : strip "diffusion_model." prefix (renaming Comfy → ltx-core interne)
        pairs: dict[str, dict[str, torch.Tensor]] = {}
        for key, tensor in raw.items():
            clean = key.replace("diffusion_model.", "", 1) if key.startswith("diffusion_model.") else key

            if ".lora_A.weight" in clean:
                base = clean.replace(".lora_A.weight", "")
                pairs.setdefault(base, {})["A"] = tensor
            elif ".lora_B.weight" in clean:
                base = clean.replace(".lora_B.weight", "")
                pairs.setdefault(base, {})["B"] = tensor
            elif ".alpha" in clean:
                base = clean.replace(".alpha", "")
                pairs.setdefault(base, {})["alpha"] = tensor

        # Calculer delta = (B @ A) * (alpha/rank) * strength
        deltas: dict[str, torch.Tensor] = {}
        for base_key, pair in pairs.items():
            if "A" not in pair or "B" not in pair:
                continue
            a_tensor = pair["A"].to(dtype=self.dtype)
            b_tensor = pair["B"].to(dtype=self.dtype)
            delta = b_tensor @ a_tensor

            if "alpha" in pair:
                rank = a_tensor.shape[0]
                alpha_val = pair["alpha"].item()
                delta = delta * (alpha_val / rank)

            delta = delta * self._camera_lora_strength
            # Prepend velocity_model. pour matcher named_parameters() du X0Model
            deltas[f"velocity_model.{base_key}.weight"] = delta
            del a_tensor, b_tensor

        if not deltas:
            log.warning("LoRA '%s': aucun delta compute (format inconnu ?)", name)
            return

        self._lora_deltas[name] = deltas
        log.info("LoRA '%s': %d parametres affectes", name, len(deltas))

    def _apply_lora_delta(self, param: torch.nn.Parameter, delta: torch.Tensor, add: bool) -> None:
        """Applique un delta LoRA in-place, en gerant le dtype (FP8 cast safe).

        Les deltas sont deja sur GPU (meme device que les params).
        """
        original_dtype = param.data.dtype
        if original_dtype != torch.bfloat16:
            param_bf16 = param.data.to(torch.bfloat16)
            if add:
                param_bf16.add_(delta)
            else:
                param_bf16.sub_(delta)
            param.data.copy_(param_bf16.to(original_dtype))
        else:
            if add:
                param.data.add_(delta)
            else:
                param.data.sub_(delta)

    def _fuse_lora(self, name: str) -> None:
        """Fuse un LoRA dans le transformer (add deltas in-place)."""
        deltas = self._lora_deltas.get(name)
        if not deltas:
            return
        transformer = self._get_orig_module()
        state_dict = dict(transformer.named_parameters())
        applied = 0
        for param_name, delta in deltas.items():
            if param_name in state_dict:
                self._apply_lora_delta(state_dict[param_name], delta, add=True)
                applied += 1
            elif log.isEnabledFor(logging.DEBUG):
                log.debug("LoRA fuse skip: %s (pas dans le transformer)", param_name)
        log.info("LoRA '%s' fused: %d/%d params", name, applied, len(deltas))
        if applied == 0 and deltas:
            # Log les cles attendues vs disponibles pour diagnostic
            expected = list(deltas.keys())[:3]
            available = list(state_dict.keys())[:3]
            log.warning(
                "LoRA '%s': AUCUN parametre fuse — cles LoRA: %s, transformer: %s",
                name, expected, available,
            )
        self._active_lora = name

    def _unfuse_lora(self, name: str) -> None:
        """Retire un LoRA du transformer (subtract deltas in-place)."""
        deltas = self._lora_deltas.get(name)
        if not deltas:
            return
        transformer = self._get_orig_module()
        state_dict = dict(transformer.named_parameters())
        applied = 0
        for param_name, delta in deltas.items():
            if param_name in state_dict:
                self._apply_lora_delta(state_dict[param_name], delta, add=False)
                applied += 1
        log.info("LoRA '%s' unfused: %d/%d params", name, applied, len(deltas))
        self._active_lora = None

    def ensure_lora(self, camera_motion: str | None) -> None:
        """Active le LoRA correspondant au camera_motion (ou aucun).

        No-op si le LoRA demande est deja fuse (~90% des cas avec dolly-in).
        """
        if not self._camera_lora_enabled:
            return
        needed = camera_motion if camera_motion in self._lora_deltas else None
        if needed == self._active_lora:
            return
        if self._active_lora is not None:
            self._unfuse_lora(self._active_lora)
        if needed is not None:
            self._fuse_lora(needed)

    # --- Depth estimation (DA3) ---

    def load_depth_model(self) -> None:
        """Charge DA3-LARGE-1.1 sur GPU (~0.4GB VRAM)."""
        if not self._depth_lora_enabled:
            return
        da3_path = os.path.join(self.models_dir, "da3-large")
        if not os.path.isdir(da3_path):
            log.warning("DA3 model introuvable: %s — depth disabled", da3_path)
            self._depth_lora_enabled = False
            return

        from depth_anything_3.api import DepthAnything3

        t0 = time.perf_counter()
        self._depth_model = DepthAnything3.from_pretrained(da3_path)
        self._depth_model = self._depth_model.to(device=self.device)
        dt = time.perf_counter() - t0
        log.info("DA3-LARGE-1.1 charge en %.1fs (VRAM ~0.4GB)", dt)

    def estimate_depth(self, image_path: str, target_h: int, target_w: int) -> str:
        """Genere une depth map depuis l'image, retourne le path PNG temporaire.

        La depth map est redimensionnee a 0.5x resolution (ref0.5 du IC-LoRA)
        et convertie en RGB grayscale (darker = closer, lighter = farther).
        """
        prediction = self._depth_model.inference([image_path])
        depth = prediction.depth[0]  # [H, W] float32

        # Normaliser en grayscale [0, 255]
        if isinstance(depth, torch.Tensor):
            depth_np = depth.cpu().numpy()
        else:
            depth_np = np.asarray(depth)
        d_min, d_max = depth_np.min(), depth_np.max()
        if d_max - d_min > 1e-6:
            depth_norm = ((depth_np - d_min) / (d_max - d_min) * 255).astype(np.uint8)
        else:
            depth_norm = np.zeros_like(depth_np, dtype=np.uint8)
        depth_img = Image.fromarray(depth_norm, mode="L")

        # Resize a 0.5x resolution (ref0.5)
        guide_h, guide_w = target_h // 2, target_w // 2
        # Assurer multiples de 32 pour le VAE
        guide_h = (guide_h // 32) * 32
        guide_w = (guide_w // 32) * 32
        depth_img = depth_img.resize((guide_w, guide_h), Image.BILINEAR)

        # Convertir en RGB (le VAE attend du RGB)
        depth_rgb = Image.merge("RGB", [depth_img, depth_img, depth_img])

        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False, prefix="depth_")
        depth_rgb.save(tmp.name)
        log.debug("Depth map: %s → %dx%d", tmp.name, guide_w, guide_h)
        return tmp.name

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
        Compatible torch.compile (SDPA natif, FlashAttention2 sur H100).
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

        # --- Depth IC-LoRA : enregistrer dans le registry et fuser par defaut ---
        if self._depth_lora_enabled:
            self._camera_lora_registry[self._depth_lora_name] = self._depth_lora_file
            # Sauvegarder le strength original camera, utiliser celui du depth
            orig_strength = self._camera_lora_strength
            self._camera_lora_strength = self._depth_lora_strength
            if self._depth_lora_name not in self._lora_deltas:
                self._load_lora_deltas(self._depth_lora_name)
            self._camera_lora_strength = orig_strength

        # --- Camera LoRA : charger deltas et fuser le defaut ---
        if self._camera_lora_enabled:
            for name in self._camera_lora_registry:
                if name not in self._lora_deltas:
                    self._load_lora_deltas(name)

        # Fuser le LoRA par defaut (depth si active, sinon dolly-in)
        if self._depth_lora_enabled and self._depth_lora_name in self._lora_deltas:
            default_lora = self._depth_lora_name
        else:
            default_lora = "dolly-in"
        if default_lora in self._lora_deltas:
            self._fuse_lora(default_lora)

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
            blocks = self._transformer.velocity_model.transformer_blocks
            log.info("torch.compile regional: %d blocs (mode=%s, dynamic=%s)...", len(blocks), compile_mode, dynamic_compile)
            for i, block in enumerate(blocks):
                blocks[i] = torch.compile(
                    block,
                    mode=compile_mode,
                    fullgraph=False,
                    dynamic=dynamic_compile,
                )

        self._log_vram("apres base transformer")
        return self._transformer

    def encode_prompt(self, prompt: str, negative: str | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode un prompt (cache hit ou Gemma GPU on-demand).

        Lookup 3-step :
        1. Preset connu (cache RAM par nom de preset)
        2. Custom en RAM (cache par hash du prompt)
        3. Custom sur volume (fichier prompt_{hash}.pt)
        4. Sinon Gemma GPU → encode → cache RAM + volume → cleanup VRAM
        """
        # 1. Chercher dans les presets (par valeur puis par key)
        for key, preset_text in CAMERA_PRESETS.items():
            if prompt == preset_text and key in self._embeddings_cache:
                return self._embeddings_cache[key]
        if prompt in self._embeddings_cache:
            return self._embeddings_cache[prompt]

        # 2. Check RAM cache (custom, par hash)
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
                torch.compiler.cudagraph_mark_step_begin()
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
                def denoise_step_s2(
                    video_state: LatentState,
                    audio_state: LatentState,
                    sigmas: torch.Tensor,
                    step_index: int,
                ) -> tuple[torch.Tensor, torch.Tensor]:
                    sigma = sigmas[step_index]
                    vid_mod = modality_from_latent_state(
                        video_state, v_context_p, sigma, enabled=True,
                    )
                    aud_mod = modality_from_latent_state(
                        audio_state, a_context_p, sigma, enabled=False,
                    )
                    torch.compiler.cudagraph_mark_step_begin()
                    return transformer(video=vid_mod, audio=aud_mod, perturbations=None)

                return euler_denoising_loop(
                    sigmas=sigmas,
                    video_state=video_state,
                    audio_state=audio_state,
                    stepper=stepper_arg,
                    denoise_fn=denoise_step_s2,
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
        encode_video_fast(video=iter(frames), fps=int(frame_rate), output_path=output_path)
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
        on_item_decoded: Callable[[int, list[torch.Tensor]], None] | None = None,
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

                # Depth guide : ajouter la depth map comme conditioning supplementaire
                # frame_idx=1 sauf si last_image occupe deja cet index (num_frames < 17)
                depth_path = item.get("depth_map_path")
                if depth_path and self._depth_lora_enabled:
                    last_idx_used = (num_frames - 1) // 8 if last_img else -1
                    depth_idx = 2 if last_idx_used == 1 else 1
                    images_i.append(ImageConditioningInput(
                        path=depth_path, frame_idx=depth_idx,
                        strength=self._depth_lora_strength,
                    ))

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
                torch.compiler.cudagraph_mark_step_begin()
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
                    # Log Inductor cache counters apres premier step
                    try:
                        from torch._dynamo.utils import counters
                        cache_counters = {k: v for k, v in counters["inductor"].items()
                                          if "cache" in k.lower() or "autograd" in k.lower()}
                        if cache_counters:
                            log.info("Inductor cache counters: %s", cache_counters)
                        else:
                            log.info("Inductor cache counters: (empty — no cache activity reported)")
                    except Exception as e:
                        log.debug("Inductor cache counters unavailable: %s", e)
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
                    vid_mod = modality_from_latent_state(video_state, v_ctx_batch, sigma, enabled=True)
                    aud_mod = modality_from_latent_state(audio_state, a_ctx_batch, sigma, enabled=False)
                    # Fix: expand scalar sigma to (batch_size,) for prompt_adaln
                    sigma_b = sigma.unsqueeze(0).expand(batch_size)
                    vid_mod = dataclasses.replace(vid_mod, sigma=sigma_b)
                    aud_mod = dataclasses.replace(aud_mod, sigma=sigma_b)
                    torch.compiler.cudagraph_mark_step_begin()
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
            t0 = time.perf_counter()
            item_latent = video_state.latent[i:i+1].contiguous()
            decoded = vae_decode_video(item_latent, self._video_decoder, tiling, generators[i])
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
