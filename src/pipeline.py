"""
MedusaPipeline — ltx-pipelines direct inference for LTX-2 19B I2V.

Remplace ComfyUI par un appel Python direct a ltx-core / ltx-pipelines.
Gestion du lifecycle des modeles entre jobs :
  - Video encoder  : persistent en VRAM (~1GB)
  - Video decoder  : persistent en VRAM (~2GB)
  - Transformer    : base (distilled + I2V) persistante, camera LoRA fuse/unfuse in-place
  - Text encoder   : charge sur CPU au warmup, embeddings caches sur disque
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator

import torch
from safetensors.torch import load_file as load_safetensors

from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.guiders import MultiModalGuider, MultiModalGuiderParams
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.protocols import DiffusionStepProtocol
from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_core.model.video_vae import decode_video as vae_decode_video

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
from prompts import CAMERA_PROMPTS, DEFAULT_NEGATIVE_PROMPT

log = logging.getLogger("medusa")

# LoRA strengths (matches current ComfyUI workflow)
DISTILLED_LORA_STRENGTH = 0.7
I2V_ADAPTER_STRENGTH = 0.8
CAMERA_LORA_STRENGTH = 1.0

# Audio skip : audio calcule seulement au step 0 / 8
AUDIO_SKIP_STEP = 99


class MedusaPipeline:
    """Pipeline I2V utilisant ltx-pipelines avec audio skip et transformer cache."""

    def __init__(self, models_dir: str, device: torch.device | None = None) -> None:
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.dtype = torch.bfloat16
        self.models_dir = models_dir

        # Paths modeles
        self._checkpoint_path = os.path.join(models_dir, "checkpoints", "ltx-2-19b-dev.safetensors")
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
        )

        # Video encoder (persistent en VRAM)
        self._video_encoder: torch.nn.Module | None = None

        # Video decoder (persistent en VRAM)
        self._video_decoder: torch.nn.Module | None = None

        # Transformer base en VRAM (distilled + I2V fusionnes, camera LoRA fuse/unfuse dynamiquement)
        self._transformer: torch.nn.Module | None = None
        self._current_camera_lora: str | None = None

        # Cache des deltas camera LoRA precalcules (path -> {param_name: delta_tensor CPU})
        self._camera_deltas: dict[str, dict[str, torch.Tensor]] = {}

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
        self._log_vram("apres video encoder")

    def load_video_decoder(self) -> None:
        """Charge le video decoder en VRAM (~2GB). Persistent entre jobs."""
        log.info("Chargement video decoder (persistent)...")
        self._video_decoder = self._base_ledger.video_decoder()
        self._log_vram("apres video decoder")

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

    def _get_orig_module(self) -> torch.nn.Module:
        """Unwrap torch.compile OptimizedModule si besoin."""
        mod = self._transformer
        if hasattr(mod, "_orig_mod"):
            return mod._orig_mod
        return mod

    def _compute_camera_delta(self, camera_lora_path: str) -> dict[str, torch.Tensor]:
        """Charge un camera LoRA et calcule les deltas poids (caches sur CPU).

        Formule : delta = strength * alpha/rank * (lora_up @ lora_down)
        Les deltas sont caches pour reutilisation lors des switch cameras.
        """
        if camera_lora_path in self._camera_deltas:
            return self._camera_deltas[camera_lora_path]

        log.info("Calcul delta camera LoRA: %s", os.path.basename(camera_lora_path))
        raw = load_safetensors(camera_lora_path, device="cpu")

        # Grouper les paires lora_up/lora_down par cle de base
        pairs: dict[str, dict[str, torch.Tensor]] = {}
        for key, tensor in raw.items():
            # LTXV_LORA_COMFY_RENAMING_MAP : strip "diffusion_model." prefix
            clean = key.replace("diffusion_model.", "", 1) if key.startswith("diffusion_model.") else key

            if ".lora_down.weight" in clean:
                base = clean.replace(".lora_down.weight", "")
                pairs.setdefault(base, {})["down"] = tensor
            elif ".lora_up.weight" in clean:
                base = clean.replace(".lora_up.weight", "")
                pairs.setdefault(base, {})["up"] = tensor
            elif ".alpha" in clean:
                base = clean.replace(".alpha", "")
                pairs.setdefault(base, {})["alpha"] = tensor

        deltas: dict[str, torch.Tensor] = {}
        for base_key, pair in pairs.items():
            if "down" not in pair or "up" not in pair:
                continue
            down = pair["down"].to(dtype=self.dtype, device=self.device)
            up = pair["up"].to(dtype=self.dtype, device=self.device)

            delta = up @ down

            if "alpha" in pair:
                rank = down.shape[0]
                alpha_val = pair["alpha"].item()
                delta = delta * (alpha_val / rank)

            delta = delta * CAMERA_LORA_STRENGTH
            deltas[f"{base_key}.weight"] = delta.cpu()
            del down, up, delta

        self._camera_deltas[camera_lora_path] = deltas
        log.info("Delta camera: %d params calcules", len(deltas))
        return deltas

    def _apply_camera_delta(self, camera_lora_path: str, sign: float = 1.0) -> None:
        """Fuse (sign=1) ou unfuse (sign=-1) un camera LoRA delta in-place."""
        deltas = self._compute_camera_delta(camera_lora_path)
        module = self._get_orig_module()
        params = dict(module.named_parameters())
        applied = 0
        for key, delta_cpu in deltas.items():
            if key in params:
                params[key].data.add_(delta_cpu.to(self.device), alpha=sign)
                applied += 1
        action = "Fuse" if sign > 0 else "Unfuse"
        log.info("%s camera LoRA: %s (%d/%d params)", action, os.path.basename(camera_lora_path), applied, len(deltas))

    def get_transformer(self, camera_lora_path: str) -> torch.nn.Module:
        """Retourne le transformer avec la camera LoRA demandee.

        Base (distilled + I2V) construite une seule fois via ModelLedger.
        Camera LoRA fuse/unfuse dynamiquement in-place (~0.1s de switch).
        Compatible torch.compile car modification in-place (memes adresses memoire).
        """
        # Meme camera → retourner directement
        if self._current_camera_lora == camera_lora_path:
            return self._transformer

        # Premier appel → build base transformer (distilled + I2V seulement)
        if self._transformer is None:
            log.info("Build base transformer (distilled + I2V)...")
            ledger = ModelLedger(
                dtype=self.dtype,
                device=self.device,
                checkpoint_path=self._checkpoint_path,
                loras=self._base_loras,
            )
            self._transformer = ledger.transformer()
            del ledger

            # torch.compile pour acceleration inference (8 steps/job)
            if os.environ.get("TORCH_COMPILE", "1") == "1":
                torch._dynamo.config.allow_unspec_int_on_nn_module = True
                log.info("torch.compile transformer (mode=reduce-overhead)...")
                self._transformer = torch.compile(
                    self._transformer,
                    mode="reduce-overhead",
                    fullgraph=False,
                )

            self._log_vram("apres base transformer")
        else:
            # Switch camera → unfuse l'ancienne
            self._apply_camera_delta(self._current_camera_lora, sign=-1.0)

        # Fuse la nouvelle camera LoRA
        self._apply_camera_delta(camera_lora_path, sign=1.0)
        self._current_camera_lora = camera_lora_path
        self._log_vram(f"apres camera {os.path.basename(camera_lora_path)}")
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
        last_image_path: str | None = None,
        last_image_strength: float = 1.0,
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
            last_image_path: Chemin vers l'image last frame (optionnel).
            last_image_strength: Force du conditioning last image (0-1).
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
        self._log_vram("debut generate")
        transformer = self.get_transformer(camera_lora_path)

        # 3. Setup denoising
        noiser = GaussianNoiser(generator=generator)
        stepper = EulerDiffusionStep()

        # 4. Image conditioning
        images = [(image_path, 0, image_strength)]
        if last_image_path is not None:
            last_latent_idx = (num_frames - 1) // 8
            images.append((last_image_path, last_latent_idx, last_image_strength))

        conditionings = image_conditionings_by_replacing_latent(
            images=images,
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

        # 7. VAE decode (sans tiling — H100 80GB)
        log.info("VAE decode...")
        decoded_video: Iterator[torch.Tensor] = vae_decode_video(
            video_state.latent,
            self._video_decoder,
            None,
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
