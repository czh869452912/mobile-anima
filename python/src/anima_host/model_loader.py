from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from anima_host.config import (
    ANIMA_DIFFUSION_DIM,
    ANIMA_NUM_HEADS,
    ANIMA_NUM_LAYERS,
    DEFAULT_MAX_TOKENS,
    LATENT_CHANNELS,
    ModelConfig,
)


def _has_diffsynth() -> bool:
    return importlib.util.find_spec("diffsynth") is not None


def _has_transformers() -> bool:
    return importlib.util.find_spec("transformers") is not None


def _has_torch() -> bool:
    return importlib.util.find_spec("torch") is not None


@dataclass(frozen=True)
class ArchitectureSummary:
    backbone: str
    params_billion: float
    num_layers: int
    hidden_dim: int
    num_heads: int
    latent_channels: int
    text_encoder_name: str
    vae_name: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "backbone": self.backbone,
            "params_billion": self.params_billion,
            "num_layers": self.num_layers,
            "hidden_dim": self.hidden_dim,
            "num_heads": self.num_heads,
            "latent_channels": self.latent_channels,
            "text_encoder_name": self.text_encoder_name,
            "vae_name": self.vae_name,
        }


class _DummyTextEncoder:
    def __init__(self) -> None:
        pass

    def __call__(self, input_ids: Any) -> Any:
        batch_size = input_ids.shape[0] if hasattr(input_ids, "shape") else 1
        seq_len = input_ids.shape[1] if hasattr(input_ids, "shape") else DEFAULT_MAX_TOKENS
        return type("Out", (), {"last_hidden_state": np.zeros((batch_size, seq_len, 16), dtype=np.float32)})()


class _DummyDenoiser:
    def __init__(self) -> None:
        self._num_layers = ANIMA_NUM_LAYERS
        self._hidden_dim = ANIMA_DIFFUSION_DIM
        self._num_heads = ANIMA_NUM_HEADS

    def __call__(self, latent: Any, timestep: Any, cond: Any, uncond: Any) -> Any:
        if hasattr(latent, "shape"):
            return latent
        return np.zeros((1, LATENT_CHANNELS, 128, 128), dtype=np.float32)


class _DummyVaeDecoder:
    def __init__(self) -> None:
        pass

    def __call__(self, latent: Any) -> Any:
        if hasattr(latent, "shape"):
            b, c, h, w = latent.shape
            return type("Out", (), {"sample": np.zeros((b, 3, h * 8, w * 8), dtype=np.float32)})()
        return type("Out", (), {"sample": np.zeros((1, 3, 1024, 1024), dtype=np.float32)})()


class _DummyTokenizer:
    def __init__(self) -> None:
        self.model_max_length = DEFAULT_MAX_TOKENS

    def __call__(self, text: str | list[str], **kwargs: Any) -> dict[str, Any]:
        if isinstance(text, str):
            batch_size = 1
        else:
            batch_size = len(text)
        return {
            "input_ids": np.zeros((batch_size, DEFAULT_MAX_TOKENS), dtype=np.int64),
            "attention_mask": np.ones((batch_size, DEFAULT_MAX_TOKENS), dtype=np.int64),
        }


class _DummyScheduler:
    def __init__(self) -> None:
        self.num_train_timesteps = 1000

    def set_timesteps(self, num_inference_steps: int) -> None:
        pass

    def step(self, model_output: Any, timestep: int, sample: Any) -> Any:
        return type("Out", (), {"prev_sample": sample})()


class AnimaModelBundle:
    def __init__(self, config: ModelConfig | None = None) -> None:
        self.config = config or ModelConfig()
        self._text_encoder: Any = None
        self._denoiser: Any = None
        self._vae_decoder: Any = None
        self._tokenizer: Any = None
        self._scheduler: Any = None
        self._available: bool = False
        self._load_error: str | None = None

    @property
    def available(self) -> bool:
        return self._available

    @property
    def load_error(self) -> str | None:
        return self._load_error

    @property
    def text_encoder(self) -> Any:
        return self._text_encoder

    @property
    def denoiser(self) -> Any:
        return self._denoiser

    @property
    def vae_decoder(self) -> Any:
        return self._vae_decoder

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    @property
    def scheduler(self) -> Any:
        return self._scheduler

    def architecture_summary(self) -> ArchitectureSummary:
        return ArchitectureSummary(
            backbone="NVIDIA Cosmos-Predict2-2B-Text2Image (DiT)",
            params_billion=2.0,
            num_layers=ANIMA_NUM_LAYERS,
            hidden_dim=ANIMA_DIFFUSION_DIM,
            num_heads=ANIMA_NUM_HEADS,
            latent_channels=LATENT_CHANNELS,
            text_encoder_name="Qwen3-0.6B",
            vae_name="Qwen-Image VAE",
        )

    def describe(self) -> str:
        arch = self.architecture_summary()
        lines = [
            f"Model: {self.config.model_id}",
            f"  Backbone: {arch.backbone}",
            f"  Parameters: {arch.params_billion}B",
            f"  Layers: {arch.num_layers}",
            f"  Hidden dim: {arch.hidden_dim}",
            f"  Attention heads: {arch.num_heads}",
            f"  Latent channels: {arch.latent_channels}",
            f"  Text encoder: {arch.text_encoder_name}",
            f"  VAE: {arch.vae_name}",
            f"  Weights available: {self._available}",
        ]
        if self._load_error:
            lines.append(f"  Load error: {self._load_error}")
        return "\n".join(lines)

    def load(self) -> None:
        try:
            self._try_load_real()
        except Exception as exc:
            self._load_error = f"{type(exc).__name__}: {exc}"
            self._install_fallbacks()

    def _try_load_real(self) -> None:
        if not _has_torch():
            raise RuntimeError("PyTorch not installed")
        if not _has_transformers():
            raise RuntimeError("transformers not installed")

        import torch

        cache_dir = self.config.local_cache_dir

        if _has_diffsynth() and cache_dir is not None:
            self._try_load_diffsynth(torch, cache_dir)
            return

        self._try_load_transformers(torch, cache_dir)

    def _try_load_diffsynth(self, torch: Any, cache_dir: Path | None) -> None:
        try:
            from diffsynth.pipelines.anima_image import AnimaImagePipeline, ModelConfig as DiffModelConfig

            model_configs = [
                DiffModelConfig(
                    model_id=self.config.model_id,
                    origin_file_pattern=self.config.submodel_paths["diffusion"],
                ),
                DiffModelConfig(
                    model_id=self.config.model_id,
                    origin_file_pattern=self.config.submodel_paths["text_encoder"],
                ),
                DiffModelConfig(
                    model_id=self.config.model_id,
                    origin_file_pattern=self.config.submodel_paths["vae"],
                ),
            ]
            tokenizer_config = DiffModelConfig(
                model_id=self.config.tokenizer_id or self.config.text_encoder_model_id,
                origin_file_pattern="./",
            )

            pipe = AnimaImagePipeline.from_pretrained(
                torch_dtype=getattr(torch, self.config.torch_dtype, torch.bfloat16),
                device=self.config.device,
                model_configs=model_configs,
                tokenizer_config=tokenizer_config,
                cache_dir=str(cache_dir) if cache_dir else None,
            )

            self._text_encoder = pipe.text_encoder
            self._denoiser = pipe.denoiser
            self._vae_decoder = pipe.vae_decoder
            self._tokenizer = pipe.tokenizer
            self._scheduler = pipe.scheduler
            self._available = True
        except Exception as exc:
            raise RuntimeError(f"DiffSynth loading failed: {exc}") from exc

    def _try_load_transformers(self, torch: Any, cache_dir: Path | None) -> None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.text_encoder_model_id,
            cache_dir=str(cache_dir) if cache_dir else None,
            trust_remote_code=True,
        )
        self._tokenizer = tokenizer

        self._text_encoder = _DummyTextEncoder()
        self._denoiser = _DummyDenoiser()
        self._vae_decoder = _DummyVaeDecoder()
        self._scheduler = _DummyScheduler()
        self._available = False
        self._load_error = "Full model load requires DiffSynth or local safetensors weights"

    def _install_fallbacks(self) -> None:
        self._text_encoder = _DummyTextEncoder()
        self._denoiser = _DummyDenoiser()
        self._vae_decoder = _DummyVaeDecoder()
        self._tokenizer = _DummyTokenizer()
        self._scheduler = _DummyScheduler()
        self._available = False


def load_anima_bundle(config: ModelConfig | None = None) -> AnimaModelBundle:
    bundle = AnimaModelBundle(config)
    bundle.load()
    return bundle
