from dataclasses import dataclass

import numpy as np
from PIL import Image

from anima_host.config import GenerationConfig
from anima_host.interfaces import Denoiser, TextEncoder, VaeDecoder


@dataclass(frozen=True)
class GenerationArtifacts:
    initial_latent_shape: tuple[int, int, int, int]
    cond_shape: tuple[int, ...]
    uncond_shape: tuple[int, ...]
    step_count: int
    final_latent_mean: float


class ReferencePipeline:
    def __init__(
        self,
        text_encoder: TextEncoder,
        denoiser: Denoiser,
        vae_decoder: VaeDecoder,
    ) -> None:
        self.text_encoder = text_encoder
        self.denoiser = denoiser
        self.vae_decoder = vae_decoder

    def build_initial_latent(self, config: GenerationConfig) -> np.ndarray:
        return np.zeros(config.latent_shape, dtype=np.float32)

    def generate(
        self,
        prompt: str,
        negative_prompt: str,
        config: GenerationConfig,
    ) -> tuple[Image.Image, GenerationArtifacts]:
        cond, uncond = self.text_encoder.encode(
            prompt=prompt,
            negative_prompt=negative_prompt,
            max_tokens=config.max_tokens,
        )
        latent = self.build_initial_latent(config)
        initial_shape = latent.shape

        for timestep in range(config.steps):
            latent = self.denoiser.step(
                latent=latent,
                timestep=timestep,
                cond=cond,
                uncond=uncond,
                cfg=config.cfg,
            )

        image = self.vae_decoder.decode(latent)
        artifacts = GenerationArtifacts(
            initial_latent_shape=initial_shape,
            cond_shape=cond.shape,
            uncond_shape=uncond.shape,
            step_count=config.steps,
            final_latent_mean=float(latent.mean()),
        )
        return image, artifacts
