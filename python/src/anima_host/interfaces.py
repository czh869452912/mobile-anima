from typing import Protocol

import numpy as np
from PIL import Image


class TextEncoder(Protocol):
    def encode(
        self,
        prompt: str,
        negative_prompt: str,
        max_tokens: int,
    ) -> tuple[np.ndarray, np.ndarray]: ...


class Denoiser(Protocol):
    def step(
        self,
        latent: np.ndarray,
        timestep: int,
        cond: np.ndarray,
        uncond: np.ndarray,
        cfg: float,
    ) -> np.ndarray: ...


class VaeDecoder(Protocol):
    def decode(self, latent: np.ndarray) -> Image.Image: ...
