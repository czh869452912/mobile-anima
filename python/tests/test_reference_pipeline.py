import numpy as np
from PIL import Image

from anima_host.config import GenerationConfig
from anima_host.reference_pipeline import ReferencePipeline


class FakeTextEncoder:
    def __init__(self) -> None:
        self.calls = []

    def encode(self, prompt: str, negative_prompt: str, max_tokens: int):
        self.calls.append((prompt, negative_prompt, max_tokens))
        cond = np.ones((1, 256, 16), dtype=np.float32)
        uncond = np.zeros((1, 256, 16), dtype=np.float32)
        return cond, uncond


class FakeDenoiser:
    def __init__(self) -> None:
        self.timesteps = []

    def step(self, latent, timestep, cond, uncond, cfg):
        self.timesteps.append((timestep, cfg, cond.shape, uncond.shape))
        return latent + 1.0


class FakeVaeDecoder:
    def decode(self, latent):
        width = latent.shape[3] * 8
        height = latent.shape[2] * 8
        return Image.new("RGB", (width, height), color="black")


def test_reference_pipeline_runs_exact_number_of_steps():
    pipeline = ReferencePipeline(
        text_encoder=FakeTextEncoder(),
        denoiser=FakeDenoiser(),
        vae_decoder=FakeVaeDecoder(),
    )
    config = GenerationConfig(width=1024, height=1024, steps=8, cfg=5.0)

    image, artifacts = pipeline.generate(
        prompt="cat astronaut",
        negative_prompt="blurry",
        config=config,
    )

    assert image.size == (1024, 1024)
    assert artifacts.step_count == 8
    assert artifacts.final_latent_mean == 8.0


def test_reference_pipeline_uses_fixed_latent_shape():
    pipeline = ReferencePipeline(
        text_encoder=FakeTextEncoder(),
        denoiser=FakeDenoiser(),
        vae_decoder=FakeVaeDecoder(),
    )
    config = GenerationConfig(width=768, height=1024, steps=12, cfg=6.0)

    _, artifacts = pipeline.generate(
        prompt="forest temple",
        negative_prompt="low quality",
        config=config,
    )

    assert artifacts.initial_latent_shape == (1, 4, 128, 96)
    assert artifacts.cond_shape == (1, 256, 16)
