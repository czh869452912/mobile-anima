from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from anima_host.config import DEFAULT_MAX_TOKENS, LATENT_CHANNELS, SUPPORTED_RESOLUTIONS


@dataclass(frozen=True)
class CalibrationSpec:
    output_path: Path
    num_samples: int = 100
    resolutions: list[tuple[int, int]] = None  # type: ignore[assignment]
    max_tokens: int = DEFAULT_MAX_TOKENS
    seed: int = 42

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "resolutions",
            self.resolutions or list(SUPPORTED_RESOLUTIONS),
        )


class CalibrationDataGenerator:
    """Generate synthetic calibration data for PTQ of Anima submodels.

    Produces a .npz file containing representative input tensors for the
    denoiser.  When real model weights are available, the generator can
    optionally run the reference pipeline to collect more realistic
    intermediate activations.
    """

    def __init__(
        self,
        spec: CalibrationSpec,
        bundle: Any | None = None,
    ) -> None:
        self.spec = spec
        self.bundle = bundle
        self._rng = np.random.default_rng(spec.seed)

    def generate(self) -> Path:
        self.spec.output_path.parent.mkdir(parents=True, exist_ok=True)

        if self.bundle is not None and getattr(self.bundle, "available", False):
            return self._generate_from_pipeline()
        return self._generate_synthetic()

    def _generate_synthetic(self) -> Path:
        """Generate purely synthetic calibration data."""
        num_samples = self.spec.num_samples
        max_tokens = self.spec.max_tokens

        # Use the default resolution for shape reference
        width, height = self.spec.resolutions[0]
        latent_h = height // 8
        latent_w = width // 8

        latents = self._rng.standard_normal(
            (num_samples, LATENT_CHANNELS, latent_h, latent_w), dtype=np.float32
        ) * 0.5
        timesteps = self._rng.integers(0, 1000, size=(num_samples, 1)).astype(np.float32)
        conds = self._rng.standard_normal(
            (num_samples, max_tokens, 16), dtype=np.float32
        ) * 0.3
        unconds = self._rng.standard_normal(
            (num_samples, max_tokens, 16), dtype=np.float32
        ) * 0.3

        np.savez(
            self.spec.output_path,
            latent=latents,
            timestep=timesteps,
            cond=conds,
            uncond=unconds,
        )

        print(f"[calibration] Synthetic calibration data saved to {self.spec.output_path}")
        print(f"[calibration]   samples: {num_samples}")
        print(f"[calibration]   resolution: {width}x{height}")
        print(f"[calibration]   latent shape: ({num_samples}, {LATENT_CHANNELS}, {latent_h}, {latent_w})")
        return self.spec.output_path

    def _generate_from_pipeline(self) -> Path:
        """Collect calibration data by running the reference pipeline.

        This is a placeholder for when real model weights are available.
        The implementation would run the reference pipeline for a small
        number of steps and capture intermediate denoiser inputs.
        """
        print("[calibration] Real pipeline calibration not yet implemented; falling back to synthetic")
        return self._generate_synthetic()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate calibration data for Anima PTQ")
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    spec = CalibrationSpec(
        output_path=args.output_path,
        num_samples=args.num_samples,
        seed=args.seed,
    )
    generator = CalibrationDataGenerator(spec)
    generator.generate()


if __name__ == "__main__":
    main()
