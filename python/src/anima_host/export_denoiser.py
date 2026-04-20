import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ExportSpec:
    bundle_dir: Path
    output_path: Path
    width: int
    height: int
    max_tokens: int


def build_dummy_inputs(spec: ExportSpec):
    latent = np.zeros((1, 4, spec.height // 8, spec.width // 8), dtype=np.float32)
    timestep = np.zeros((1,), dtype=np.float32)
    cond = np.zeros((1, spec.max_tokens, 16), dtype=np.float32)
    uncond = np.zeros((1, spec.max_tokens, 16), dtype=np.float32)
    return latent, timestep, cond, uncond


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()
    spec = ExportSpec(
        bundle_dir=args.bundle_dir,
        output_path=args.output_path,
        width=args.width,
        height=args.height,
        max_tokens=args.max_tokens,
    )
    latent, timestep, cond, uncond = build_dummy_inputs(spec)
    print(
        {
            "latent": tuple(latent.shape),
            "timestep": tuple(timestep.shape),
            "cond": tuple(cond.shape),
            "uncond": tuple(uncond.shape),
        }
    )


if __name__ == "__main__":
    main()
