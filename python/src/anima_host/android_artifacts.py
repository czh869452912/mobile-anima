from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from anima_host.config import SUPPORTED_RESOLUTIONS, DEFAULT_MAX_TOKENS


@dataclass(frozen=True)
class ArtifactSpec:
    build_dir: Path
    android_asset_dir: Path
    model_id: str = "circlestone-labs/Anima"
    resolutions: list[tuple[int, int]] = field(default_factory=lambda: list(SUPPORTED_RESOLUTIONS))
    max_tokens: int = DEFAULT_MAX_TOKENS


@dataclass(frozen=True)
class ArtifactReport:
    assets_dir: Path
    metadata_path: Path
    files_copied: list[str]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "assets_dir": str(self.assets_dir),
            "metadata_path": str(self.metadata_path),
            "files_copied": self.files_copied,
            "metadata": self.metadata,
        }


class AndroidArtifactPackager:
    """Package ONNX models and metadata into Android asset directory.

    Produces a directory structure consumable by the Android
    ``ArtifactManager``:

        android_asset_dir/
            models/
                denoiser.onnx
                denoiser_qdq.onnx
                text_encoder.onnx
                vae.onnx
            metadata.json
    """

    def __init__(self, spec: ArtifactSpec) -> None:
        self.spec = spec

    def package(self) -> ArtifactReport:
        build_dir = self.spec.build_dir
        assets_dir = self.spec.android_asset_dir / "models"
        assets_dir.mkdir(parents=True, exist_ok=True)

        files_copied: list[str] = []
        expected_files = [
            "denoiser.onnx",
            "denoiser_qdq.onnx",
            "text_encoder.onnx",
            "vae.onnx",
        ]

        for name in expected_files:
            src = build_dir / name
            if src.exists():
                dst = assets_dir / name
                import shutil
                shutil.copy2(str(src), str(dst))
                files_copied.append(name)

        metadata = {
            "model_id": self.spec.model_id,
            "resolutions": [f"{w}x{h}" for w, h in self.spec.resolutions],
            "max_tokens": self.spec.max_tokens,
            "files": {
                "denoiser": "models/denoiser.onnx",
                "denoiser_qdq": "models/denoiser_qdq.onnx",
                "text_encoder": "models/text_encoder.onnx",
                "vae": "models/vae.onnx",
            },
            "input_shapes": {
                "denoiser": {
                    "latent": [1, 4, 128, 128],
                    "timestep": [1],
                    "cond": [1, self.spec.max_tokens, 16],
                    "uncond": [1, self.spec.max_tokens, 16],
                },
                "text_encoder": {
                    "input_ids": [1, self.spec.max_tokens],
                },
                "vae": {
                    "latent": [1, 4, 128, 128],
                },
            },
            "output_shapes": {
                "denoiser": {
                    "noise_pred": [1, 4, 128, 128],
                },
                "text_encoder": {
                    "text_embeddings": [1, self.spec.max_tokens, 16],
                },
                "vae": {
                    "image": [1, 3, 1024, 1024],
                },
            },
        }

        metadata_path = self.spec.android_asset_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return ArtifactReport(
            assets_dir=assets_dir,
            metadata_path=metadata_path,
            files_copied=files_copied,
            metadata=metadata,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Package Anima ONNX artifacts for Android")
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--android-assets", type=Path, required=True)
    parser.add_argument("--model-id", default="circlestone-labs/Anima")
    args = parser.parse_args()

    spec = ArtifactSpec(
        build_dir=args.build_dir,
        android_asset_dir=args.android_assets,
        model_id=args.model_id,
    )
    packager = AndroidArtifactPackager(spec)
    report = packager.package()

    print(f"Packaged {len(report.files_copied)} files to {report.assets_dir}")
    print(f"Metadata written to {report.metadata_path}")


if __name__ == "__main__":
    main()
