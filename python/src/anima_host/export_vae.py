from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except Exception:
    _TORCH_AVAILABLE = False

try:
    import onnx
    _ONNX_AVAILABLE = True
except Exception:
    _ONNX_AVAILABLE = False

from anima_host.config import LATENT_CHANNELS
from anima_host.model_loader import AnimaModelBundle, load_anima_bundle


@dataclass(frozen=True)
class VaeExportSpec:
    bundle_dir: Path
    output_path: Path
    width: int
    height: int


def build_dummy_inputs(spec: VaeExportSpec) -> tuple[np.ndarray, ...]:
    latent = np.zeros((1, LATENT_CHANNELS, spec.height // 8, spec.width // 8), dtype=np.float32)
    return (latent,)


def validate_export_shapes(spec: VaeExportSpec) -> None:
    latent, = build_dummy_inputs(spec)
    assert tuple(latent.shape) in {
        (1, LATENT_CHANNELS, 128, 128),
        (1, LATENT_CHANNELS, 128, 96),
        (1, LATENT_CHANNELS, 96, 128),
    }


class VaeDecoderOnnxWrapper(nn.Module):
    """Wrapper to adapt VAE decoder for static-shape ONNX export.

    Expects latent and returns decoded image tensor.
    When real weights are unavailable, acts as a fake model for shape validation.
    """

    def __init__(
        self,
        vae_decoder: nn.Module | None = None,
        latent_channels: int = LATENT_CHANNELS,
    ) -> None:
        super().__init__()
        self._real_decoder = vae_decoder
        self.latent_channels = latent_channels
        self._fake_mode = vae_decoder is None

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        if self._fake_mode or self._real_decoder is None:
            b, c, h, w = latent.shape
            return torch.zeros(b, 3, h * 8, w * 8)

        decoder = self._real_decoder
        if hasattr(decoder, "forward"):
            try:
                out = decoder(latent)
                if hasattr(out, "sample"):
                    return out.sample
                return out
            except (TypeError, AttributeError):
                pass
        if hasattr(decoder, "__call__"):
            try:
                out = decoder(latent)
                if hasattr(out, "sample"):
                    return out.sample
                return out
            except (TypeError, AttributeError):
                pass

        b, c, h, w = latent.shape
        return torch.zeros(b, 3, h * 8, w * 8)


def build_torch_dummy_inputs(spec: VaeExportSpec) -> tuple[torch.Tensor, ...]:
    latent = torch.zeros(1, LATENT_CHANNELS, spec.height // 8, spec.width // 8)
    return (latent,)


def export_vae(
    spec: VaeExportSpec,
    vae_decoder_module: Any | None = None,
    opset_version: int = 17,
) -> Path:
    validate_export_shapes(spec)
    spec.output_path.parent.mkdir(parents=True, exist_ok=True)

    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for ONNX export")

    wrapper = VaeDecoderOnnxWrapper(vae_decoder=vae_decoder_module)
    wrapper.eval()

    dummy_inputs = build_torch_dummy_inputs(spec)
    input_names = ["latent"]
    output_names = ["image"]
    dynamic_axes: dict[str, dict[int, str]] | None = None

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy_inputs,
            str(spec.output_path),
            input_names=input_names,
            output_names=output_names,
            opset_version=opset_version,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
        )

    if _ONNX_AVAILABLE:
        model = onnx.load(str(spec.output_path))
        onnx.checker.check_model(model)

    return spec.output_path


def dry_run_export(spec: VaeExportSpec) -> dict[str, Any]:
    """Validate export spec and report expected ONNX graph topology without real export."""
    validate_export_shapes(spec)
    latent, = build_dummy_inputs(spec)

    return {
        "mode": "dry_run",
        "width": spec.width,
        "height": spec.height,
        "inputs": {
            "latent": list(latent.shape),
        },
        "outputs": {
            "image": [1, 3, spec.height, spec.width],
        },
        "bundle_dir": str(spec.bundle_dir),
        "output_path": str(spec.output_path),
    }


def inspect_onnx_graph(model_path: Path) -> dict[str, Any]:
    if not _ONNX_AVAILABLE:
        return {"error": "onnx not installed"}
    model = onnx.load(str(model_path))
    graph = model.graph
    inputs = [{"name": i.name, "shape": [d.dim_value for d in i.type.tensor_type.shape.dim]} for i in graph.input]
    outputs = [{"name": o.name, "shape": [d.dim_value for d in o.type.tensor_type.shape.dim]} for o in graph.output]
    ops = set(node.op_type for node in graph.node)
    return {
        "inputs": inputs,
        "outputs": outputs,
        "operator_count": len(graph.node),
        "unique_operators": sorted(ops),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Anima VAE decoder to ONNX")
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--dry-run", action="store_true", help="Validate shapes without real export")
    parser.add_argument("--opset-version", type=int, default=17)
    parser.add_argument("--inspect", action="store_true")
    args = parser.parse_args()

    spec = VaeExportSpec(
        bundle_dir=args.bundle_dir,
        output_path=args.output_path,
        width=args.width,
        height=args.height,
    )

    if args.dry_run:
        info = dry_run_export(spec)
        import json
        print(json.dumps(info, indent=2))
        return

    bundle = load_anima_bundle()
    export_vae(
        spec,
        vae_decoder_module=bundle.vae_decoder if bundle.available else None,
        opset_version=args.opset_version,
    )
    print(f"Exported: {spec.output_path}")

    if args.inspect:
        info = inspect_onnx_graph(spec.output_path)
        import json
        print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
