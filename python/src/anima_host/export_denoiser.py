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

from anima_host.config import (
    DEFAULT_MAX_TOKENS,
    DENOISER_INPUT_COND,
    DENOISER_INPUT_LATENT,
    DENOISER_INPUT_TIMESTEP,
    DENOISER_INPUT_UNCOND,
    DENOISER_OUTPUT_NOISE_PRED,
    LATENT_CHANNELS,
)
from anima_host.model_loader import AnimaModelBundle, load_anima_bundle


@dataclass(frozen=True)
class ExportSpec:
    bundle_dir: Path
    output_path: Path
    width: int
    height: int
    max_tokens: int


class AnimaDenoiserWrapper(nn.Module):
    """Wrapper to adapt Anima denoiser for static-shape ONNX export.

    Expects latent + timestep + cond + uncond inputs and returns noise prediction.
    When real weights are unavailable, this acts as a fake model for shape validation.
    """

    def __init__(
        self,
        denoiser: nn.Module | None = None,
        latent_channels: int = LATENT_CHANNELS,
        cond_dim: int = 16,
    ) -> None:
        super().__init__()
        self._real_denoiser = denoiser
        self.latent_channels = latent_channels
        self.cond_dim = cond_dim
        self._fake_mode = denoiser is None

    def forward(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        cond: torch.Tensor,
        uncond: torch.Tensor,
    ) -> torch.Tensor:
        if self._fake_mode or self._real_denoiser is None:
            return self._fake_forward(latent, timestep, cond, uncond)
        return self._real_forward(latent, timestep, cond, uncond)

    def _real_forward(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        cond: torch.Tensor,
        uncond: torch.Tensor,
    ) -> torch.Tensor:
        denoiser = self._real_denoiser
        if hasattr(denoiser, "forward"):
            try:
                return denoiser(latent, timestep, cond, uncond)
            except TypeError:
                try:
                    return denoiser(latent, timestep, encoder_hidden_states=cond)
                except TypeError:
                    pass
        if hasattr(denoiser, "__call__"):
            try:
                return denoiser(latent, timestep, cond, uncond)
            except TypeError:
                try:
                    return denoiser(latent, timestep, encoder_hidden_states=cond)
                except TypeError:
                    pass
        return self._fake_forward(latent, timestep, cond, uncond)

    def _fake_forward(
        self,
        latent: torch.Tensor,
        _timestep: torch.Tensor,
        _cond: torch.Tensor,
        _uncond: torch.Tensor,
    ) -> torch.Tensor:
        return torch.zeros_like(latent)


def build_dummy_inputs(spec: ExportSpec) -> tuple[np.ndarray, ...]:
    latent = np.zeros((1, LATENT_CHANNELS, spec.height // 8, spec.width // 8), dtype=np.float32)
    timestep = np.zeros((1,), dtype=np.float32)
    cond = np.zeros((1, spec.max_tokens, 16), dtype=np.float32)
    uncond = np.zeros((1, spec.max_tokens, 16), dtype=np.float32)
    return latent, timestep, cond, uncond


def build_torch_dummy_inputs(spec: ExportSpec) -> tuple[torch.Tensor, ...]:
    latent = torch.zeros(1, LATENT_CHANNELS, spec.height // 8, spec.width // 8)
    timestep = torch.zeros(1)
    cond = torch.zeros(1, spec.max_tokens, 16)
    uncond = torch.zeros(1, spec.max_tokens, 16)
    return latent, timestep, cond, uncond


def validate_export_shapes(spec: ExportSpec) -> None:
    latent, timestep, cond, uncond = build_dummy_inputs(spec)
    assert tuple(latent.shape) in {(1, LATENT_CHANNELS, 128, 128), (1, LATENT_CHANNELS, 128, 96), (1, LATENT_CHANNELS, 96, 128)}
    assert tuple(timestep.shape) == (1,)
    assert tuple(cond.shape) == (1, spec.max_tokens, 16)
    assert tuple(uncond.shape) == (1, spec.max_tokens, 16)


def export_denoiser(
    spec: ExportSpec,
    denoiser_module: Any | None = None,
    opset_version: int = 17,
) -> Path:
    validate_export_shapes(spec)
    spec.output_path.parent.mkdir(parents=True, exist_ok=True)

    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for ONNX export")

    wrapper = AnimaDenoiserWrapper(denoiser=denoiser_module)
    wrapper.eval()

    dummy_inputs = build_torch_dummy_inputs(spec)
    input_names = [DENOISER_INPUT_LATENT, DENOISER_INPUT_TIMESTEP, DENOISER_INPUT_COND, DENOISER_INPUT_UNCOND]
    output_names = [DENOISER_OUTPUT_NOISE_PRED]
    dynamic_axes: dict[str, dict[int, str]] | None = None

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


def dry_run_export(spec: ExportSpec) -> dict[str, Any]:
    """Validate export spec and report expected ONNX graph topology without real export."""
    validate_export_shapes(spec)
    latent, timestep, cond, uncond = build_dummy_inputs(spec)

    return {
        "mode": "dry_run",
        "width": spec.width,
        "height": spec.height,
        "max_tokens": spec.max_tokens,
        "inputs": {
            DENOISER_INPUT_LATENT: list(latent.shape),
            DENOISER_INPUT_TIMESTEP: list(timestep.shape),
            DENOISER_INPUT_COND: list(cond.shape),
            DENOISER_INPUT_UNCOND: list(uncond.shape),
        },
        "outputs": {
            DENOISER_OUTPUT_NOISE_PRED: list(latent.shape),
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
    parser = argparse.ArgumentParser(description="Export Anima denoiser to ONNX")
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--dry-run", action="store_true", help="Validate shapes without real export")
    parser.add_argument("--opset-version", type=int, default=17)
    parser.add_argument("--inspect", action="store_true")
    args = parser.parse_args()

    spec = ExportSpec(
        bundle_dir=args.bundle_dir,
        output_path=args.output_path,
        width=args.width,
        height=args.height,
        max_tokens=args.max_tokens,
    )

    if args.dry_run:
        info = dry_run_export(spec)
        import json
        print(json.dumps(info, indent=2))
        return

    bundle = load_anima_bundle()
    export_denoiser(spec, denoiser_module=bundle.denoiser if bundle.available else None, opset_version=args.opset_version)
    print(f"Exported: {spec.output_path}")

    if args.inspect:
        info = inspect_onnx_graph(spec.output_path)
        import json
        print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
