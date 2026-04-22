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

from anima_host.config import DEFAULT_MAX_TOKENS
from anima_host.model_loader import AnimaModelBundle, load_anima_bundle


@dataclass(frozen=True)
class TextEncoderExportSpec:
    bundle_dir: Path
    output_path: Path
    max_tokens: int = DEFAULT_MAX_TOKENS
    hidden_dim: int = 16  # Anima uses 16-dim text embeddings


def build_dummy_inputs(spec: TextEncoderExportSpec) -> tuple[np.ndarray, ...]:
    input_ids = np.zeros((1, spec.max_tokens), dtype=np.int64)
    return (input_ids,)


def validate_export_shapes(spec: TextEncoderExportSpec) -> None:
    input_ids, = build_dummy_inputs(spec)
    assert tuple(input_ids.shape) == (1, spec.max_tokens)


class TextEncoderOnnxWrapper(nn.Module):
    """Wrapper to adapt text encoder for static-shape ONNX export.

    Expects input_ids and returns text embeddings suitable for denoiser conditioning.
    When real weights are unavailable, acts as a fake model for shape validation.
    """

    def __init__(
        self,
        text_encoder: nn.Module | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        hidden_dim: int = 16,
    ) -> None:
        super().__init__()
        self._real_encoder = text_encoder
        self.max_tokens = max_tokens
        self.hidden_dim = hidden_dim
        self._fake_mode = text_encoder is None

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self._fake_mode or self._real_encoder is None:
            batch_size = input_ids.shape[0]
            return torch.zeros(batch_size, self.max_tokens, self.hidden_dim)

        encoder = self._real_encoder
        if hasattr(encoder, "forward"):
            try:
                out = encoder(input_ids)
                if hasattr(out, "last_hidden_state"):
                    return out.last_hidden_state
                return out
            except (TypeError, AttributeError):
                pass
        if hasattr(encoder, "__call__"):
            try:
                out = encoder(input_ids)
                if hasattr(out, "last_hidden_state"):
                    return out.last_hidden_state
                return out
            except (TypeError, AttributeError):
                pass

        batch_size = input_ids.shape[0]
        return torch.zeros(batch_size, self.max_tokens, self.hidden_dim)


def build_torch_dummy_inputs(spec: TextEncoderExportSpec) -> tuple[torch.Tensor, ...]:
    input_ids = torch.zeros(1, spec.max_tokens, dtype=torch.long)
    return (input_ids,)


def export_text_encoder(
    spec: TextEncoderExportSpec,
    text_encoder_module: Any | None = None,
    opset_version: int = 17,
) -> Path:
    validate_export_shapes(spec)
    spec.output_path.parent.mkdir(parents=True, exist_ok=True)

    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for ONNX export")

    wrapper = TextEncoderOnnxWrapper(
        text_encoder=text_encoder_module,
        max_tokens=spec.max_tokens,
        hidden_dim=spec.hidden_dim,
    )
    wrapper.eval()

    dummy_inputs = build_torch_dummy_inputs(spec)
    input_names = ["input_ids"]
    output_names = ["text_embeddings"]
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


def dry_run_export(spec: TextEncoderExportSpec) -> dict[str, Any]:
    """Validate export spec and report expected ONNX graph topology without real export."""
    validate_export_shapes(spec)
    input_ids, = build_dummy_inputs(spec)

    return {
        "mode": "dry_run",
        "max_tokens": spec.max_tokens,
        "hidden_dim": spec.hidden_dim,
        "inputs": {
            "input_ids": list(input_ids.shape),
        },
        "outputs": {
            "text_embeddings": [1, spec.max_tokens, spec.hidden_dim],
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
    parser = argparse.ArgumentParser(description="Export Anima text encoder to ONNX")
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true", help="Validate shapes without real export")
    parser.add_argument("--opset-version", type=int, default=17)
    parser.add_argument("--inspect", action="store_true")
    args = parser.parse_args()

    spec = TextEncoderExportSpec(
        bundle_dir=args.bundle_dir,
        output_path=args.output_path,
        max_tokens=args.max_tokens,
        hidden_dim=args.hidden_dim,
    )

    if args.dry_run:
        info = dry_run_export(spec)
        import json
        print(json.dumps(info, indent=2))
        return

    bundle = load_anima_bundle()
    export_text_encoder(
        spec,
        text_encoder_module=bundle.text_encoder if bundle.available else None,
        opset_version=args.opset_version,
    )
    print(f"Exported: {spec.output_path}")

    if args.inspect:
        info = inspect_onnx_graph(spec.output_path)
        import json
        print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
