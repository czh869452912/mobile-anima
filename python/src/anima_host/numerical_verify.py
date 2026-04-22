from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import onnxruntime as ort
    _ORT_AVAILABLE = True
except Exception:
    _ORT_AVAILABLE = False

try:
    import torch
    _TORCH_AVAILABLE = True
except Exception:
    _TORCH_AVAILABLE = False


@dataclass(frozen=True)
class VerificationResult:
    max_abs_diff: float
    mean_abs_diff: float
    mse: float
    cosine_similarity: float
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_abs_diff": self.max_abs_diff,
            "mean_abs_diff": self.mean_abs_diff,
            "mse": self.mse,
            "cosine_similarity": self.cosine_similarity,
            "passed": self.passed,
        }


def _compare_arrays(a: np.ndarray, b: np.ndarray, rtol: float = 1e-3, atol: float = 1e-5) -> VerificationResult:
    diff = np.abs(a - b)
    max_diff = float(np.max(diff))
    mean_diff = float(np.mean(diff))
    mse = float(np.mean(diff ** 2))

    flat_a = a.flatten()
    flat_b = b.flatten()
    dot = np.dot(flat_a, flat_b)
    norm_a = np.linalg.norm(flat_a)
    norm_b = np.linalg.norm(flat_b)
    cos_sim = float(dot / (norm_a * norm_b + 1e-12))

    passed = bool(np.allclose(a, b, rtol=rtol, atol=atol))

    return VerificationResult(
        max_abs_diff=max_diff,
        mean_abs_diff=mean_diff,
        mse=mse,
        cosine_similarity=cos_sim,
        passed=passed,
    )


def verify_onnx_against_onnx(
    model_a: Path,
    model_b: Path,
    dummy_inputs: dict[str, np.ndarray],
    rtol: float = 1e-3,
    atol: float = 1e-5,
) -> VerificationResult:
    if not _ORT_AVAILABLE:
        raise RuntimeError("onnxruntime is required for numerical verification")

    sess_a = ort.InferenceSession(str(model_a))
    sess_b = ort.InferenceSession(str(model_b))

    out_a = sess_a.run(None, dummy_inputs)[0]
    out_b = sess_b.run(None, dummy_inputs)[0]

    return _compare_arrays(out_a, out_b, rtol=rtol, atol=atol)


def verify_pytorch_against_onnx(
    pytorch_model: Any,
    onnx_model: Path,
    dummy_inputs: tuple[Any, ...],
    input_names: list[str] | None = None,
    rtol: float = 1e-3,
    atol: float = 1e-5,
) -> VerificationResult:
    if not _ORT_AVAILABLE:
        raise RuntimeError("onnxruntime is required for numerical verification")
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for numerical verification")

    with torch.no_grad():
        pytorch_out = pytorch_model(*dummy_inputs)
        if hasattr(pytorch_out, "detach"):
            pytorch_out = pytorch_out.detach().cpu().numpy()
        else:
            pytorch_out = np.array(pytorch_out)

    sess = ort.InferenceSession(str(onnx_model))
    if input_names is None:
        input_names = [inp.name for inp in sess.get_inputs()]

    onnx_inputs = {name: dummy_inputs[i].detach().cpu().numpy() if hasattr(dummy_inputs[i], "detach") else np.array(dummy_inputs[i]) for i, name in enumerate(input_names)}
    onnx_out = sess.run(None, onnx_inputs)[0]

    return _compare_arrays(pytorch_out, onnx_out, rtol=rtol, atol=atol)


def summarize_result(result: VerificationResult) -> str:
    lines = [
        "Numerical Verification Result",
        f"  Max absolute diff:  {result.max_abs_diff:.6e}",
        f"  Mean absolute diff: {result.mean_abs_diff:.6e}",
        f"  MSE:                {result.mse:.6e}",
        f"  Cosine similarity:  {result.cosine_similarity:.6f}",
        f"  Passed:             {result.passed}",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Numerically verify ONNX export against baseline")
    subparsers = parser.add_subparsers(dest="command")

    onnx_vs_onnx = subparsers.add_parser("onnx-vs-onnx")
    onnx_vs_onnx.add_argument("--model-a", type=Path, required=True)
    onnx_vs_onnx.add_argument("--model-b", type=Path, required=True)

    pytorch_vs_onnx = subparsers.add_parser("pytorch-vs-onnx")
    pytorch_vs_onnx.add_argument("--onnx-model", type=Path, required=True)

    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-5)

    args = parser.parse_args()

    if args.command == "onnx-vs-onnx":
        print("onnx-vs-onnx requires --inputs (not implemented in CLI)")
    elif args.command == "pytorch-vs-onnx":
        print("pytorch-vs-onnx requires --pytorch-model (not implemented in CLI)")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
