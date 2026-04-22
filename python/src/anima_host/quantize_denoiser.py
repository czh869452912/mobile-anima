from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from onnxruntime.quantization import quantize_static, quantize_dynamic, CalibrationDataReader
    from onnxruntime.quantization.shape_inference import quant_pre_process
    _ORT_QUANTIZE_AVAILABLE = True
except Exception:
    _ORT_QUANTIZE_AVAILABLE = False
    # Define a stub base class so downstream type hints work without ORT.
    class CalibrationDataReader:  # type: ignore[no-redef]
        def get_next(self) -> dict[str, Any] | None:
            return None

        def rewind(self) -> None:
            pass


@dataclass(frozen=True)
class QuantizeSpec:
    input_model: Path
    output_model: Path
    calibration_data: Path | None = None
    per_channel: bool = True
    activation_type: str = "QInt8"
    weight_type: str = "QInt8"
    reduce_range: bool = False
    quant_format: str = "QDQ"  # QDQ or QOperator

    def __post_init__(self) -> None:
        if self.quant_format not in {"QDQ", "QOperator"}:
            raise ValueError(f"Unsupported quant_format: {self.quant_format}")
        if self.activation_type not in {"QInt8", "QUInt8"}:
            raise ValueError(f"Unsupported activation_type: {self.activation_type}")
        if self.weight_type not in {"QInt8", "QUInt8"}:
            raise ValueError(f"Unsupported weight_type: {self.weight_type}")


def _has_onnx() -> bool:
    try:
        import onnx
        return True
    except Exception:
        return False


def _run_shape_inference(input_model: Path, output_model: Path) -> None:
    if not _ORT_QUANTIZE_AVAILABLE:
        return
    quant_pre_process(str(input_model), str(output_model), skip_onnx_shape=False)


class _SyntheticCalibrationDataReader(CalibrationDataReader):
    """Calibration data reader that yields synthetic inputs.

    Used when no real calibration data is provided.
    """

    def __init__(self, input_shapes: dict[str, list[int]], num_samples: int = 10) -> None:
        self.input_shapes = input_shapes
        self.num_samples = num_samples
        self._current = 0

    def get_next(self) -> dict[str, Any] | None:
        import numpy as np
        if self._current >= self.num_samples:
            return None
        self._current += 1
        return {
            name: np.random.randn(*shape).astype(np.float32) * 0.5
            for name, shape in self.input_shapes.items()
        }

    def rewind(self) -> None:
        self._current = 0


def _load_calibration_data_reader(calibration_path: Path | None, input_shapes: dict[str, list[int]]) -> CalibrationDataReader:
    if calibration_path is not None and calibration_path.exists():
        import numpy as np
        data = dict(np.load(calibration_path))
        class NpzReader(CalibrationDataReader):
            def __init__(self, arrays: dict[str, Any]) -> None:
                self.arrays = arrays
                self.names = list(arrays.keys())
                self._idx = 0
                self._count = len(next(iter(arrays.values())))

            def get_next(self) -> dict[str, Any] | None:
                if self._idx >= self._count:
                    return None
                result = {name: self.arrays[name][self._idx] for name in self.names}
                self._idx += 1
                return result

            def rewind(self) -> None:
                self._idx = 0
        return NpzReader(data)

    return _SyntheticCalibrationDataReader(input_shapes, num_samples=10)


def _extract_input_shapes(model_path: Path) -> dict[str, list[int]]:
    import onnx
    model = onnx.load(str(model_path))
    shapes = {}
    for inp in model.graph.input:
        shape = []
        for dim in inp.type.tensor_type.shape.dim:
            if dim.dim_value > 0:
                shape.append(dim.dim_value)
            elif dim.dim_param:
                shape.append(1)
            else:
                shape.append(1)
        shapes[inp.name] = shape
    return shapes


def quantize_denoiser(spec: QuantizeSpec, dry_run: bool = False) -> Path | None:
    if dry_run or not _ORT_QUANTIZE_AVAILABLE:
        return _dry_run_quantize(spec)

    if not _has_onnx():
        raise RuntimeError("onnx is required for quantization")

    spec.output_model.parent.mkdir(parents=True, exist_ok=True)

    # Pre-process: shape inference
    preprocessed = spec.output_model.parent / f"{spec.input_model.stem}_inferred.onnx"
    _run_shape_inference(spec.input_model, preprocessed)

    input_shapes = _extract_input_shapes(preprocessed)
    calibrator = _load_calibration_data_reader(spec.calibration_data, input_shapes)

    from onnxruntime.quantization import QuantType, QuantFormat

    activation_qtype = QuantType.QInt8 if spec.activation_type == "QInt8" else QuantType.QUInt8
    weight_qtype = QuantType.QInt8 if spec.weight_type == "QInt8" else QuantType.QUInt8
    quant_format = QuantFormat.QDQ if spec.quant_format == "QDQ" else QuantFormat.QOperator

    if spec.calibration_data is not None:
        quantize_static(
            model_input=str(preprocessed),
            model_output=str(spec.output_model),
            calibration_data_reader=calibrator,
            quant_format=quant_format,
            activation_type=activation_qtype,
            weight_type=weight_qtype,
            per_channel=spec.per_channel,
            reduce_range=spec.reduce_range,
        )
    else:
        quantize_dynamic(
            model_input=str(preprocessed),
            model_output=str(spec.output_model),
            weight_type=weight_qtype,
            per_channel=spec.per_channel,
            reduce_range=spec.reduce_range,
        )

    if preprocessed.exists():
        preprocessed.unlink()

    print(f"[quantize] Quantized model saved to {spec.output_model}")
    return spec.output_model


def _dry_run_quantize(spec: QuantizeSpec) -> None:
    print(f"[dry-run] Quantization spec validated")
    print(f"[dry-run]   input: {spec.input_model}")
    print(f"[dry-run]   output: {spec.output_model}")
    print(f"[dry-run]   calibration: {spec.calibration_data}")
    print(f"[dry-run]   per_channel: {spec.per_channel}")
    print(f"[dry-run]   activation_type: {spec.activation_type}")
    print(f"[dry-run]   weight_type: {spec.weight_type}")
    print(f"[dry-run]   quant_format: {spec.quant_format}")
    print(f"[dry-run]   reduce_range: {spec.reduce_range}")

    if _ORT_QUANTIZE_AVAILABLE:
        print(f"[dry-run]   onnxruntime.quantization: available")
    else:
        print(f"[dry-run]   onnxruntime.quantization: NOT available (would use dynamic quantization fallback)")
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantize Anima denoiser ONNX model")
    parser.add_argument("--input-model", type=Path, required=True)
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--calibration-data", type=Path)
    parser.add_argument("--per-channel", action="store_true", default=True)
    parser.add_argument("--no-per-channel", dest="per_channel", action="store_false")
    parser.add_argument("--activation-type", default="QInt8", choices=["QInt8", "QUInt8"])
    parser.add_argument("--weight-type", default="QInt8", choices=["QInt8", "QUInt8"])
    parser.add_argument("--quant-format", default="QDQ", choices=["QDQ", "QOperator"])
    parser.add_argument("--reduce-range", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    spec = QuantizeSpec(
        input_model=args.input_model,
        output_model=args.output_model,
        calibration_data=args.calibration_data,
        per_channel=args.per_channel,
        activation_type=args.activation_type,
        weight_type=args.weight_type,
        quant_format=args.quant_format,
        reduce_range=args.reduce_range,
    )

    quantize_denoiser(spec, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
