from pathlib import Path

from anima_host.quantize_denoiser import QuantizeSpec, quantize_denoiser


def test_quantize_spec_uses_qdq_defaults():
    spec = QuantizeSpec(
        input_model=Path("build/denoiser.onnx"),
        output_model=Path("build/denoiser_qdq.onnx"),
    )
    assert spec.per_channel is True
    assert spec.activation_type == "QInt8"
    assert spec.weight_type == "QInt8"
    assert spec.quant_format == "QDQ"
    assert spec.reduce_range is False


def test_quantize_spec_rejects_invalid_quant_format():
    import pytest
    with pytest.raises(ValueError, match="Unsupported quant_format"):
        QuantizeSpec(
            input_model=Path("a.onnx"),
            output_model=Path("b.onnx"),
            quant_format="INVALID",
        )


def test_quantize_spec_rejects_invalid_activation_type():
    import pytest
    with pytest.raises(ValueError, match="Unsupported activation_type"):
        QuantizeSpec(
            input_model=Path("a.onnx"),
            output_model=Path("b.onnx"),
            activation_type="FLOAT16",
        )


def test_quantize_dry_run_does_not_raise():
    spec = QuantizeSpec(
        input_model=Path("build/denoiser.onnx"),
        output_model=Path("build/denoiser_qdq.onnx"),
        calibration_data=None,
    )
    result = quantize_denoiser(spec, dry_run=True)
    assert result is None
