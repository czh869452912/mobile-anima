from pathlib import Path

from anima_host.compile_qnn import build_qnn_context_command
from anima_host.export_denoiser import (
    ExportSpec,
    build_dummy_inputs,
    dry_run_export,
    validate_export_shapes,
)
from anima_host.quantize_denoiser import QuantizeSpec


def test_build_dummy_inputs_matches_fixed_shapes():
    spec = ExportSpec(
        bundle_dir=Path("/models/anima"),
        output_path=Path("build/denoiser.onnx"),
        width=1024,
        height=1024,
        max_tokens=256,
    )

    latent, timestep, cond, uncond = build_dummy_inputs(spec)

    assert tuple(latent.shape) == (1, 4, 128, 128)
    assert tuple(timestep.shape) == (1,)
    assert tuple(cond.shape) == (1, 256, 16)
    assert tuple(uncond.shape) == (1, 256, 16)


def test_validate_export_shapes_accepts_all_resolutions():
    for w, h in [(1024, 1024), (768, 1024), (1024, 768)]:
        spec = ExportSpec(
            bundle_dir=Path("/models"),
            output_path=Path("build/denoiser.onnx"),
            width=w,
            height=h,
            max_tokens=256,
        )
        validate_export_shapes(spec)


def test_dry_run_export_returns_expected_topology():
    spec = ExportSpec(
        bundle_dir=Path("/models"),
        output_path=Path("build/denoiser.onnx"),
        width=1024,
        height=1024,
        max_tokens=256,
    )

    info = dry_run_export(spec)

    assert info["mode"] == "dry_run"
    assert info["width"] == 1024
    assert info["height"] == 1024
    assert info["max_tokens"] == 256
    assert info["inputs"]["latent"] == [1, 4, 128, 128]
    assert info["inputs"]["timestep"] == [1]
    assert info["inputs"]["cond"] == [1, 256, 16]
    assert info["inputs"]["uncond"] == [1, 256, 16]
    assert info["outputs"]["noise_pred"] == [1, 4, 128, 128]


def test_dry_run_export_matches_768x1024():
    spec = ExportSpec(
        bundle_dir=Path("/models"),
        output_path=Path("build/denoiser.onnx"),
        width=768,
        height=1024,
        max_tokens=256,
    )

    info = dry_run_export(spec)
    assert info["inputs"]["latent"] == [1, 4, 128, 96]
    assert info["outputs"]["noise_pred"] == [1, 4, 128, 96]


def test_build_qnn_context_command_contains_no_fallback_profile_flags():
    command = build_qnn_context_command(
        onnx_model=Path("build/denoiser_qdq.onnx"),
        output_dir=Path("build/qnn"),
        sdk_root=Path("/opt/qnn"),
        profiling_level="detailed",
    )

    joined = " ".join(command)
    assert "qnn-context-binary-generator" in joined
    assert "build/denoiser_qdq.onnx" in joined
    assert "--profiling_level" in joined
    assert "detailed" in joined


def test_quantize_spec_uses_qdq_defaults():
    spec = QuantizeSpec(
        input_model=Path("build/denoiser.onnx"),
        output_model=Path("build/denoiser_qdq.onnx"),
    )

    assert spec.per_channel is True
    assert spec.activation_type == "QInt8"
    assert spec.weight_type == "QInt8"
