from pathlib import Path

from anima_host.export_text_encoder import (
    TextEncoderExportSpec,
    build_dummy_inputs,
    dry_run_export,
    validate_export_shapes,
)


def test_build_dummy_inputs_matches_expected_shape():
    spec = TextEncoderExportSpec(
        bundle_dir=Path("/models"),
        output_path=Path("build/text_encoder.onnx"),
        max_tokens=256,
    )
    input_ids, = build_dummy_inputs(spec)
    assert tuple(input_ids.shape) == (1, 256)
    assert input_ids.dtype == "int64"


def test_validate_export_shapes_accepts_valid_spec():
    spec = TextEncoderExportSpec(
        bundle_dir=Path("/models"),
        output_path=Path("build/text_encoder.onnx"),
        max_tokens=256,
    )
    validate_export_shapes(spec)


def test_dry_run_export_returns_expected_topology():
    spec = TextEncoderExportSpec(
        bundle_dir=Path("/models"),
        output_path=Path("build/text_encoder.onnx"),
        max_tokens=256,
        hidden_dim=16,
    )
    info = dry_run_export(spec)
    assert info["mode"] == "dry_run"
    assert info["max_tokens"] == 256
    assert info["hidden_dim"] == 16
    assert info["inputs"]["input_ids"] == [1, 256]
    assert info["outputs"]["text_embeddings"] == [1, 256, 16]


def test_dry_run_export_with_custom_hidden_dim():
    spec = TextEncoderExportSpec(
        bundle_dir=Path("/models"),
        output_path=Path("build/text_encoder.onnx"),
        max_tokens=256,
        hidden_dim=32,
    )
    info = dry_run_export(spec)
    assert info["outputs"]["text_embeddings"] == [1, 256, 32]
