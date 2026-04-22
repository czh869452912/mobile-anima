from pathlib import Path

from anima_host.export_vae import (
    VaeExportSpec,
    build_dummy_inputs,
    dry_run_export,
    validate_export_shapes,
)


def test_build_dummy_inputs_matches_expected_shape():
    spec = VaeExportSpec(
        bundle_dir=Path("/models"),
        output_path=Path("build/vae.onnx"),
        width=1024,
        height=1024,
    )
    latent, = build_dummy_inputs(spec)
    assert tuple(latent.shape) == (1, 4, 128, 128)


def test_validate_export_shapes_accepts_all_resolutions():
    for w, h in [(1024, 1024), (768, 1024), (1024, 768)]:
        spec = VaeExportSpec(
            bundle_dir=Path("/models"),
            output_path=Path("build/vae.onnx"),
            width=w,
            height=h,
        )
        validate_export_shapes(spec)


def test_dry_run_export_returns_expected_topology():
    spec = VaeExportSpec(
        bundle_dir=Path("/models"),
        output_path=Path("build/vae.onnx"),
        width=1024,
        height=1024,
    )
    info = dry_run_export(spec)
    assert info["mode"] == "dry_run"
    assert info["width"] == 1024
    assert info["height"] == 1024
    assert info["inputs"]["latent"] == [1, 4, 128, 128]
    assert info["outputs"]["image"] == [1, 3, 1024, 1024]


def test_dry_run_export_matches_768x1024():
    spec = VaeExportSpec(
        bundle_dir=Path("/models"),
        output_path=Path("build/vae.onnx"),
        width=768,
        height=1024,
    )
    info = dry_run_export(spec)
    assert info["inputs"]["latent"] == [1, 4, 128, 96]
    assert info["outputs"]["image"] == [1, 3, 1024, 768]
