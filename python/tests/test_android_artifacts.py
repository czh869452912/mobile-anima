from pathlib import Path

import json

from anima_host.android_artifacts import (
    AndroidArtifactPackager,
    ArtifactSpec,
)


def test_package_creates_metadata_and_directories():
    build_dir = Path("/tmp/test_anima_build")
    build_dir.mkdir(parents=True, exist_ok=True)

    # Create a dummy denoiser.onnx so the packager has something to copy
    (build_dir / "denoiser.onnx").write_text("dummy")

    assets_dir = Path("/tmp/test_anima_android_assets")
    spec = ArtifactSpec(
        build_dir=build_dir,
        android_asset_dir=assets_dir,
        model_id="circlestone-labs/Anima",
        resolutions=[(1024, 1024)],
    )
    packager = AndroidArtifactPackager(spec)
    report = packager.package()

    assert report.metadata_path.exists()
    assert report.assets_dir.exists()
    assert (report.assets_dir / "denoiser.onnx").exists()

    with open(report.metadata_path) as f:
        metadata = json.load(f)
    assert metadata["model_id"] == "circlestone-labs/Anima"
    assert metadata["max_tokens"] == 256
    assert "files" in metadata
    assert "input_shapes" in metadata
    assert "output_shapes" in metadata

    # Cleanup
    import shutil
    shutil.rmtree(build_dir, ignore_errors=True)
    shutil.rmtree(assets_dir, ignore_errors=True)


def test_package_report_to_dict():
    build_dir = Path("/tmp/test_anima_build2")
    build_dir.mkdir(parents=True, exist_ok=True)
    (build_dir / "denoiser.onnx").write_text("dummy")

    assets_dir = Path("/tmp/test_anima_android_assets2")
    spec = ArtifactSpec(
        build_dir=build_dir,
        android_asset_dir=assets_dir,
    )
    packager = AndroidArtifactPackager(spec)
    report = packager.package()

    d = report.to_dict()
    assert "assets_dir" in d
    assert "metadata_path" in d
    assert "files_copied" in d
    assert "metadata" in d

    import shutil
    shutil.rmtree(build_dir, ignore_errors=True)
    shutil.rmtree(assets_dir, ignore_errors=True)
