from pathlib import Path

from ort_qnn.package.verify_package import BundleSpec, missing_runtime_files


def test_missing_runtime_files_reports_absent_qnn_libraries(tmp_path: Path):
    spec = BundleSpec(bundle_root=tmp_path)

    missing = missing_runtime_files(spec)

    assert Path("onnxruntime/capi/onnxruntime_pybind11_state.so") in missing
    assert Path("onnxruntime/capi/libonnxruntime_providers_qnn.so") in missing


def test_missing_runtime_files_is_empty_for_minimal_valid_layout(tmp_path: Path):
    capi = tmp_path / "onnxruntime" / "capi"
    capi.mkdir(parents=True)
    for name in [
        "onnxruntime_pybind11_state.so",
        "libonnxruntime.so.1.23.2",
        "libonnxruntime_providers_qnn.so",
        "libonnxruntime_providers_shared.so",
        "libQnnCpu.so",
        "libQnnSystem.so",
    ]:
        (capi / name).write_text("ok")

    spec = BundleSpec(bundle_root=tmp_path)

    assert missing_runtime_files(spec) == []
