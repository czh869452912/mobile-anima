from pathlib import Path

from ort_qnn.package.package_manifest import PackageSpec, manifest_dict, required_runtime_files


def test_required_runtime_files_include_qnn_provider_and_pybind_module():
    spec = PackageSpec(
        release_root=Path("ort_qnn/artifacts/ort/build/linux_qnn/Release"),
        qairt_root=Path("/opt/qcom/aistack/qairt/2.41.0.251128"),
        ort_version="1.23.2",
        qairt_version="2.41.0",
    )

    files = required_runtime_files(spec)

    assert Path("onnxruntime/capi/onnxruntime_pybind11_state.so") in files
    assert Path("onnxruntime/capi/libonnxruntime_providers_qnn.so") in files
    assert Path("onnxruntime/capi/libQnnCpu.so") in files


def test_manifest_dict_records_versions_and_expected_env_vars():
    spec = PackageSpec(
        release_root=Path("ort_qnn/artifacts/ort/build/linux_qnn/Release"),
        qairt_root=Path("/opt/qcom/aistack/qairt/2.41.0.251128"),
        ort_version="1.23.2",
        qairt_version="2.41.0",
    )

    manifest = manifest_dict(spec)

    assert manifest["ort_version"] == "1.23.2"
    assert manifest["qairt_version"] == "2.41.0"
    assert "PYTHONPATH" in manifest["required_env"]
    assert "LD_LIBRARY_PATH" in manifest["required_env"]
