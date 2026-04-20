from dataclasses import dataclass
from pathlib import Path
import sysconfig


@dataclass(frozen=True)
class PackageSpec:
    release_root: Path
    qairt_root: Path
    ort_version: str
    qairt_version: str


def required_runtime_files(spec: PackageSpec) -> list[Path]:
    return [
        Path("onnxruntime/capi/onnxruntime_pybind11_state.so"),
        Path(f"onnxruntime/capi/libonnxruntime.so.{spec.ort_version}"),
        Path("onnxruntime/capi/libonnxruntime_providers_qnn.so"),
        Path("onnxruntime/capi/libonnxruntime_providers_shared.so"),
        Path("onnxruntime/capi/libQnnCpu.so"),
        Path("onnxruntime/capi/libQnnSystem.so"),
    ]


def pybind_extension_filenames(ext_suffix: str | None = None) -> list[str]:
    suffix = ext_suffix or sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    names = ["onnxruntime_pybind11_state.so"]
    abi_name = f"onnxruntime_pybind11_state{suffix}"
    if abi_name not in names:
        names.append(abi_name)
    return names


def manifest_dict(spec: PackageSpec) -> dict:
    return {
        "package_name": "onnxruntime-qnn-local",
        "package_version": spec.ort_version,
        "ort_version": spec.ort_version,
        "qairt_version": spec.qairt_version,
        "release_root": str(spec.release_root),
        "qairt_root": str(spec.qairt_root),
        "required_env": {
            "PYTHONPATH": "<bundle_root>",
            "LD_LIBRARY_PATH": "<bundle_root>/onnxruntime/capi",
        },
    }
