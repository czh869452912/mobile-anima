from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BundleSpec:
    bundle_root: Path


REQUIRED_NAMES = [
    "onnxruntime/capi/onnxruntime_pybind11_state.so",
    "onnxruntime/capi/libonnxruntime.so.1.23.2",
    "onnxruntime/capi/libonnxruntime_providers_qnn.so",
    "onnxruntime/capi/libonnxruntime_providers_shared.so",
    "onnxruntime/capi/libQnnCpu.so",
    "onnxruntime/capi/libQnnSystem.so",
]


def missing_runtime_files(spec: BundleSpec) -> list[Path]:
    missing: list[Path] = []
    for rel in REQUIRED_NAMES:
        rel_path = Path(rel)
        if not (spec.bundle_root / rel_path).exists():
            missing.append(rel_path)
    return missing
