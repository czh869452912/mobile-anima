from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BuildSpec:
    qairt_root: Path
    ort_src_root: Path
    python310: Path
    build_dir: Path


def build_ort_command(spec: BuildSpec) -> list[str]:
    return [
        str(spec.python310),
        "tools/ci_build/build.py",
        "--config",
        "Release",
        "--build_shared_lib",
        "--build_wheel",
        "--parallel",
        "--use_qnn",
        "--qnn_home",
        str(spec.qairt_root),
        "--build_dir",
        str(spec.build_dir),
    ]
