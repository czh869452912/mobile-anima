from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExportArtifactSpec:
    artifact_root: Path
    graph_name: str


def onnx_output_path(spec: ExportArtifactSpec) -> Path:
    return spec.artifact_root / "onnx" / f"{spec.graph_name}.onnx"
