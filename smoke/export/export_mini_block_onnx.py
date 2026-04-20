from pathlib import Path

from smoke.export.export_common import ExportArtifactSpec, onnx_output_path


def resolve_mini_block_output(artifact_root: Path) -> Path:
    return onnx_output_path(ExportArtifactSpec(artifact_root=artifact_root, graph_name="mini_block"))
