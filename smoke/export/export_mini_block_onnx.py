import argparse
from pathlib import Path

from smoke.export.export_common import ExportArtifactSpec, onnx_output_path


def resolve_mini_block_output(artifact_root: Path) -> Path:
    return onnx_output_path(ExportArtifactSpec(artifact_root=artifact_root, graph_name="mini_block"))


def parse_mini_export_args(argv: list[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", required=True)
    return parser.parse_args(argv)
