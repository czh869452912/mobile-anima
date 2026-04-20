import argparse
from pathlib import Path

from smoke.export.export_common import ExportArtifactSpec, onnx_output_path


def resolve_toy_output(artifact_root: Path) -> Path:
    return onnx_output_path(ExportArtifactSpec(artifact_root=artifact_root, graph_name="toy"))


def parse_toy_export_args(argv: list[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", required=True)
    return parser.parse_args(argv)
