from pathlib import Path

from smoke.export.export_common import ExportArtifactSpec, onnx_output_path


def test_onnx_output_path_places_files_under_artifacts_onnx():
    spec = ExportArtifactSpec(artifact_root=Path("smoke/artifacts"), graph_name="toy")

    path = onnx_output_path(spec)

    assert path == Path("smoke/artifacts/onnx/toy.onnx")
