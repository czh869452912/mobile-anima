from pathlib import Path

from smoke.compile.compile_qnn import CompileSpec, build_compile_command
from smoke.compile.inspect_compile_result import expected_qnn_outputs


def test_build_compile_command_points_to_context_generator():
    spec = CompileSpec(
        sdk_root=Path("/opt/qairt"),
        onnx_model=Path("smoke/artifacts/onnx/toy.onnx"),
        output_dir=Path("smoke/artifacts/qnn/toy"),
        profiling_level="detailed",
    )

    command = build_compile_command(spec)

    assert "qnn-context-binary-generator" in command[0]
    assert "--model" in command
    assert str(spec.onnx_model) in command
    assert "--profiling_level" in command


def test_expected_qnn_outputs_names_context_and_profile_targets():
    outputs = expected_qnn_outputs(Path("smoke/artifacts/qnn/toy"), "toy")

    assert outputs["context"] == Path("smoke/artifacts/qnn/toy/toy_ctx.onnx")
    assert outputs["profile"] == Path("smoke/artifacts/qnn/toy/toy_profile.csv")
