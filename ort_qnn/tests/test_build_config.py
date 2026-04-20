from pathlib import Path

from ort_qnn.build.build_config import BuildSpec, build_ort_command


def test_build_ort_command_requests_qnn_and_python_wheel():
    spec = BuildSpec(
        qairt_root=Path("/opt/qcom/aistack/qairt/2.41.0.251128"),
        ort_src_root=Path("ort_qnn/artifacts/ort/onnxruntime"),
        python310=Path("/opt/qcom/qairt-py310/bin/python"),
        build_dir=Path("ort_qnn/artifacts/ort/build/linux_qnn"),
    )

    command = build_ort_command(spec)

    assert command[0].endswith("python")
    assert "tools/ci_build/build.py" in command
    assert "--use_qnn" in command
    assert "--qnn_home" in command
    assert str(spec.qairt_root) in command
    assert "--build_wheel" in command
    assert "--allow_running_as_root" in command
