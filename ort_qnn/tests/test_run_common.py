from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ort_qnn.runtime.run_common import RunSpec, provider_visible, run_once


@dataclass
class FakeSession:
    providers: list[str]
    provider_options: list[dict[str, str]]

    def run(self, _, feed):
        return [np.asarray(feed["input"]) + 1.0]


class FakeOrt:
    @staticmethod
    def get_available_providers():
        return ["CPUExecutionProvider", "QNNExecutionProvider"]

    class InferenceSession:
        def __init__(self, model_path, providers, provider_options):
            self.inner = FakeSession(providers, provider_options)

        def run(self, output_names, feed):
            return self.inner.run(output_names, feed)


def test_provider_visible_checks_provider_membership():
    assert provider_visible("QNNExecutionProvider", ["CPUExecutionProvider", "QNNExecutionProvider"])
    assert not provider_visible("QNNExecutionProvider", ["CPUExecutionProvider"])


def test_run_once_marks_success_when_fake_ort_executes():
    spec = RunSpec(
        model_path=Path("smoke/artifacts/onnx/toy.onnx"),
        backend_path=Path("/opt/qcom/aistack/qairt/2.41.0.251128/lib/x86_64-linux-clang/libQnnCpu.so"),
        profiling_path=Path("ort_qnn/artifacts/profiles/toy_qnn.csv"),
        input_feed={"input": np.zeros((1, 8), dtype=np.float32)},
    )

    result = run_once(FakeOrt, spec)

    assert result.provider_visible is True
    assert result.session_ok is True
    assert result.execute_ok is True
    assert result.failing_stage == ""
