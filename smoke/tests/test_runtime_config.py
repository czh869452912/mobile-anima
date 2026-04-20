from pathlib import Path

from smoke.runtime.run_ort_qnn import OrtQnnRunSpec, qnn_provider_options


def test_qnn_provider_options_enable_context_and_disable_cpu_fallback():
    spec = OrtQnnRunSpec(
        model_path=Path("smoke/artifacts/qnn/toy/toy_ctx.onnx"),
        backend_path=Path("/opt/qairt/lib/x86_64-linux-clang/libQnnHtp.so"),
        profiling_path=Path("smoke/artifacts/profiles/toy_profile.csv"),
        disable_cpu_fallback=True,
    )

    options = qnn_provider_options(spec)

    assert options["backend_path"].endswith("libQnnHtp.so")
    assert options["ep.context_enable"] == "1"
    assert options["session.disable_cpu_ep_fallback"] == "1"
