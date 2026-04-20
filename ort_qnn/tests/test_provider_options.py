from pathlib import Path

from ort_qnn.runtime.provider_options import ProviderSpec, qnn_provider_options


def test_qnn_provider_options_enable_context_and_disable_cpu_fallback():
    spec = ProviderSpec(
        backend_path=Path("/opt/qcom/aistack/qairt/2.41.0.251128/lib/x86_64-linux-clang/libQnnCpu.so"),
        profiling_path=Path("ort_qnn/artifacts/profiles/toy_ort_qnn.csv"),
        disable_cpu_fallback=True,
    )

    options = qnn_provider_options(spec)

    assert options["backend_path"].endswith("libQnnCpu.so")
    assert options["ep.context_enable"] == "1"
    assert options["session.disable_cpu_ep_fallback"] == "1"
