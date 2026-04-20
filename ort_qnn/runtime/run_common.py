from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ort_qnn.runtime.provider_options import ProviderSpec, qnn_provider_options


@dataclass(frozen=True)
class RunSpec:
    model_path: Path
    backend_path: Path
    profiling_path: Path
    input_feed: dict[str, Any]


@dataclass(frozen=True)
class RunOutcome:
    provider_visible: bool
    session_ok: bool
    execute_ok: bool
    failing_stage: str


def provider_visible(name: str, providers: list[str]) -> bool:
    return name in providers


def run_once(ort_module, spec: RunSpec) -> RunOutcome:
    providers = ort_module.get_available_providers()
    if not provider_visible("QNNExecutionProvider", providers):
        return RunOutcome(False, False, False, "provider_unavailable")

    options = qnn_provider_options(
        ProviderSpec(
            backend_path=spec.backend_path,
            profiling_path=spec.profiling_path,
            disable_cpu_fallback=True,
        )
    )

    try:
        session = ort_module.InferenceSession(
            str(spec.model_path),
            providers=["QNNExecutionProvider"],
            provider_options=[options],
        )
    except Exception:
        return RunOutcome(True, False, False, "session_create_failed")

    try:
        session.run(None, spec.input_feed)
    except Exception:
        return RunOutcome(True, True, False, "execute_failed")

    return RunOutcome(True, True, True, "")
