from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunResult:
    graph_name: str
    cpu_ort_ok: bool
    qnn_compile_ok: bool
    ort_qnn_session_ok: bool
    ort_qnn_execute_ok: bool
    profile_generated: bool
    failing_stage: str
    log_path: Path


def to_markdown_row(result: RunResult) -> str:
    def mark(value: bool) -> str:
        return "PASS" if value else "FAIL"

    return (
        f"| {result.graph_name} | {mark(result.cpu_ort_ok)} | {mark(result.qnn_compile_ok)} | "
        f"{mark(result.ort_qnn_session_ok)} | {mark(result.ort_qnn_execute_ok)} | "
        f"{mark(result.profile_generated)} | {result.failing_stage} | {result.log_path} |"
    )
