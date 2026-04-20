from pathlib import Path

from smoke.runtime.collect_profile import RunResult, to_markdown_row


def test_to_markdown_row_formats_pass_fail_statuses():
    result = RunResult(
        graph_name="toy",
        cpu_ort_ok=True,
        qnn_compile_ok=True,
        ort_qnn_session_ok=False,
        ort_qnn_execute_ok=False,
        profile_generated=False,
        failing_stage="ort_qnn_session",
        log_path=Path("smoke/artifacts/logs/toy_runtime.log"),
    )

    row = to_markdown_row(result)

    assert "| toy | PASS | PASS | FAIL | FAIL | FAIL | ort_qnn_session |" in row
from pathlib import Path


def test_smoke_matrix_template_contains_all_validation_columns():
    text = Path("smoke/docs/smoke-matrix.md").read_text()

    assert "| Graph | CPU ORT | QNN Compile | ORT QNN Session | ORT QNN Execute | Profile | Failing Stage |" in text
    assert "SDK version" in text
