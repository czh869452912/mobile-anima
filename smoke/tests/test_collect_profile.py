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
