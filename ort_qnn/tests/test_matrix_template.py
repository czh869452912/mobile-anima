from pathlib import Path


def test_matrix_template_contains_provider_and_execution_columns():
    text = Path("ort_qnn/docs/ort-qnn-matrix.md").read_text()

    assert "Acquisition path" in text
    assert "ORT build" in text
    assert "| Graph | Provider Visible | Session | Execute | Profile | Failing Stage |" in text
