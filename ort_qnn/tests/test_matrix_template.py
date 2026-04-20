from pathlib import Path


def test_matrix_template_contains_acquisition_path_section():
    text = Path("ort_qnn/docs/ort-qnn-matrix.md").read_text()

    assert "Acquisition path" in text
    assert "ORT build" in text
    assert "| Graph | Session | Execute | Profile | Failing Stage |" in text
