from pathlib import Path


def test_bootstrap_script_mentions_qpm_detection_and_rerun_flow():
    text = Path("scripts/bootstrap_ubuntu2404.sh").read_text()

    assert "command -v qpm-cli" in text
    assert "rerun" in text.lower()
    assert "qairt" in text.lower()


def test_bootstrap_script_installs_android_and_python_prereqs():
    text = Path("scripts/bootstrap_ubuntu2404.sh").read_text()

    assert "openjdk" in text.lower()
    assert "android" in text.lower()
    assert "python3.10" in text.lower() or "Python 3.10" in text
