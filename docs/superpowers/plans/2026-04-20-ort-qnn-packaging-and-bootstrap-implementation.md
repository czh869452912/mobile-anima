# ORT QNN Packaging and Bootstrap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Package the already-validated desktop Linux `ORT + QNN` build into a portable runtime bundle and a Python-installable wheel path, add a rerunnable Ubuntu 24.04 bootstrap script, and update the docs so a stronger host can be prepared without chat history.

**Architecture:** Keep packaging logic in `ort_qnn/package/`, keep host setup logic in `scripts/bootstrap_ubuntu2404.sh`, and keep runtime validation in the existing `ort_qnn/runtime/` scripts. Build the portable bundle first, verify it against `toy` and `mini_block`, then layer a wheel path on top. Treat `qpm-cli` as an optional second-stage capability: bootstrap should detect it, continue when present, and stop with explicit guidance when absent.

**Tech Stack:** Bash, Python 3.10, pytest, ONNX Runtime build outputs, Qualcomm `QAIRT/QNN SDK`, Ubuntu 24.04 system packages, Markdown

---

## Prerequisites

Before Task 1, verify these conditions on the host machine.

- The validated ORT/QNN build tree exists at `ort_qnn/artifacts/ort/build/linux_qnn/Release`
- The build tree already exposes `QNNExecutionProvider` via direct `PYTHONPATH` import
- `toy.onnx` and `mini_block.onnx` already exist under `smoke/artifacts/onnx/`
- The QAIRT runtime exists at `/opt/qcom/aistack/qairt/2.41.0.251128`
- The QAIRT Python 3.10 environment exists at `/opt/qcom/qairt-py310`

Run:

```bash
mkdir -p ort_qnn/package ort_qnn/artifacts/package/portable ort_qnn/artifacts/package/wheels ort_qnn/tests scripts
```

Expected: packaging directories exist and the repo is ready for new scripts/tests.

## File Structure

- Create: `ort_qnn/package/package_manifest.py` — shared manifest and path helper logic for portable and wheel packaging
- Create: `ort_qnn/package/verify_package.py` — bundle/wheel verification helper that reuses existing runtime runners
- Create: `ort_qnn/package/build_portable_bundle.sh` — package the validated build tree into a reusable directory
- Create: `ort_qnn/package/build_python_wheel.sh` — create a Python-installable package artifact from the validated build tree
- Create: `scripts/bootstrap_ubuntu2404.sh` — rerunnable Ubuntu 24.04 environment bootstrap with `qpm-cli` detection/fallback
- Modify: `README.md` — add stronger-host bootstrap and packaging entrypoints
- Modify: `ort_qnn/README.md` — add portable bundle, wheel, and verification instructions
- Create: `ort_qnn/tests/test_package_manifest.py` — manifest/path helper tests
- Create: `ort_qnn/tests/test_verify_package.py` — package verification helper tests
- Create: `ort_qnn/tests/test_bootstrap_script_layout.py` — shell-script structure and branch messaging tests

### Task 1: Add packaging manifest helpers and their tests

**Files:**
- Create: `ort_qnn/package/package_manifest.py`
- Create: `ort_qnn/tests/test_package_manifest.py`

- [ ] **Step 1: Write the failing manifest tests**

```python
# ort_qnn/tests/test_package_manifest.py
from pathlib import Path

from ort_qnn.package.package_manifest import PackageSpec, manifest_dict, required_runtime_files


def test_required_runtime_files_include_qnn_provider_and_pybind_module():
    spec = PackageSpec(
        release_root=Path("ort_qnn/artifacts/ort/build/linux_qnn/Release"),
        qairt_root=Path("/opt/qcom/aistack/qairt/2.41.0.251128"),
        ort_version="1.23.2",
        qairt_version="2.41.0",
    )

    files = required_runtime_files(spec)

    assert Path("onnxruntime/capi/onnxruntime_pybind11_state.so") in files
    assert Path("onnxruntime/capi/libonnxruntime_providers_qnn.so") in files
    assert Path("onnxruntime/capi/libQnnCpu.so") in files


def test_manifest_dict_records_versions_and_expected_env_vars():
    spec = PackageSpec(
        release_root=Path("ort_qnn/artifacts/ort/build/linux_qnn/Release"),
        qairt_root=Path("/opt/qcom/aistack/qairt/2.41.0.251128"),
        ort_version="1.23.2",
        qairt_version="2.41.0",
    )

    manifest = manifest_dict(spec)

    assert manifest["ort_version"] == "1.23.2"
    assert manifest["qairt_version"] == "2.41.0"
    assert "PYTHONPATH" in manifest["required_env"]
    assert "LD_LIBRARY_PATH" in manifest["required_env"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
python3 -m pytest ort_qnn/tests/test_package_manifest.py -q
```

Expected: FAIL with `ModuleNotFoundError` for `ort_qnn.package.package_manifest`.

- [ ] **Step 3: Write the minimal implementation**

```python
# ort_qnn/package/package_manifest.py
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PackageSpec:
    release_root: Path
    qairt_root: Path
    ort_version: str
    qairt_version: str


def required_runtime_files(spec: PackageSpec) -> list[Path]:
    return [
        Path("onnxruntime/capi/onnxruntime_pybind11_state.so"),
        Path("onnxruntime/capi/libonnxruntime.so.1.23.2"),
        Path("onnxruntime/capi/libonnxruntime_providers_qnn.so"),
        Path("onnxruntime/capi/libonnxruntime_providers_shared.so"),
        Path("onnxruntime/capi/libQnnCpu.so"),
        Path("onnxruntime/capi/libQnnSystem.so"),
    ]


def manifest_dict(spec: PackageSpec) -> dict:
    return {
        "ort_version": spec.ort_version,
        "qairt_version": spec.qairt_version,
        "release_root": str(spec.release_root),
        "qairt_root": str(spec.qairt_root),
        "required_env": {
            "PYTHONPATH": "<bundle_root>",
            "LD_LIBRARY_PATH": "<bundle_root>/onnxruntime/capi",
        },
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
python3 -m pytest ort_qnn/tests/test_package_manifest.py -q
```

Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add ort_qnn/package/package_manifest.py ort_qnn/tests/test_package_manifest.py
git commit -m "test: add ort qnn packaging manifest helpers"
```

### Task 2: Add portable bundle verification helpers and packaging script

**Files:**
- Create: `ort_qnn/package/verify_package.py`
- Create: `ort_qnn/package/build_portable_bundle.sh`
- Create: `ort_qnn/tests/test_verify_package.py`

- [ ] **Step 1: Write the failing verification tests**

```python
# ort_qnn/tests/test_verify_package.py
from pathlib import Path

from ort_qnn.package.verify_package import BundleSpec, missing_runtime_files


def test_missing_runtime_files_reports_absent_qnn_libraries(tmp_path: Path):
    spec = BundleSpec(bundle_root=tmp_path)

    missing = missing_runtime_files(spec)

    assert Path("onnxruntime/capi/onnxruntime_pybind11_state.so") in missing
    assert Path("onnxruntime/capi/libonnxruntime_providers_qnn.so") in missing


def test_missing_runtime_files_is_empty_for_minimal_valid_layout(tmp_path: Path):
    capi = tmp_path / "onnxruntime" / "capi"
    capi.mkdir(parents=True)
    for name in [
        "onnxruntime_pybind11_state.so",
        "libonnxruntime.so.1.23.2",
        "libonnxruntime_providers_qnn.so",
        "libonnxruntime_providers_shared.so",
        "libQnnCpu.so",
        "libQnnSystem.so",
    ]:
        (capi / name).write_text("ok")

    spec = BundleSpec(bundle_root=tmp_path)

    assert missing_runtime_files(spec) == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
python3 -m pytest ort_qnn/tests/test_verify_package.py -q
```

Expected: FAIL with `ModuleNotFoundError` for `ort_qnn.package.verify_package`.

- [ ] **Step 3: Write the minimal implementation**

```python
# ort_qnn/package/verify_package.py
from dataclasses import dataclass
from pathlib import Path

from ort_qnn.package.package_manifest import PackageSpec, required_runtime_files


@dataclass(frozen=True)
class BundleSpec:
    bundle_root: Path


REQUIRED_NAMES = [
    "onnxruntime/capi/onnxruntime_pybind11_state.so",
    "onnxruntime/capi/libonnxruntime.so.1.23.2",
    "onnxruntime/capi/libonnxruntime_providers_qnn.so",
    "onnxruntime/capi/libonnxruntime_providers_shared.so",
    "onnxruntime/capi/libQnnCpu.so",
    "onnxruntime/capi/libQnnSystem.so",
]


def missing_runtime_files(spec: BundleSpec) -> list[Path]:
    missing: list[Path] = []
    for rel in REQUIRED_NAMES:
        rel_path = Path(rel)
        if not (spec.bundle_root / rel_path).exists():
            missing.append(rel_path)
    return missing
```

```bash
# ort_qnn/package/build_portable_bundle.sh
#!/usr/bin/env bash
set -euo pipefail

: "${ORT_RELEASE_ROOT:?Set ORT_RELEASE_ROOT}"
: "${PORTABLE_OUT_DIR:?Set PORTABLE_OUT_DIR}"

rm -rf "$PORTABLE_OUT_DIR"
mkdir -p "$PORTABLE_OUT_DIR"
cp -a "$ORT_RELEASE_ROOT/onnxruntime" "$PORTABLE_OUT_DIR/"
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
python3 -m pytest ort_qnn/tests/test_verify_package.py -q
```

Expected: `2 passed`.

- [ ] **Step 5: Real portable bundle verification**

Run:

```bash
export ORT_RELEASE_ROOT=/project/mobile_anima_npu/ort_qnn/artifacts/ort/build/linux_qnn/Release
export PORTABLE_OUT_DIR=/project/mobile_anima_npu/ort_qnn/artifacts/package/portable
bash ort_qnn/package/build_portable_bundle.sh

PYTHONPATH=/project/mobile_anima_npu/ort_qnn/artifacts/package/portable:/project/mobile_anima_npu \
LD_LIBRARY_PATH=/project/mobile_anima_npu/ort_qnn/artifacts/package/portable/onnxruntime/capi:/opt/python3.10.19/lib:/opt/qcom/aistack/qairt/2.41.0.251128/lib/x86_64-linux-clang \
/opt/qcom/qairt-py310/bin/python - <<'PY'
import onnxruntime as ort
print(ort.get_available_providers())
PY
```

Expected: provider list includes `QNNExecutionProvider`.

- [ ] **Step 6: Commit**

```bash
git add ort_qnn/package/verify_package.py ort_qnn/package/build_portable_bundle.sh ort_qnn/tests/test_verify_package.py
git commit -m "feat: add ort qnn portable bundle packaging"
```

### Task 3: Add wheel/package builder from the validated build tree

**Files:**
- Create: `ort_qnn/package/build_python_wheel.sh`
- Modify: `ort_qnn/package/package_manifest.py` if wheel metadata helpers are needed

- [ ] **Step 1: Extend manifest tests with wheel staging metadata**

Append to `ort_qnn/tests/test_package_manifest.py`:

```python
def test_manifest_dict_includes_package_identity():
    spec = PackageSpec(
        release_root=Path("ort_qnn/artifacts/ort/build/linux_qnn/Release"),
        qairt_root=Path("/opt/qcom/aistack/qairt/2.41.0.251128"),
        ort_version="1.23.2",
        qairt_version="2.41.0",
    )

    manifest = manifest_dict(spec)

    assert manifest["package_name"] == "onnxruntime-qnn-local"
    assert manifest["package_version"] == "1.23.2"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
python3 -m pytest ort_qnn/tests/test_package_manifest.py -q
```

Expected: FAIL because package identity keys do not exist yet.

- [ ] **Step 3: Write the minimal implementation**

Update `manifest_dict()` in `ort_qnn/package/package_manifest.py` to include:

```python
        "package_name": "onnxruntime-qnn-local",
        "package_version": spec.ort_version,
```

Create `ort_qnn/package/build_python_wheel.sh` with these responsibilities:

```bash
#!/usr/bin/env bash
set -euo pipefail

: "${ORT_RELEASE_ROOT:?Set ORT_RELEASE_ROOT}"
: "${WHEEL_STAGE_DIR:?Set WHEEL_STAGE_DIR}"
: "${WHEEL_OUT_DIR:?Set WHEEL_OUT_DIR}"
: "${PYTHON310:?Set PYTHON310}"

rm -rf "$WHEEL_STAGE_DIR"
mkdir -p "$WHEEL_STAGE_DIR/src"
cp -a "$ORT_RELEASE_ROOT/onnxruntime" "$WHEEL_STAGE_DIR/src/"
cat > "$WHEEL_STAGE_DIR/pyproject.toml" <<'TOML'
[build-system]
requires = ["setuptools>=69", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "onnxruntime-qnn-local"
version = "1.23.2"
requires-python = ">=3.10,<3.11"
TOML
cat > "$WHEEL_STAGE_DIR/setup.py" <<'PY'
from setuptools import find_packages, setup
setup(
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    package_data={"onnxruntime": ["capi/*", "capi/training/*"]},
)
PY
"$PYTHON310" -m pip install --upgrade build wheel
"$PYTHON310" -m build --wheel --outdir "$WHEEL_OUT_DIR" "$WHEEL_STAGE_DIR"
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
python3 -m pytest ort_qnn/tests/test_package_manifest.py -q
```

Expected: all manifest tests pass.

- [ ] **Step 5: Real wheel/package verification**

Run:

```bash
export ORT_RELEASE_ROOT=/project/mobile_anima_npu/ort_qnn/artifacts/ort/build/linux_qnn/Release
export WHEEL_STAGE_DIR=/project/mobile_anima_npu/ort_qnn/artifacts/package/wheel_staging
export WHEEL_OUT_DIR=/project/mobile_anima_npu/ort_qnn/artifacts/package/wheels
export PYTHON310=/opt/qcom/qairt-py310/bin/python
bash ort_qnn/package/build_python_wheel.sh

WHEEL=$(find ort_qnn/artifacts/package/wheels -name 'onnxruntime_qnn_local-*.whl' -o -name 'onnxruntime_qnn_local*.whl' | head -n 1)
/opt/qcom/qairt-py310/bin/python -m pip install --force-reinstall "$WHEEL"
```

Then verify:

```bash
PYTHONPATH= LD_LIBRARY_PATH=/opt/python3.10.19/lib:/opt/qcom/aistack/qairt/2.41.0.251128/lib/x86_64-linux-clang \
/opt/qcom/qairt-py310/bin/python - <<'PY'
import onnxruntime as ort
print(ort.get_available_providers())
PY
```

Expected: provider list includes `QNNExecutionProvider`. If it does not, classify the failure explicitly in docs and keep the portable bundle as the primary artifact.

- [ ] **Step 6: Commit**

```bash
git add ort_qnn/package/build_python_wheel.sh ort_qnn/package/package_manifest.py ort_qnn/tests/test_package_manifest.py
git commit -m "feat: add ort qnn wheel packaging path"
```

### Task 4: Add Ubuntu 24.04 bootstrap script with staged `qpm-cli` handling

**Files:**
- Create: `scripts/bootstrap_ubuntu2404.sh`
- Create: `ort_qnn/tests/test_bootstrap_script_layout.py`

- [ ] **Step 1: Write the failing bootstrap-layout tests**

```python
# ort_qnn/tests/test_bootstrap_script_layout.py
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
python3 -m pytest ort_qnn/tests/test_bootstrap_script_layout.py -q
```

Expected: FAIL because the script does not exist yet.

- [ ] **Step 3: Write the minimal implementation**

Create `scripts/bootstrap_ubuntu2404.sh` with these properties:

- `set -euo pipefail`
- checks Ubuntu 24.04 explicitly
- installs public packages via `apt-get`
- prepares Python 3.10 build/runtime prerequisites
- prepares Android base prerequisites and environment variable hints
- checks `command -v qpm-cli`
- if absent, prints a clear rerun message and exits `0`
- if present, checks login status and only then attempts `license-activate` and `install`
- validates the expected QAIRT root after install

Example structure:

```bash
#!/usr/bin/env bash
set -euo pipefail

echo "[bootstrap] installing Ubuntu 24.04 prerequisites"
apt-get update
apt-get install -y build-essential clang cmake ninja-build git curl unzip zip pkg-config openjdk-21-jdk-headless libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev libffi-dev liblzma-dev libc++1 libc++abi1

echo "[bootstrap] preparing Android base environment"
# install or validate command-line tools here

if ! command -v qpm-cli >/dev/null 2>&1; then
  echo "[bootstrap] qpm-cli not detected. Install qpm-cli, complete login, and rerun this script to finish QAIRT setup."
  exit 0
fi

if ! qpm-cli --check-login; then
  echo "[bootstrap] qpm-cli is installed but not logged in. Complete login, then rerun this script."
  exit 0
fi

qpm-cli --license-activate qualcomm_ai_runtime_sdk || true
qpm-cli --install qualcomm_ai_runtime_sdk --version 2.41.0.251128
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
python3 -m pytest ort_qnn/tests/test_bootstrap_script_layout.py -q
```

Expected: all bootstrap layout tests pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/bootstrap_ubuntu2404.sh ort_qnn/tests/test_bootstrap_script_layout.py
git commit -m "feat: add ubuntu bootstrap for ort qnn hosts"
```

### Task 5: Update README files and final verification workflow

**Files:**
- Modify: `README.md`
- Modify: `ort_qnn/README.md`
- Optional: `docs/manual/bootstrap-ubuntu2404.md`

- [ ] **Step 1: Update docs for bundle, wheel, and bootstrap flows**

`README.md` must cover:

- stronger-host bootstrap entrypoint
- how the portable bundle and wheel outputs differ
- how to rerun bootstrap after manual `qpm-cli` installation

`ort_qnn/README.md` must cover:

- how to build the portable bundle
- how to verify it using `toy` and `mini_block`
- how to build the wheel/package path
- how to interpret provider/session/execute failures

- [ ] **Step 2: Run targeted docs/template tests if any were added**

Run:

```bash
python3 -m pytest ort_qnn/tests/test_package_manifest.py ort_qnn/tests/test_verify_package.py ort_qnn/tests/test_bootstrap_script_layout.py -q
```

Expected: all targeted tests pass.

- [ ] **Step 3: Run end-to-end fresh verification evidence**

Run:

```bash
python3 -m pytest ort_qnn/tests -q

PYTHONPATH=/project/mobile_anima_npu/ort_qnn/artifacts/package/portable:/project/mobile_anima_npu \
LD_LIBRARY_PATH=/project/mobile_anima_npu/ort_qnn/artifacts/package/portable/onnxruntime/capi:/opt/python3.10.19/lib:/opt/qcom/aistack/qairt/2.41.0.251128/lib/x86_64-linux-clang \
/opt/qcom/qairt-py310/bin/python - <<'PY'
import json
import onnxruntime as ort
from ort_qnn.runtime.run_toy_ort_qnn import run_toy_once
from ort_qnn.runtime.run_mini_block_ort_qnn import run_mini_block_once
backend = "/opt/qcom/aistack/qairt/2.41.0.251128/lib/x86_64-linux-clang/libQnnCpu.so"
print(json.dumps({"providers": ort.get_available_providers()}))
print(json.dumps({"toy": run_toy_once(ort, "smoke/artifacts/onnx/toy.onnx", backend, "ort_qnn/artifacts/profiles/toy_ort_qnn.csv").__dict__}))
print(json.dumps({"mini_block": run_mini_block_once(ort, "smoke/artifacts/onnx/mini_block.onnx", backend, "ort_qnn/artifacts/profiles/mini_block_ort_qnn.csv").__dict__}))
PY

git status --short
```

Expected:

- ORT/QNN tests pass
- provider list includes `QNNExecutionProvider`
- `toy` and `mini_block` both pass
- only intentional file changes remain in `git status`

- [ ] **Step 4: Commit**

```bash
git add README.md ort_qnn/README.md docs/manual/bootstrap-ubuntu2404.md 2>/dev/null || true
git add README.md ort_qnn/README.md
# add optional doc if created
git commit -m "docs: add ort qnn packaging and bootstrap guides"
```

## Self-Review

- Spec coverage checked: the plan covers portable bundle packaging, wheel/package path, Ubuntu 24.04 bootstrap, staged `qpm-cli` behavior, and documentation updates.
- Placeholder scan checked: no `TODO`, `TBD`, or vague “figure this out later” tasks remain.
- Type consistency checked: `PackageSpec`, `BundleSpec`, `required_runtime_files`, `manifest_dict`, and `missing_runtime_files` are reused consistently across tests and implementation tasks.
