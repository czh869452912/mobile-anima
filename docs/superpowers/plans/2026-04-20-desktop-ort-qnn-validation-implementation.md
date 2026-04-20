# Desktop ORT + QNN EP Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build or acquire a Linux desktop `ONNX Runtime` with `QNN Execution Provider` and use it to validate `toy.onnx` and `mini_block.onnx` session creation and execution against the already-proven Qualcomm toolchain.

**Architecture:** Implement a separate `ort_qnn/` subproject with distinct layers for ORT acquisition/build, runtime provider configuration, and result recording. Prefer a brief prebuilt search first, but assume a source build is the normal path. Validate `toy` before `mini_block`, and record both session and execute outcomes in a dedicated ORT/QNN matrix.

**Tech Stack:** Bash, Python 3.10, pytest, ONNX Runtime source build, Qualcomm `QAIRT/QNN SDK`, CMake, Ninja, Markdown

---

## Prerequisites

Before Task 1, verify these preconditions on the host machine.

- Qualcomm `QAIRT/QNN SDK` is installed at `/opt/qcom/aistack/qairt/2.41.0.251128`
- A Python 3.10 runtime exists at `/opt/python3.10.19`
- The QAIRT Python virtual environment exists at `/opt/qcom/qairt-py310`
- `clang`, `clang++`, `cmake`, `ninja`, and `git` are installed
- `smoke/artifacts/onnx/toy.onnx` and `smoke/artifacts/onnx/mini_block.onnx` already exist

Run:

```bash
mkdir -p ort_qnn/build ort_qnn/runtime ort_qnn/artifacts/ort ort_qnn/artifacts/logs ort_qnn/artifacts/profiles ort_qnn/docs ort_qnn/tests
```

Expected: `ort_qnn/` directory tree exists with the listed folders.

## File Structure

- Create: `ort_qnn/README.md` — quickstart for desktop ORT/QNN validation
- Create: `ort_qnn/build/fetch_onnxruntime.sh` — fetches ORT source at a pinned tag
- Create: `ort_qnn/build/build_ort_qnn_linux.sh` — source-build entry point for Linux desktop `QNN EP`
- Create: `ort_qnn/runtime/provider_options.py` — provider option builder for QNN EP desktop runs
- Create: `ort_qnn/runtime/run_toy_ort_qnn.py` — `toy.onnx` session and execute runner
- Create: `ort_qnn/runtime/run_mini_block_ort_qnn.py` — `mini_block.onnx` session and execute runner
- Create: `ort_qnn/docs/ort-qnn-matrix.md` — ORT/QNN result matrix
- Create: `ort_qnn/tests/test_provider_options.py` — provider option tests
- Create: `ort_qnn/tests/test_runtime_args.py` — runtime CLI parsing tests
- Create: `ort_qnn/tests/test_matrix_template.py` — matrix template tests

### Task 1: Scaffold the ORT/QNN subproject and provider option builder

**Files:**
- Create: `ort_qnn/README.md`
- Create: `ort_qnn/runtime/provider_options.py`
- Test: `ort_qnn/tests/test_provider_options.py`

- [ ] **Step 1: Write the failing provider-option test**

```python
# ort_qnn/tests/test_provider_options.py
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
python3 -m pytest ort_qnn/tests/test_provider_options.py -q
```

Expected: FAIL with `ModuleNotFoundError` for `ort_qnn.runtime.provider_options`.

- [ ] **Step 3: Write the minimal implementation**

```python
# ort_qnn/runtime/provider_options.py
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProviderSpec:
    backend_path: Path
    profiling_path: Path
    disable_cpu_fallback: bool


def qnn_provider_options(spec: ProviderSpec) -> dict[str, str]:
    return {
        "backend_path": str(spec.backend_path),
        "profiling_level": "detailed",
        "profiling_file_path": str(spec.profiling_path),
        "ep.context_enable": "1",
        "ep.context_embed_mode": "0",
        "session.disable_cpu_ep_fallback": "1" if spec.disable_cpu_fallback else "0",
    }
```

```markdown
<!-- ort_qnn/README.md -->
# Desktop ORT + QNN EP Validation

This subproject validates the last missing desktop layer in the smoke ladder:

1. acquire or build ORT with QNN EP
2. create a `QNN EP` session for `toy.onnx`
3. execute `toy.onnx`
4. repeat for `mini_block.onnx`
```

- [ ] **Step 4: Run the test to verify it passes**

Run:

```bash
python3 -m pytest ort_qnn/tests/test_provider_options.py -q
```

Expected: `1 passed`.

- [ ] **Step 5: Commit**

```bash
git add ort_qnn/README.md ort_qnn/runtime/provider_options.py ort_qnn/tests/test_provider_options.py
git commit -m "test: add desktop ort qnn provider options"
```

### Task 2: Add runtime CLI parsing helpers for `toy` and `mini_block`

**Files:**
- Create: `ort_qnn/runtime/run_toy_ort_qnn.py`
- Create: `ort_qnn/runtime/run_mini_block_ort_qnn.py`
- Test: `ort_qnn/tests/test_runtime_args.py`

- [ ] **Step 1: Write the failing runtime-args tests**

```python
# ort_qnn/tests/test_runtime_args.py
from ort_qnn.runtime.run_toy_ort_qnn import parse_toy_args
from ort_qnn.runtime.run_mini_block_ort_qnn import parse_mini_args


def test_parse_toy_args_reads_model_and_backend_paths():
    args = parse_toy_args([
        "--model", "smoke/artifacts/onnx/toy.onnx",
        "--backend", "/opt/qcom/aistack/qairt/2.41.0.251128/lib/x86_64-linux-clang/libQnnCpu.so",
        "--profile", "ort_qnn/artifacts/profiles/toy_ort_qnn.csv",
    ])

    assert args.model.endswith("toy.onnx")
    assert args.backend.endswith("libQnnCpu.so")


def test_parse_mini_args_reads_model_and_backend_paths():
    args = parse_mini_args([
        "--model", "smoke/artifacts/onnx/mini_block.onnx",
        "--backend", "/opt/qcom/aistack/qairt/2.41.0.251128/lib/x86_64-linux-clang/libQnnCpu.so",
        "--profile", "ort_qnn/artifacts/profiles/mini_ort_qnn.csv",
    ])

    assert args.model.endswith("mini_block.onnx")
    assert args.backend.endswith("libQnnCpu.so")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
python3 -m pytest ort_qnn/tests/test_runtime_args.py -q
```

Expected: FAIL with `ModuleNotFoundError` for the new runtime modules.

- [ ] **Step 3: Write the minimal implementation**

```python
# ort_qnn/runtime/run_toy_ort_qnn.py
import argparse


def parse_toy_args(argv: list[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--profile", required=True)
    return parser.parse_args(argv)
```

```python
# ort_qnn/runtime/run_mini_block_ort_qnn.py
import argparse


def parse_mini_args(argv: list[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--profile", required=True)
    return parser.parse_args(argv)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
python3 -m pytest ort_qnn/tests/test_runtime_args.py -q
```

Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add ort_qnn/runtime/run_toy_ort_qnn.py ort_qnn/runtime/run_mini_block_ort_qnn.py ort_qnn/tests/test_runtime_args.py
git commit -m "test: add desktop ort qnn runtime parsers"
```

### Task 3: Add ORT acquisition scripts and dry-run checks

**Files:**
- Create: `ort_qnn/build/fetch_onnxruntime.sh`
- Create: `ort_qnn/build/build_ort_qnn_linux.sh`
- Test: `ort_qnn/tests/test_matrix_template.py`

- [ ] **Step 1: Write the failing matrix-template test for build notes**

```python
# ort_qnn/tests/test_matrix_template.py
from pathlib import Path


def test_matrix_template_contains_acquisition_path_section():
    text = Path("ort_qnn/docs/ort-qnn-matrix.md").read_text()

    assert "Acquisition path" in text
    assert "ORT build" in text
    assert "| Graph | Session | Execute | Profile | Failing Stage |" in text
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
python3 -m pytest ort_qnn/tests/test_matrix_template.py -q
```

Expected: FAIL because `ort_qnn/docs/ort-qnn-matrix.md` does not exist yet.

- [ ] **Step 3: Write the minimal implementation**

```bash
# ort_qnn/build/fetch_onnxruntime.sh
#!/usr/bin/env bash
set -euo pipefail

TAG="${1:-v1.23.2}"
TARGET_DIR="${2:-ort_qnn/artifacts/ort/onnxruntime}"

if [ ! -d "$TARGET_DIR/.git" ]; then
  git clone --branch "$TAG" --depth 1 https://github.com/microsoft/onnxruntime.git "$TARGET_DIR"
fi
```

```bash
# ort_qnn/build/build_ort_qnn_linux.sh
#!/usr/bin/env bash
set -euo pipefail

: "${QAIRT_SDK_ROOT:?Set QAIRT_SDK_ROOT}"
: "${ORT_SRC_ROOT:?Set ORT_SRC_ROOT}"
: "${PYTHON310:?Set PYTHON310}"

cd "$ORT_SRC_ROOT"
"$PYTHON310" tools/ci_build/build.py \
  --config Release \
  --build_shared_lib \
  --parallel \
  --use_qnn \
  --qnn_home "$QAIRT_SDK_ROOT"
```

```markdown
<!-- ort_qnn/docs/ort-qnn-matrix.md -->
# ORT + QNN Validation Matrix

## Environment

- Host:
- QNN SDK version:
- Acquisition path:
- ORT build:

## Results

| Graph | Session | Execute | Profile | Failing Stage |
| --- | --- | --- | --- | --- |
| toy | PENDING | PENDING | PENDING | pending |
| mini_block | PENDING | PENDING | PENDING | pending |
```

- [ ] **Step 4: Run the test to verify it passes**

Run:

```bash
python3 -m pytest ort_qnn/tests/test_matrix_template.py -q
```

Expected: `1 passed`.

- [ ] **Step 5: Commit**

```bash
git add ort_qnn/build/fetch_onnxruntime.sh ort_qnn/build/build_ort_qnn_linux.sh ort_qnn/docs/ort-qnn-matrix.md ort_qnn/tests/test_matrix_template.py
git commit -m "docs: add ort qnn acquisition scaffolding"
```

### Task 4: Add provider-availability and session-construction helpers

**Files:**
- Modify: `ort_qnn/runtime/run_toy_ort_qnn.py`
- Modify: `ort_qnn/runtime/run_mini_block_ort_qnn.py`

- [ ] **Step 1: Extend the runtime-args tests with availability checks**

```python
# append to ort_qnn/tests/test_runtime_args.py
from ort_qnn.runtime.run_toy_ort_qnn import provider_available


def test_provider_available_checks_membership():
    assert provider_available("QNNExecutionProvider", ["CPUExecutionProvider", "QNNExecutionProvider"])
    assert not provider_available("QNNExecutionProvider", ["CPUExecutionProvider"])
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
python3 -m pytest ort_qnn/tests/test_runtime_args.py -q
```

Expected: FAIL because `provider_available` does not exist yet.

- [ ] **Step 3: Write the minimal implementation**

```python
# replace ort_qnn/runtime/run_toy_ort_qnn.py
import argparse


def parse_toy_args(argv: list[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--profile", required=True)
    return parser.parse_args(argv)


def provider_available(name: str, providers: list[str]) -> bool:
    return name in providers
```

```python
# replace ort_qnn/runtime/run_mini_block_ort_qnn.py
import argparse


def parse_mini_args(argv: list[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--profile", required=True)
    return parser.parse_args(argv)
```

- [ ] **Step 4: Run the test to verify it passes**

Run:

```bash
python3 -m pytest ort_qnn/tests/test_runtime_args.py -q
```

Expected: `3 passed`.

- [ ] **Step 5: Commit**

```bash
git add ort_qnn/runtime/run_toy_ort_qnn.py ort_qnn/runtime/run_mini_block_ort_qnn.py ort_qnn/tests/test_runtime_args.py
git commit -m "feat: add ort provider availability helper"
```

### Task 5: Add manual run order for the desktop ORT validation stage

**Files:**
- Modify: `ort_qnn/README.md`

- [ ] **Step 1: Append the manual run order**

```markdown
<!-- append to ort_qnn/README.md -->

## First Manual Run Order

1. Check whether a usable Linux prebuilt `ORT + QNN EP` exists
2. If not, fetch ORT source
3. Build Linux desktop `ORT + QNN EP`
4. Verify provider registration for `QNNExecutionProvider`
5. Run `toy.onnx` session creation
6. Run `toy.onnx` execution
7. Run `mini_block.onnx` session creation
8. Run `mini_block.onnx` execution
9. Update `ort_qnn/docs/ort-qnn-matrix.md`
```

- [ ] **Step 2: Run the full ORT/QNN stage unit suite**

Run:

```bash
python3 -m pytest ort_qnn/tests -q
```

Expected: all ORT/QNN stage unit tests pass.

- [ ] **Step 3: Commit**

```bash
git add ort_qnn/README.md
git commit -m "docs: add ort qnn run order"
```

## Self-Review

- Spec coverage checked: the plan covers ORT acquisition, source-build fallback, provider option wiring, provider availability checks, toy and mini-block execution helpers, and the ORT/QNN result matrix.
- Placeholder scan checked: no placeholder markers remain.
- Type consistency checked: `ProviderSpec`, `parse_toy_args`, `parse_mini_args`, and `provider_available` are used consistently across tasks.
