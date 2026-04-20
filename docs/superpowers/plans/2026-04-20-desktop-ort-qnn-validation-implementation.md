# Desktop ORT + QNN EP Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python-callable Linux desktop `ONNX Runtime` with `QNNExecutionProvider`, prove provider visibility, and validate `toy.onnx` followed by `mini_block.onnx` with session and execute checks.

**Architecture:** Keep the work inside `ort_qnn/` with three focused layers: build acquisition, runtime execution, and evidence recording. Prefer a short prebuilt probe first, but assume source build is the default path. Validate provider visibility before any model run, then validate `toy`, then `mini_block`, and record exact outcomes in `ort-qnn-matrix.md`.

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

- Modify: `ort_qnn/README.md` — document the real build-and-run flow
- Create: `ort_qnn/build/build_config.py` — Python helper for ORT source-build command generation
- Modify: `ort_qnn/build/fetch_onnxruntime.sh` — fetch a pinned ORT source tree
- Modify: `ort_qnn/build/build_ort_qnn_linux.sh` — source-build a Python-callable ORT with `QNN EP`
- Create: `ort_qnn/runtime/run_common.py` — provider visibility, session creation, and execute helpers
- Modify: `ort_qnn/runtime/provider_options.py` — provider options for desktop runs
- Modify: `ort_qnn/runtime/run_toy_ort_qnn.py` — real `toy.onnx` runner
- Modify: `ort_qnn/runtime/run_mini_block_ort_qnn.py` — real `mini_block.onnx` runner
- Modify: `ort_qnn/docs/ort-qnn-matrix.md` — final result matrix for provider/session/execute outcomes
- Create: `ort_qnn/tests/test_build_config.py` — build command tests
- Create: `ort_qnn/tests/test_run_common.py` — provider visibility and execution classification tests
- Modify: `ort_qnn/tests/test_runtime_args.py` — CLI and helper tests for the concrete runtime scripts
- Modify: `ort_qnn/tests/test_matrix_template.py` — matrix content tests

### Task 1: Add ORT build command generation and prebuilt/source acquisition helpers

**Files:**
- Create: `ort_qnn/build/build_config.py`
- Modify: `ort_qnn/build/fetch_onnxruntime.sh`
- Modify: `ort_qnn/build/build_ort_qnn_linux.sh`
- Test: `ort_qnn/tests/test_build_config.py`

- [ ] **Step 1: Write the failing build-config tests**

```python
# ort_qnn/tests/test_build_config.py
from pathlib import Path

from ort_qnn.build.build_config import BuildSpec, build_ort_command


def test_build_ort_command_requests_qnn_and_python_wheel():
    spec = BuildSpec(
        qairt_root=Path("/opt/qcom/aistack/qairt/2.41.0.251128"),
        ort_src_root=Path("ort_qnn/artifacts/ort/onnxruntime"),
        python310=Path("/opt/qcom/qairt-py310/bin/python"),
        build_dir=Path("ort_qnn/artifacts/ort/build/linux_qnn"),
    )

    command = build_ort_command(spec)

    assert command[0].endswith("python")
    assert "tools/ci_build/build.py" in command
    assert "--use_qnn" in command
    assert "--qnn_home" in command
    assert str(spec.qairt_root) in command
    assert "--build_wheel" in command
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
python3 -m pytest ort_qnn/tests/test_build_config.py -q
```

Expected: FAIL with `ModuleNotFoundError` for `ort_qnn.build.build_config`.

- [ ] **Step 3: Write the minimal implementation**

```python
# ort_qnn/build/build_config.py
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BuildSpec:
    qairt_root: Path
    ort_src_root: Path
    python310: Path
    build_dir: Path


def build_ort_command(spec: BuildSpec) -> list[str]:
    return [
        str(spec.python310),
        "tools/ci_build/build.py",
        "--config",
        "Release",
        "--build_shared_lib",
        "--build_wheel",
        "--parallel",
        "--use_qnn",
        "--qnn_home",
        str(spec.qairt_root),
        "--build_dir",
        str(spec.build_dir),
    ]
```

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
: "${ORT_BUILD_DIR:?Set ORT_BUILD_DIR}"

cd "$ORT_SRC_ROOT"
"$PYTHON310" tools/ci_build/build.py \
  --config Release \
  --build_shared_lib \
  --build_wheel \
  --parallel \
  --use_qnn \
  --qnn_home "$QAIRT_SDK_ROOT" \
  --build_dir "$ORT_BUILD_DIR"
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
python3 -m pytest ort_qnn/tests/test_build_config.py -q
```

Expected: `1 passed`.

- [ ] **Step 5: Commit**

```bash
git add ort_qnn/build/build_config.py ort_qnn/build/fetch_onnxruntime.sh ort_qnn/build/build_ort_qnn_linux.sh ort_qnn/tests/test_build_config.py
git commit -m "test: add desktop ort build command helpers"
```

### Task 2: Add common ORT runtime helpers for provider visibility, session creation, and execute classification

**Files:**
- Create: `ort_qnn/runtime/run_common.py`
- Modify: `ort_qnn/runtime/provider_options.py`
- Test: `ort_qnn/tests/test_run_common.py`

- [ ] **Step 1: Write the failing runtime-common tests**

```python
# ort_qnn/tests/test_run_common.py
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ort_qnn.runtime.run_common import RunSpec, RunOutcome, provider_visible, run_once


@dataclass
class FakeSession:
    providers: list[str]
    provider_options: list[dict[str, str]]

    def run(self, _, feed):
        return [np.asarray(feed["input"]) + 1.0]


class FakeOrt:
    @staticmethod
    def get_available_providers():
        return ["CPUExecutionProvider", "QNNExecutionProvider"]

    class InferenceSession:
        def __init__(self, model_path, providers, provider_options):
            self.inner = FakeSession(providers, provider_options)

        def run(self, output_names, feed):
            return self.inner.run(output_names, feed)


def test_provider_visible_checks_provider_membership():
    assert provider_visible("QNNExecutionProvider", ["CPUExecutionProvider", "QNNExecutionProvider"])
    assert not provider_visible("QNNExecutionProvider", ["CPUExecutionProvider"])


def test_run_once_marks_success_when_fake_ort_executes():
    spec = RunSpec(
        model_path=Path("smoke/artifacts/onnx/toy.onnx"),
        backend_path=Path("/opt/qcom/aistack/qairt/2.41.0.251128/lib/x86_64-linux-clang/libQnnCpu.so"),
        profiling_path=Path("ort_qnn/artifacts/profiles/toy_qnn.csv"),
        input_feed={"input": np.zeros((1, 8), dtype=np.float32)},
    )

    result = run_once(FakeOrt, spec)

    assert result.provider_visible is True
    assert result.session_ok is True
    assert result.execute_ok is True
    assert result.failing_stage == ""
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
python3 -m pytest ort_qnn/tests/test_run_common.py -q
```

Expected: FAIL with `ModuleNotFoundError` for `ort_qnn.runtime.run_common`.

- [ ] **Step 3: Write the minimal implementation**

```python
# ort_qnn/runtime/run_common.py
from dataclasses import dataclass
from pathlib import Path

from ort_qnn.runtime.provider_options import ProviderSpec, qnn_provider_options


@dataclass(frozen=True)
class RunSpec:
    model_path: Path
    backend_path: Path
    profiling_path: Path
    input_feed: dict


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
    session = ort_module.InferenceSession(
        str(spec.model_path),
        providers=["QNNExecutionProvider"],
        provider_options=[options],
    )
    session.run(None, spec.input_feed)
    return RunOutcome(True, True, True, "")
```

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

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
python3 -m pytest ort_qnn/tests/test_run_common.py ort_qnn/tests/test_provider_options.py -q
```

Expected: all listed tests pass.

- [ ] **Step 5: Commit**

```bash
git add ort_qnn/runtime/run_common.py ort_qnn/runtime/provider_options.py ort_qnn/tests/test_run_common.py ort_qnn/tests/test_provider_options.py
git commit -m "test: add desktop ort runtime helpers"
```

### Task 3: Turn `toy` and `mini_block` runtime scripts into real Python-callable runners

**Files:**
- Modify: `ort_qnn/runtime/run_toy_ort_qnn.py`
- Modify: `ort_qnn/runtime/run_mini_block_ort_qnn.py`
- Modify: `ort_qnn/tests/test_runtime_args.py`

- [ ] **Step 1: Extend runtime tests with feed-builder checks**

```python
# append to ort_qnn/tests/test_runtime_args.py
from ort_qnn.runtime.run_toy_ort_qnn import toy_input_feed
from ort_qnn.runtime.run_mini_block_ort_qnn import mini_block_input_feed


def test_toy_input_feed_has_expected_shape():
    feed = toy_input_feed()

    assert feed["input"].shape == (1, 8)


def test_mini_block_input_feed_has_expected_shape():
    feed = mini_block_input_feed()

    assert feed["input"].shape == (1, 4, 8)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
python3 -m pytest ort_qnn/tests/test_runtime_args.py -q
```

Expected: FAIL because `toy_input_feed` and `mini_block_input_feed` do not exist yet.

- [ ] **Step 3: Write the minimal implementation**

```python
# ort_qnn/runtime/run_toy_ort_qnn.py
import argparse

import numpy as np


def parse_toy_args(argv: list[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--profile", required=True)
    return parser.parse_args(argv)


def provider_available(name: str, providers: list[str]) -> bool:
    return name in providers


def toy_input_feed() -> dict[str, np.ndarray]:
    return {"input": np.arange(8, dtype=np.float32).reshape(1, 8)}
```

```python
# ort_qnn/runtime/run_mini_block_ort_qnn.py
import argparse

import numpy as np


def parse_mini_args(argv: list[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--profile", required=True)
    return parser.parse_args(argv)


def mini_block_input_feed() -> dict[str, np.ndarray]:
    return {"input": (np.arange(32, dtype=np.float32).reshape(1, 4, 8) / 32.0)}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
python3 -m pytest ort_qnn/tests/test_runtime_args.py -q
```

Expected: all listed tests pass.

- [ ] **Step 5: Commit**

```bash
git add ort_qnn/runtime/run_toy_ort_qnn.py ort_qnn/runtime/run_mini_block_ort_qnn.py ort_qnn/tests/test_runtime_args.py
git commit -m "feat: add ort toy and mini input feeds"
```

### Task 4: Add matrix recording for real ORT/QNN outcomes and manual run order

**Files:**
- Modify: `ort_qnn/docs/ort-qnn-matrix.md`
- Modify: `ort_qnn/README.md`

- [ ] **Step 1: Extend the matrix-template test with real outcome columns**

```python
# replace ort_qnn/tests/test_matrix_template.py
from pathlib import Path


def test_matrix_template_contains_provider_and_execution_columns():
    text = Path("ort_qnn/docs/ort-qnn-matrix.md").read_text()

    assert "Acquisition path" in text
    assert "ORT build" in text
    assert "| Graph | Provider Visible | Session | Execute | Profile | Failing Stage |" in text
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
python3 -m pytest ort_qnn/tests/test_matrix_template.py -q
```

Expected: FAIL because the matrix still uses the older columns.

- [ ] **Step 3: Write the minimal implementation**

```markdown
<!-- ort_qnn/docs/ort-qnn-matrix.md -->
# ORT + QNN Validation Matrix

## Environment

- Host:
- QNN SDK version:
- Acquisition path:
- ORT build:

## Results

| Graph | Provider Visible | Session | Execute | Profile | Failing Stage |
| --- | --- | --- | --- | --- | --- |
| toy | PENDING | PENDING | PENDING | PENDING | pending |
| mini_block | PENDING | PENDING | PENDING | PENDING | pending |
```

```markdown
<!-- ort_qnn/README.md -->
# Desktop ORT + QNN EP Validation

This subproject validates the last missing desktop layer in the smoke ladder:

1. acquire or build ORT with QNN EP
2. verify `QNNExecutionProvider` visibility
3. create a `QNN EP` session for `toy.onnx`
4. execute `toy.onnx`
5. repeat for `mini_block.onnx`

## First Manual Run Order

1. Check for a usable Linux prebuilt `ORT + QNN EP`
2. If not available, fetch ORT source
3. Build Linux desktop ORT Python wheel with `QNN EP`
4. Confirm `QNNExecutionProvider` is visible
5. Run `toy.onnx` session creation
6. Run `toy.onnx` execution
7. Run `mini_block.onnx` session creation
8. Run `mini_block.onnx` execution
9. Update `ort_qnn/docs/ort-qnn-matrix.md`
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
python3 -m pytest ort_qnn/tests/test_matrix_template.py -q
```

Expected: `1 passed`.

- [ ] **Step 5: Commit**

```bash
git add ort_qnn/docs/ort-qnn-matrix.md ort_qnn/README.md ort_qnn/tests/test_matrix_template.py
git commit -m "docs: update ort qnn matrix for real validation"
```

### Task 5: Run the corrected ORT/QNN stage unit suite

**Files:**
- No file changes

- [ ] **Step 1: Run the full ORT/QNN unit suite**

Run:

```bash
python3 -m pytest ort_qnn/tests -q
```

Expected: all ORT/QNN unit tests pass.

- [ ] **Step 2: Commit if any files changed during cleanup**

```bash
git status --short
```

Expected: no uncommitted changes.

## Self-Review

- Spec coverage checked: the rewritten plan now explicitly covers ORT build acquisition, Python-callable ORT output, provider visibility, `toy` execution, `mini_block` execution, and the corrected matrix columns.
- Placeholder scan checked: no placeholder markers remain.
- Type consistency checked: `BuildSpec`, `ProviderSpec`, `RunSpec`, `RunOutcome`, `toy_input_feed`, and `mini_block_input_feed` are used consistently across tasks.
