# QNN Smoke Test + Mini Denoiser Block Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a desktop-first smoke-test subproject that validates `ONNX -> QNN compile` and `ORT + QNN EP` runtime execution for a toy graph and a mini transformer-like block before attempting the real `Anima` denoiser.

**Architecture:** Implement an isolated `smoke/` subproject with separate layers for graph definition, ONNX export, Qualcomm compile, and runtime execution. Validate each graph twice: first with CPU `ORT` to prove ONNX correctness, then with Qualcomm compile plus `ORT + QNN EP` to prove the NPU toolchain path. Record all outcomes in a machine-readable results artifact and a human-readable smoke matrix.

**Tech Stack:** Python 3.12, pytest, numpy, ONNX, ONNX Runtime, Qualcomm `QNN/QAIRT SDK`, shell scripts, Markdown

---

## Prerequisites

Before Task 1, verify these preconditions on the host machine.

- `python3` is available
- `pytest` is installed for `python3`
- Qualcomm SDK download entitlement is confirmed or known test credentials are available
- `gradle`, Android SDK, and JDK may remain installed from prior work but are not required for this stage's desktop scope

Run:

```bash
mkdir -p smoke/models smoke/export smoke/compile smoke/runtime smoke/artifacts/onnx smoke/artifacts/qnn smoke/artifacts/profiles smoke/artifacts/logs smoke/docs smoke/tests
```

Expected: `smoke/` directory tree exists with the listed folders.

## File Structure

- Create: `smoke/README.md` — quickstart for the smoke-test subproject
- Create: `smoke/models/__init__.py` — package marker
- Create: `smoke/models/toy_graph.py` — toy graph definition and sample input generator
- Create: `smoke/models/mini_block.py` — mini transformer-like block definition and sample input generator
- Create: `smoke/export/__init__.py` — package marker
- Create: `smoke/export/export_common.py` — shared ONNX export helpers and artifact naming
- Create: `smoke/export/export_toy_onnx.py` — toy ONNX export entry point
- Create: `smoke/export/export_mini_block_onnx.py` — mini-block ONNX export entry point
- Create: `smoke/compile/__init__.py` — package marker
- Create: `smoke/compile/compile_qnn.py` — QNN compile command builder and runner
- Create: `smoke/compile/inspect_compile_result.py` — compile-artifact inspection helpers
- Create: `smoke/runtime/__init__.py` — package marker
- Create: `smoke/runtime/run_ort_cpu.py` — CPU ORT validation runner
- Create: `smoke/runtime/run_ort_qnn.py` — ORT + QNN EP validation runner
- Create: `smoke/runtime/collect_profile.py` — result aggregation helper for profiles/logs
- Create: `smoke/docs/smoke-matrix.md` — manual result matrix template
- Create: `smoke/tests/test_toy_graph.py` — toy graph shape and sample-data tests
- Create: `smoke/tests/test_mini_block.py` — mini-block shape and exportability tests
- Create: `smoke/tests/test_export_common.py` — shared export-path and naming tests
- Create: `smoke/tests/test_compile_qnn.py` — QNN command-building tests
- Create: `smoke/tests/test_runtime_config.py` — runtime configuration and result-record tests
- Create: `smoke/tests/test_collect_profile.py` — profile/result aggregation tests

### Task 1: Scaffold the smoke subproject and graph definitions

**Files:**
- Create: `smoke/README.md`
- Create: `smoke/models/__init__.py`
- Create: `smoke/models/toy_graph.py`
- Create: `smoke/models/mini_block.py`
- Test: `smoke/tests/test_toy_graph.py`
- Test: `smoke/tests/test_mini_block.py`

- [ ] **Step 1: Write the failing graph-definition tests**

```python
# smoke/tests/test_toy_graph.py
import numpy as np

from smoke.models.toy_graph import ToyGraphSpec, build_toy_inputs


def test_build_toy_inputs_returns_expected_shapes():
    spec = ToyGraphSpec(batch=1, in_features=8, out_features=4)

    left, weight, bias = build_toy_inputs(spec)

    assert left.shape == (1, 8)
    assert weight.shape == (8, 4)
    assert bias.shape == (4,)
    assert left.dtype == np.float32
```

```python
# smoke/tests/test_mini_block.py
import numpy as np

from smoke.models.mini_block import MiniBlockSpec, build_mini_block_inputs


def test_build_mini_block_inputs_returns_expected_shapes():
    spec = MiniBlockSpec(batch=1, tokens=4, hidden_size=8)

    hidden, residual, weight = build_mini_block_inputs(spec)

    assert hidden.shape == (1, 4, 8)
    assert residual.shape == (1, 4, 8)
    assert weight.shape == (8, 8)
    assert hidden.dtype == np.float32
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
python3 -m pytest smoke/tests/test_toy_graph.py smoke/tests/test_mini_block.py -q
```

Expected: FAIL with `ModuleNotFoundError` for `smoke.models.toy_graph` and `smoke.models.mini_block`.

- [ ] **Step 3: Write the minimal implementation**

```python
# smoke/models/__init__.py
__all__ = ["toy_graph", "mini_block"]
```

```python
# smoke/models/toy_graph.py
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ToyGraphSpec:
    batch: int
    in_features: int
    out_features: int


def build_toy_inputs(spec: ToyGraphSpec):
    left = np.zeros((spec.batch, spec.in_features), dtype=np.float32)
    weight = np.zeros((spec.in_features, spec.out_features), dtype=np.float32)
    bias = np.zeros((spec.out_features,), dtype=np.float32)
    return left, weight, bias
```

```python
# smoke/models/mini_block.py
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MiniBlockSpec:
    batch: int
    tokens: int
    hidden_size: int


def build_mini_block_inputs(spec: MiniBlockSpec):
    hidden = np.zeros((spec.batch, spec.tokens, spec.hidden_size), dtype=np.float32)
    residual = np.zeros((spec.batch, spec.tokens, spec.hidden_size), dtype=np.float32)
    weight = np.zeros((spec.hidden_size, spec.hidden_size), dtype=np.float32)
    return hidden, residual, weight
```

```markdown
<!-- smoke/README.md -->
# QNN Smoke Tests

This subproject validates Qualcomm compile and runtime feasibility in the cheapest order:

1. Toy graph
2. Mini transformer-like block
3. Real `Anima` denoiser only after the first two are understood
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
python3 -m pytest smoke/tests/test_toy_graph.py smoke/tests/test_mini_block.py -q
```

Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add smoke/README.md smoke/models/__init__.py smoke/models/toy_graph.py smoke/models/mini_block.py smoke/tests/test_toy_graph.py smoke/tests/test_mini_block.py
git commit -m "test: scaffold smoke graph definitions"
```

### Task 2: Add shared ONNX export helpers and artifact naming

**Files:**
- Create: `smoke/export/__init__.py`
- Create: `smoke/export/export_common.py`
- Create: `smoke/export/export_toy_onnx.py`
- Create: `smoke/export/export_mini_block_onnx.py`
- Test: `smoke/tests/test_export_common.py`

- [ ] **Step 1: Write the failing export-helper tests**

```python
# smoke/tests/test_export_common.py
from pathlib import Path

from smoke.export.export_common import ExportArtifactSpec, onnx_output_path


def test_onnx_output_path_places_files_under_artifacts_onnx():
    spec = ExportArtifactSpec(artifact_root=Path("smoke/artifacts"), graph_name="toy")

    path = onnx_output_path(spec)

    assert path == Path("smoke/artifacts/onnx/toy.onnx")
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
python3 -m pytest smoke/tests/test_export_common.py -q
```

Expected: FAIL with `ModuleNotFoundError` for `smoke.export.export_common`.

- [ ] **Step 3: Write the minimal implementation**

```python
# smoke/export/__init__.py
__all__ = ["export_common"]
```

```python
# smoke/export/export_common.py
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExportArtifactSpec:
    artifact_root: Path
    graph_name: str


def onnx_output_path(spec: ExportArtifactSpec) -> Path:
    return spec.artifact_root / "onnx" / f"{spec.graph_name}.onnx"
```

```python
# smoke/export/export_toy_onnx.py
from pathlib import Path

from smoke.export.export_common import ExportArtifactSpec, onnx_output_path


def resolve_toy_output(artifact_root: Path) -> Path:
    return onnx_output_path(ExportArtifactSpec(artifact_root=artifact_root, graph_name="toy"))
```

```python
# smoke/export/export_mini_block_onnx.py
from pathlib import Path

from smoke.export.export_common import ExportArtifactSpec, onnx_output_path


def resolve_mini_block_output(artifact_root: Path) -> Path:
    return onnx_output_path(ExportArtifactSpec(artifact_root=artifact_root, graph_name="mini_block"))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
python3 -m pytest smoke/tests/test_export_common.py -q
```

Expected: `1 passed`.

- [ ] **Step 5: Commit**

```bash
git add smoke/export/__init__.py smoke/export/export_common.py smoke/export/export_toy_onnx.py smoke/export/export_mini_block_onnx.py smoke/tests/test_export_common.py
git commit -m "test: add smoke export artifact helpers"
```

### Task 3: Add Qualcomm compile command builder and artifact inspection

**Files:**
- Create: `smoke/compile/__init__.py`
- Create: `smoke/compile/compile_qnn.py`
- Create: `smoke/compile/inspect_compile_result.py`
- Test: `smoke/tests/test_compile_qnn.py`

- [ ] **Step 1: Write the failing compile-helper tests**

```python
# smoke/tests/test_compile_qnn.py
from pathlib import Path

from smoke.compile.compile_qnn import CompileSpec, build_compile_command
from smoke.compile.inspect_compile_result import expected_qnn_outputs


def test_build_compile_command_points_to_context_generator():
    spec = CompileSpec(
        sdk_root=Path("/opt/qairt"),
        onnx_model=Path("smoke/artifacts/onnx/toy.onnx"),
        output_dir=Path("smoke/artifacts/qnn/toy"),
        profiling_level="detailed",
    )

    command = build_compile_command(spec)

    assert "qnn-context-binary-generator" in command[0]
    assert "--model" in command
    assert str(spec.onnx_model) in command
    assert "--profiling_level" in command


def test_expected_qnn_outputs_names_context_and_profile_targets():
    outputs = expected_qnn_outputs(Path("smoke/artifacts/qnn/toy"), "toy")

    assert outputs["context"] == Path("smoke/artifacts/qnn/toy/toy_ctx.onnx")
    assert outputs["profile"] == Path("smoke/artifacts/qnn/toy/toy_profile.csv")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
python3 -m pytest smoke/tests/test_compile_qnn.py -q
```

Expected: FAIL with `ModuleNotFoundError` for `smoke.compile.compile_qnn`.

- [ ] **Step 3: Write the minimal implementation**

```python
# smoke/compile/__init__.py
__all__ = ["compile_qnn", "inspect_compile_result"]
```

```python
# smoke/compile/compile_qnn.py
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CompileSpec:
    sdk_root: Path
    onnx_model: Path
    output_dir: Path
    profiling_level: str


def build_compile_command(spec: CompileSpec) -> list[str]:
    return [
        str(spec.sdk_root / "bin" / "x86_64-linux-clang" / "qnn-context-binary-generator"),
        "--model",
        str(spec.onnx_model),
        "--backend",
        str(spec.sdk_root / "lib" / "x86_64-linux-clang" / "libQnnHtp.so"),
        "--output_dir",
        str(spec.output_dir),
        "--profiling_level",
        spec.profiling_level,
    ]
```

```python
# smoke/compile/inspect_compile_result.py
from pathlib import Path


def expected_qnn_outputs(output_dir: Path, graph_name: str) -> dict[str, Path]:
    return {
        "context": output_dir / f"{graph_name}_ctx.onnx",
        "binary": output_dir / f"{graph_name}_qnn.bin",
        "profile": output_dir / f"{graph_name}_profile.csv",
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
python3 -m pytest smoke/tests/test_compile_qnn.py -q
```

Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add smoke/compile/__init__.py smoke/compile/compile_qnn.py smoke/compile/inspect_compile_result.py smoke/tests/test_compile_qnn.py
git commit -m "test: add qnn compile command scaffolding"
```

### Task 4: Add runtime config, result recording, and profile collection helpers

**Files:**
- Create: `smoke/runtime/__init__.py`
- Create: `smoke/runtime/run_ort_cpu.py`
- Create: `smoke/runtime/run_ort_qnn.py`
- Create: `smoke/runtime/collect_profile.py`
- Test: `smoke/tests/test_runtime_config.py`
- Test: `smoke/tests/test_collect_profile.py`

- [ ] **Step 1: Write the failing runtime-helper tests**

```python
# smoke/tests/test_runtime_config.py
from pathlib import Path

from smoke.runtime.run_ort_qnn import OrtQnnRunSpec, qnn_provider_options


def test_qnn_provider_options_enable_context_and_disable_cpu_fallback():
    spec = OrtQnnRunSpec(
        model_path=Path("smoke/artifacts/qnn/toy/toy_ctx.onnx"),
        backend_path=Path("/opt/qairt/lib/x86_64-linux-clang/libQnnHtp.so"),
        profiling_path=Path("smoke/artifacts/profiles/toy_profile.csv"),
        disable_cpu_fallback=True,
    )

    options = qnn_provider_options(spec)

    assert options["backend_path"].endswith("libQnnHtp.so")
    assert options["ep.context_enable"] == "1"
    assert options["session.disable_cpu_ep_fallback"] == "1"
```

```python
# smoke/tests/test_collect_profile.py
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
python3 -m pytest smoke/tests/test_runtime_config.py smoke/tests/test_collect_profile.py -q
```

Expected: FAIL with `ModuleNotFoundError` for `smoke.runtime.run_ort_qnn` and `smoke.runtime.collect_profile`.

- [ ] **Step 3: Write the minimal implementation**

```python
# smoke/runtime/__init__.py
__all__ = ["run_ort_cpu", "run_ort_qnn", "collect_profile"]
```

```python
# smoke/runtime/run_ort_qnn.py
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OrtQnnRunSpec:
    model_path: Path
    backend_path: Path
    profiling_path: Path
    disable_cpu_fallback: bool


def qnn_provider_options(spec: OrtQnnRunSpec) -> dict[str, str]:
    return {
        "backend_path": str(spec.backend_path),
        "profiling_level": "detailed",
        "profiling_file_path": str(spec.profiling_path),
        "ep.context_enable": "1",
        "ep.context_embed_mode": "0",
        "session.disable_cpu_ep_fallback": "1" if spec.disable_cpu_fallback else "0",
    }
```

```python
# smoke/runtime/run_ort_cpu.py
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OrtCpuRunSpec:
    model_path: Path
```

```python
# smoke/runtime/collect_profile.py
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
python3 -m pytest smoke/tests/test_runtime_config.py smoke/tests/test_collect_profile.py -q
```

Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add smoke/runtime/__init__.py smoke/runtime/run_ort_cpu.py smoke/runtime/run_ort_qnn.py smoke/runtime/collect_profile.py smoke/tests/test_runtime_config.py smoke/tests/test_collect_profile.py
git commit -m "test: add smoke runtime result scaffolding"
```

### Task 5: Add smoke-matrix template and preflight SDK checklist

**Files:**
- Create: `smoke/docs/smoke-matrix.md`
- Modify: `smoke/README.md`

- [ ] **Step 1: Write the failing documentation expectation as file-content checks**

```python
# append to smoke/tests/test_collect_profile.py
from pathlib import Path


def test_smoke_matrix_template_contains_all_validation_columns():
    text = Path("smoke/docs/smoke-matrix.md").read_text()

    assert "| Graph | CPU ORT | QNN Compile | ORT QNN Session | ORT QNN Execute | Profile | Failing Stage |" in text
    assert "SDK version" in text
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
python3 -m pytest smoke/tests/test_collect_profile.py -q
```

Expected: FAIL because `smoke/docs/smoke-matrix.md` does not exist yet.

- [ ] **Step 3: Write the minimal implementation**

```markdown
<!-- smoke/docs/smoke-matrix.md -->
# Smoke Test Matrix

## Environment

- Host:
- SDK version:
- ORT build:
- Backend path:

## Results

| Graph | CPU ORT | QNN Compile | ORT QNN Session | ORT QNN Execute | Profile | Failing Stage |
| --- | --- | --- | --- | --- | --- | --- |
| toy | PENDING | PENDING | PENDING | PENDING | PENDING | pending |
| mini_block | PENDING | PENDING | PENDING | PENDING | PENDING | pending |
```

```markdown
<!-- append to smoke/README.md -->

## Preflight Checklist

- Qualcomm `QNN/QAIRT SDK` is installed or download access is confirmed
- `QPM` login / entitlement / license activation completed if required
- Backend library path is known
- `ONNX Runtime` with `QNN EP` is available on the host
- `smoke/docs/smoke-matrix.md` is updated after every run
```

- [ ] **Step 4: Run the test to verify it passes**

Run:

```bash
python3 -m pytest smoke/tests/test_collect_profile.py -q
```

Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add smoke/docs/smoke-matrix.md smoke/README.md smoke/tests/test_collect_profile.py
git commit -m "docs: add smoke matrix and sdk preflight checklist"
```

### Task 6: Add CLI entry points that wire the smoke-test phases together

**Files:**
- Modify: `smoke/export/export_toy_onnx.py`
- Modify: `smoke/export/export_mini_block_onnx.py`
- Modify: `smoke/compile/compile_qnn.py`
- Modify: `smoke/runtime/run_ort_cpu.py`
- Modify: `smoke/runtime/run_ort_qnn.py`

- [ ] **Step 1: Write the failing smoke CLI expectations**

```python
# append to smoke/tests/test_compile_qnn.py
from smoke.compile.compile_qnn import parse_compile_args


def test_parse_compile_args_reads_graph_name_and_output_dir():
    args = parse_compile_args([
        "--graph-name", "toy",
        "--onnx-model", "smoke/artifacts/onnx/toy.onnx",
        "--sdk-root", "/opt/qairt",
        "--output-dir", "smoke/artifacts/qnn/toy",
    ])

    assert args.graph_name == "toy"
    assert args.output_dir.endswith("smoke/artifacts/qnn/toy")
```

```python
# append to smoke/tests/test_runtime_config.py
from smoke.runtime.run_ort_cpu import parse_cpu_args


def test_parse_cpu_args_reads_model_path():
    args = parse_cpu_args(["--model", "smoke/artifacts/onnx/toy.onnx"])

    assert args.model == "smoke/artifacts/onnx/toy.onnx"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
python3 -m pytest smoke/tests/test_compile_qnn.py smoke/tests/test_runtime_config.py -q
```

Expected: FAIL because the parsing helpers do not exist yet.

- [ ] **Step 3: Write the minimal implementation**

```python
# append to smoke/compile/compile_qnn.py
import argparse


def parse_compile_args(argv: list[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-name", required=True)
    parser.add_argument("--onnx-model", required=True)
    parser.add_argument("--sdk-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--profiling-level", default="detailed")
    return parser.parse_args(argv)
```

```python
# append to smoke/runtime/run_ort_cpu.py
import argparse


def parse_cpu_args(argv: list[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    return parser.parse_args(argv)
```

```python
# append to smoke/runtime/run_ort_qnn.py
import argparse


def parse_qnn_args(argv: list[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--profile", required=True)
    return parser.parse_args(argv)
```

```python
# append to smoke/export/export_toy_onnx.py
import argparse


def parse_toy_export_args(argv: list[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", required=True)
    return parser.parse_args(argv)
```

```python
# append to smoke/export/export_mini_block_onnx.py
import argparse


def parse_mini_export_args(argv: list[str]):
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", required=True)
    return parser.parse_args(argv)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
python3 -m pytest smoke/tests/test_compile_qnn.py smoke/tests/test_runtime_config.py -q
```

Expected: all listed tests pass.

- [ ] **Step 5: Commit**

```bash
git add smoke/export/export_toy_onnx.py smoke/export/export_mini_block_onnx.py smoke/compile/compile_qnn.py smoke/runtime/run_ort_cpu.py smoke/runtime/run_ort_qnn.py smoke/tests/test_compile_qnn.py smoke/tests/test_runtime_config.py
git commit -m "feat: add smoke cli parsing helpers"
```

### Task 7: Run the smoke-test unit suite and document the first manual run order

**Files:**
- Modify: `smoke/README.md`

- [ ] **Step 1: Add the manual run order to the README**

```markdown
<!-- append to smoke/README.md -->

## First Manual Run Order

1. Export `toy.onnx`
2. Run CPU `ORT` on `toy.onnx`
3. Compile `toy.onnx` with `QNN`
4. Run `ORT + QNN EP` on the compiled `toy` artifact
5. Repeat steps 1-4 for `mini_block.onnx`
6. Record all outcomes in `smoke/docs/smoke-matrix.md`
```

- [ ] **Step 2: Run the full smoke-test unit suite**

Run:

```bash
python3 -m pytest smoke/tests -q
```

Expected: all smoke-test unit tests pass.

- [ ] **Step 3: Commit**

```bash
git add smoke/README.md
git commit -m "docs: add smoke test run order"
```

## Self-Review

- Spec coverage checked: the plan covers preflight SDK risk, toy and mini-block graph definitions, ONNX export naming, QNN compile command scaffolding, ORT/QNN runtime configuration, result recording, and the smoke matrix.
- Placeholder scan checked: no placeholder markers remain.
- Type consistency checked: `ToyGraphSpec`, `MiniBlockSpec`, `ExportArtifactSpec`, `CompileSpec`, `OrtQnnRunSpec`, and `RunResult` are used consistently across tasks.
