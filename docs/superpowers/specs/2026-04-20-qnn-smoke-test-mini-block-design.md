# QNN Smoke Test + Mini Denoiser Block Validation Design

**Status:** Approved for planning

**Owner:** Codex + user

**Date:** 2026-04-20

## Goal

Build the lowest-cost validation project that proves the Qualcomm `QNN/QAIRT` toolchain and `ONNX Runtime + QNN EP` runtime path are working before attempting the real `Anima` denoiser export.

This stage is intentionally separate from the Android image-generation app. The purpose is to isolate toolchain and runtime feasibility, not to extend end-user functionality.

## Why This Stage Exists

The main project risk is not the Android UI. The main risk is whether an `Anima`-like denoiser graph can be exported, compiled, loaded, and executed through Qualcomm's NPU path.

This stage reduces risk in the cheapest order:

1. Eliminate environment and SDK issues
2. Eliminate `QNN` compile-path issues
3. Eliminate `ORT + QNN EP` runtime issues
4. Only then attempt the real `Anima` denoiser

## Scope

This stage covers:

- Qualcomm SDK access and installation as an explicit preflight dependency
- A `toy` graph smoke test
- A `mini transformer-like block` smoke test
- Validation at both layers:
  - `ONNX -> QNN compile`
  - `ORT + QNN EP` runtime load and execute
- Desktop/Linux-first execution
- Android extension only after desktop validation passes

This stage does not yet cover:

- Real `Anima` denoiser execution
- Real image generation
- Text encoder integration
- VAE integration
- End-user Android inference flow

## User Decisions Captured

- Validate `denoiser -> ONNX -> QNN` before deeper Android product work
- Use a two-stage validation path:
  - first a minimal smoke test
  - then a more `denoiser-like` block
- Validate both compiler and runtime layers, not just one
- Run desktop/Linux validation first, then extend to Android if desktop passes
- Treat Qualcomm SDK acquisition as likely available, but not guaranteed to be already configured

## Recommended Strategy

Use a three-milestone validation ladder.

### Milestone 1: Toy Graph

Use a minimal graph such as:

- `MatMul + Add + Relu`

Purpose:

- prove ONNX export is correct
- prove the `QNN` compile path works
- prove `ORT + QNN EP` can build a session and execute
- prove profiling/logging paths are wired correctly

### Milestone 2: Mini Transformer-Like Block

Use a block that is more representative of denoiser compute. Preferred first option:

- `LayerNorm + MatMul + residual`

Fallback if `LayerNorm` introduces avoidable early complexity:

- `MatMul + Add + residual + activation`

Purpose:

- approximate the shape of a denoiser subgraph without the full `Anima` export burden
- identify operator support gaps earlier
- separate `toolchain is broken` from `real model graph is too complex`

### Milestone 3: Real Anima Denoiser

Only begin this milestone when Milestones 1 and 2 are complete and documented.

Purpose:

- attempt real `Anima` denoiser export and compilation with prior confidence that the surrounding toolchain is not the primary unknown

## Architecture

The stage should be implemented as an independent smoke-test subproject.

### Directory Structure

- `smoke/`
  - `models/`
    - `toy_graph.py`
    - `mini_block.py`
  - `export/`
    - `export_toy_onnx.py`
    - `export_mini_block_onnx.py`
  - `compile/`
    - `compile_qnn.py`
    - `inspect_compile_result.py`
  - `runtime/`
    - `run_ort_cpu.py`
    - `run_ort_qnn.py`
    - `collect_profile.py`
  - `artifacts/`
    - `onnx/`
    - `qnn/`
    - `profiles/`
    - `logs/`
  - `docs/`
    - `smoke-matrix.md`

### Component Responsibilities

- `models/` defines graph structure only
- `export/` converts model definitions into ONNX artifacts
- `compile/` performs Qualcomm compile and output inspection only
- `runtime/` executes ONNX first on CPU, then through `QNN EP`
- `artifacts/` contains generated files and logs
- `docs/` records results and failures with enough detail to guide the next stage

## Execution Flow

### Phase 1: Toy Graph

1. Generate `toy.onnx`
2. Run with CPU `ORT` to prove the ONNX artifact is valid
3. Compile with Qualcomm `QNN`
4. Run with `ORT + QNN EP`
5. Save logs and profiling outputs
6. Record result in `smoke-matrix.md`

### Phase 2: Mini Block

1. Generate `mini_block.onnx`
2. Run with CPU `ORT`
3. Compile with Qualcomm `QNN`
4. Run with `ORT + QNN EP`
5. Save logs and profiling outputs
6. Record result in `smoke-matrix.md`

### Phase 3: Decision Gate

Interpret outcomes as follows:

- `toy` passes and `mini_block` passes → proceed to real `Anima` denoiser
- `toy` passes and `mini_block` fails → analyze operator/support gap before touching `Anima`
- `toy` fails → do not start `Anima`; toolchain/runtime path is still unproven

## Validation Layers

Each graph must be validated at two separate layers.

### Layer 1: Compile Validation

Success means:

- ONNX artifact exists
- `QNN` compile command completes successfully
- expected compile outputs are produced

Failure categories to record:

- export failure
- compile command failure
- missing backend/runtime dependency
- unsupported operator during compilation

### Layer 2: Runtime Validation

Success means:

- `ORT + QNN EP` can construct a session
- the graph executes once successfully
- profiling or equivalent execution evidence is written

Failure categories to record:

- `ORT` session creation failure
- provider option mismatch
- runtime library lookup failure
- execution failure after session creation
- profiling output missing

## Preflight Dependencies

This stage must explicitly validate the following before smoke tests begin:

- Qualcomm `QNN/QAIRT SDK` can be obtained
- any required `QPM` login / entitlement / license activation is completed
- host runtime libraries and backend libraries are locatable
- `ONNX Runtime` with `QNN EP` is available or buildable on the test machine

The design should not assume the SDK is already installed, only that it is likely obtainable.

## Error Handling and Evidence Policy

Every failed run must produce a concrete classification and artifact.

For each smoke-test attempt, record:

- graph name
- host environment summary
- SDK version
- compile command used
- runtime command used
- success/failure state for each validation layer
- exact failing stage
- error message or log path
- whether profiling was generated

This prevents the common failure mode of ending with only a vague statement such as "QNN failed".

## Acceptance Criteria

### Milestone 1 Acceptance

The stage passes Milestone 1 only if:

- `toy.onnx` is generated
- `toy.onnx` executes with CPU `ORT`
- `toy.onnx` compiles through `QNN`
- `toy.onnx` executes with `ORT + QNN EP`
- some profiling or equivalent Qualcomm execution evidence is produced

### Milestone 2 Acceptance

The stage passes Milestone 2 only if:

- `mini_block.onnx` is generated
- `mini_block.onnx` executes with CPU `ORT`
- the `QNN` compile result is clearly classified as success or failure
- the `ORT + QNN EP` run result is clearly classified as success or failure
- any failure is tied to a concrete stage and artifact

### Stage Success Definition

This entire stage is successful when:

- the toolchain is proven on `toy`
- the mini block has a definitive result at compile and runtime layers
- the team knows whether the next step is:
  - proceed to `Anima`
  - perform operator-gap analysis
  - repair environment/toolchain first

## Testing Strategy

Testing should be layered in increasing cost order.

1. Unit tests for command builders and artifact-path logic
2. CPU `ORT` execution checks for exported ONNX validity
3. `QNN` compile execution checks
4. `ORT + QNN EP` runtime checks
5. Android extension only after desktop results are stable

## Risks

### Risk 1: SDK access is slower or more restricted than expected

Mitigation:

- treat SDK availability as explicit preflight work
- do not start real-model work until SDK acquisition is confirmed

### Risk 2: Toy graph passes but mini block fails on unsupported operators

Mitigation:

- record exact failing operator or stage
- use that information to decide whether `Anima` is still worth attempting

### Risk 3: ORT/QNN runtime fails for reasons unrelated to graph content

Mitigation:

- validate compile layer and runtime layer separately
- keep a CPU `ORT` path for baseline execution proof

### Risk 4: Desktop passes but Android later fails

Mitigation:

- explicitly define Android as a follow-on extension, not part of desktop smoke acceptance
- isolate Android runtime differences to the next design stage

## Out of Scope

- Full Android integration for this smoke-test stage
- Real-image generation
- Real text encoding
- Real VAE decode
- Prompt handling
- Scheduler behavior
- Real `Anima` denoiser export until milestones 1 and 2 are complete

## Decision Summary

- Use a dedicated smoke-test subproject
- Validate compile and runtime layers separately
- Use `toy` first, then `mini transformer-like block`
- Run desktop first, extend to Android later
- Treat Qualcomm SDK acquisition as a first-class dependency and risk
