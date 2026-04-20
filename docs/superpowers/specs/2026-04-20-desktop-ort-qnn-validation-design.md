# Desktop ORT + QNN EP Validation Design

**Status:** Approved for planning

**Owner:** Codex + user

**Date:** 2026-04-20

## Goal

Validate the final missing layer in the current smoke-test ladder by building a Linux desktop `ONNX Runtime` with `QNN Execution Provider` and using it to create sessions and execute the already-proven `toy` and `mini_block` models.

This stage exists to prove `ORT + QNN EP` integration independently before attempting the real `Anima` denoiser.

## Why This Stage Exists

The project has already proven the following on Linux desktop:

- `ONNX` generation for `toy` and `mini_block`
- `qnn-onnx-converter`
- `qnn-model-lib-generator`
- `qnn-context-binary-generator`
- `qnn-net-run`
- CPU `ORT` baseline execution for both graphs

The only missing validation layer in the smoke matrix is:

- `ORT QNN Session`
- `ORT QNN Execute`

Without this stage, the team still does not know whether `ONNX Runtime` itself can integrate with the Qualcomm path on the desktop host.

## Scope

This stage covers:

- Linux desktop `ONNX Runtime` acquisition strategy
- Prefer prebuilt search first, then source build fallback
- Building a Linux `ORT` with `QNN EP` if no usable prebuilt exists
- Running `toy.onnx` with `ORT + QNN EP`
- Running `mini_block.onnx` with `ORT + QNN EP`
- Recording session-creation and execution results in a dedicated matrix

This stage does not cover:

- Android `ORT + QNN EP`
- Real `Anima` denoiser export
- Performance optimization
- Quantization tuning
- Mobile packaging

## User Decisions Captured

- Desktop validation comes before Android expansion
- If a prebuilt `ORT + QNN EP` exists, prefer it
- If not, build from source
- Validate `toy` first, then `mini_block`
- Do not start the real `Anima` denoiser until this stage is complete or clearly blocked

## Recommended Strategy

Use a two-path acquisition strategy.

### Option 1: Prebuilt ORT + QNN EP (fast path)

Search for a usable Linux desktop `ORT` build with `QNN EP` support.

**Pros**

- Fastest path to proof
- Avoids a potentially long ORT source build

**Cons**

- Linux desktop prebuilt support may not exist or may not match the required environment
- Harder to control build flags and SDK compatibility

### Option 2: Build ORT from Source (fallback and likely default)

Fetch ORT source and build a Linux desktop package with `QNN EP` enabled against the installed `QAIRT/QNN` SDK.

**Pros**

- Full control of build flags and SDK paths
- More reliable if prebuilt artifacts are not available

**Cons**

- Slower
- Requires more host dependencies

### Recommendation

Attempt Option 1 briefly, but assume Option 2 is the likely path. The design should optimize for a clean source build fallback rather than betting on a prebuilt artifact.

## Architecture

Implement this stage as a separate desktop-validation subproject.

### Directory Structure

- `ort_qnn/`
  - `build/`
    - `fetch_onnxruntime.sh`
    - `build_ort_qnn_linux.sh`
  - `runtime/`
    - `run_toy_ort_qnn.py`
    - `run_mini_block_ort_qnn.py`
    - `provider_options.py`
  - `artifacts/`
    - `ort/`
    - `logs/`
    - `profiles/`
  - `docs/`
    - `ort-qnn-matrix.md`

### Component Responsibilities

- `build/` fetches or builds Linux desktop ORT with `QNN EP`
- `runtime/` creates sessions and runs inference for `toy` and `mini_block`
- `artifacts/` stores build logs, runtime logs, and profiling outputs
- `docs/` records results for session creation and execution in a compact matrix

## Execution Order

### Step 1: Acquire ORT

- Check for a usable prebuilt Linux `ORT + QNN EP`
- If not available, fetch ORT source and build it

### Step 2: Validate Build Artifact

- Confirm the resulting runtime exposes `QNNExecutionProvider` or equivalent QNN provider binding
- Confirm the runtime can locate required Qualcomm backend libraries

### Step 3: Validate `toy.onnx`

- Create `ORT + QNN EP` session for `toy.onnx`
- If session creation succeeds, run one inference
- Save runtime logs and any profiling outputs

### Step 4: Validate `mini_block.onnx`

- Repeat session creation and execution checks for `mini_block.onnx`
- Record any failure at a concrete stage

### Step 5: Update Matrix and Decide Next Move

Interpret outcomes as follows:

- `toy` and `mini_block` both pass → proceed to real `Anima` denoiser planning
- `toy` passes and `mini_block` fails → analyze model/operator gap before touching `Anima`
- `toy` fails → focus on ORT/QNN integration before any model work

## Validation Layers

Each graph must be validated at two `ORT`-specific layers.

### Layer 1: Session Validation

Success means:

- the model loads into `ORT`
- `QNN EP` is requested successfully
- the session is created without falling back silently to CPU

Failure categories to record:

- provider unavailable
- runtime library not found
- provider option mismatch
- model rejected during session construction

### Layer 2: Execution Validation

Success means:

- one inference run completes successfully
- output file or output tensor is produced
- profiling or provider evidence is generated if configured

Failure categories to record:

- session exists but execute fails
- input formatting mismatch
- provider runtime failure during execute
- profiling output missing

## Evidence Policy

For each graph, record:

- ORT version or build identity
- acquisition path (`prebuilt` or `source build`)
- QNN SDK version in use
- provider options used
- session result
- execute result
- log path
- profile path if generated

## Acceptance Criteria

### Toy Acceptance

The `toy` validation passes if:

- desktop `ORT + QNN EP` artifact is available
- `toy.onnx` session creation succeeds with `QNN EP`
- `toy.onnx` executes successfully

### Mini Block Acceptance

The `mini_block` validation passes if:

- `mini_block.onnx` session creation succeeds with `QNN EP`
- `mini_block.onnx` executes successfully

If it fails, the stage still produces useful output only if the failure is classified precisely enough to guide the next step.

### Stage Success Definition

This stage is successful when:

- the team knows whether desktop `ORT + QNN EP` works for `toy`
- the team knows whether desktop `ORT + QNN EP` works for `mini_block`
- the result matrix makes the next step obvious:
  - proceed to `Anima`
  - analyze operator gap
  - repair ORT/QNN integration first

## Risks

### Risk 1: No usable Linux prebuilt exists

Mitigation:

- treat source build as the default fallback path

### Risk 2: Source build succeeds but provider is not actually registered

Mitigation:

- explicitly validate provider availability before running any model

### Risk 3: ORT/QNN fails on desktop even though raw QAIRT tools already work

Mitigation:

- keep this stage isolated so failures are attributed to ORT/QNN integration, not to model conversion

### Risk 4: `toy` passes but `mini_block` fails at session or execute time

Mitigation:

- record exact failure phase and preserve logs for operator-gap analysis

## Out of Scope

- Android ORT integration
- Real `Anima` denoiser export
- Mobile UI changes
- Production packaging
- Benchmark optimization

## Decision Summary

- Validate only desktop `ORT + QNN EP`
- Prefer prebuilt search first, fallback to source build
- Run `toy` before `mini_block`
- Use the resulting matrix to decide whether the project is ready for the real `Anima` denoiser
