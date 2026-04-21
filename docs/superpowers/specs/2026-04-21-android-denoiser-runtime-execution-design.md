# Android Denoiser Runtime Execution Design

**Date:** 2026-04-21

## Goal

Implement the next lowest-cost Android runtime milestone after session creation: execute one real local denoiser inference by dynamically reading input metadata from the Android `ORT` session, binding the four existing runtime `raw` inputs, and writing the denoiser output tensor back to app storage.

This stage still does not attempt full prompt-to-image generation. It focuses on turning the existing Android `ORT + QNN` session seam into a real one-shot denoiser execution path.

## Why This Stage Exists

The repository now already has:

- an Android `ORT + QNN` session-construction seam
- explicit runtime failure categories for ORT/QNN provider and session creation failures
- JVM tests proving the Android runtime layer can distinguish session-setup failures before touching device execution

However, the current runtime still stops at `EXECUTE_NOT_IMPLEMENTED`.

That means the Android app can prove that it can discover ORT, discover QNN, and create a session boundary, but it still cannot:

- inspect the real session inputs
- validate the runtime `raw` payloads against the model contract
- bind tensors into the session
- run a denoiser pass
- persist the output tensor from real inference

The next cheapest, most valuable move is to add a real one-shot denoiser execute path without prematurely expanding into text encoder, VAE, scheduler, or full prompt-to-image orchestration.

## Scope

This stage covers:

- dynamically reading denoiser input metadata from the Android ORT session
- mapping session inputs onto the existing four runtime files:
  - `latent.raw`
  - `timestep.raw`
  - `cond.raw`
  - `uncond.raw`
- validating input tensor shapes, element types, and file sizes before execution
- creating ORT tensors from the runtime `raw` files
- executing the denoiser once through the already-created Android session
- writing the primary output tensor to `denoiser_output.raw`
- surfacing precise execute-stage failure categories in `GenerationResult`

This stage does not cover:

- full prompt-to-image generation
- text encoder execution on Android
- VAE decode on Android
- ONNX graph parsing outside the ORT session
- sidecar metadata files such as `inputs.json`
- dynamic runtime support for arbitrary numbers of denoiser inputs
- Android instrumentation or device automation in this stage

## User Decisions Captured

- Continue from the newly-added Android `ORT + QNN` session seam instead of switching focus back to desktop work
- Use dynamic input inspection from the live ORT session rather than hardcoding shapes or introducing a sidecar metadata file
- Keep the four existing runtime `raw` filenames as the input contract for this stage
- Fail explicitly when metadata is ambiguous rather than guessing and silently continuing

## Approaches Considered

### Option 1: Hardcode the current v1 tensor contract in the app

**Recommendation:** No

**Pros**

- lowest immediate implementation cost
- no reflection or metadata adaptation work beyond current session seam

**Cons**

- fragile if the exported denoiser contract changes
- duplicates model contract assumptions in Android code
- makes debugging shape drift harder

### Option 2: Read a sidecar metadata file next to the raw inputs

**Recommendation:** No

**Pros**

- more flexible than hardcoding
- easier to reason about than full dynamic inspection

**Cons**

- introduces one more artifact that must stay in sync with the model
- duplicates information already available from the ORT session
- adds packaging complexity without proving more runtime value

### Option 3: Read input metadata directly from the live ORT session

**Recommendation:** Yes

**Pros**

- uses the runtime as the source of truth
- avoids an extra parser or metadata artifact
- best balance between correctness and implementation cost
- keeps Android execution aligned with the actual model contract

**Cons**

- requires a thin metadata inspection layer in the Android ORT bridge
- needs careful ambiguous-input failure handling

## Recommended Strategy

Use the existing Android ORT bridge as the source of truth for input metadata.

The runtime should keep the current architecture:

- `OrtQnnDenoiserEngine` remains the top-level Android runtime entrypoint
- `OrtSessionFactory` still owns the session boundary
- the ORT bridge remains the only place that knows how to speak `ai.onnxruntime.*`

What changes in this stage is the session handle.

Instead of returning `EXECUTE_NOT_IMPLEMENTED`, the handle should:

1. inspect the session input metadata
2. map those inputs to the four known runtime files
3. validate shapes, types, and file sizes
4. build the input tensor feed
5. execute the session once
6. extract the primary output tensor
7. write it to `denoiser_output.raw`

## System Architecture

This stage extends the existing Android runtime with three focused responsibilities.

### 1. Session Metadata Layer

Primary responsibility:

- expose input names, element types, and shapes from the live ORT session

This should live under the Android ORT bridge boundary so the rest of the runtime does not need to understand ONNX Runtime Java APIs directly.

### 2. Input Binding Layer

Primary responsibility:

- map discovered session inputs onto the four runtime raw files
- validate element count and byte size
- reject ambiguous or unsupported session contracts

This layer should be deterministic and explicit. If the runtime cannot confidently map the inputs, it must fail instead of guessing.

### 3. Execute-and-Persist Layer

Primary responsibility:

- create ORT tensors from validated runtime files
- run the denoiser once
- persist the primary output tensor to `denoiser_output.raw`

This is the first stage where the Android runtime performs real denoiser execution rather than stopping at session creation.

## Data and Control Flow

1. `OrtQnnDenoiserEngine.generate()` validates the required runtime files still exist.
2. `OrtSessionFactory.createDenoiserSession(...)` returns a session handle backed by a real ORT session.
3. `OrtSessionHandle.run(outputPath)` asks the ORT bridge for session input metadata.
4. The runtime maps the discovered inputs to:
   - `latent.raw`
   - `timestep.raw`
   - `cond.raw`
   - `uncond.raw`
5. The runtime validates, for each mapped input:
   - input name is known or inferable
   - shape is fully static
   - element type is supported
   - file byte count matches expected element count × type width
6. The runtime creates ORT tensors and executes the session.
7. The runtime selects the primary output tensor.
8. The runtime writes the output bytes to `denoiser_output.raw`.
9. The engine returns a `GenerationResult` with timing, output path, and success/failure state.

## Input Mapping Rules

The mapping rules must use metadata plus lightweight heuristics, not file-order guessing.

### `timestep`

Map to the input that is scalar-like or single-element, with an integer- or float-like tensor type acceptable to the exported denoiser contract.

If there is more than one plausible timestep input, fail with `input_mapping_failed`.

### `latent`

Map to the input whose shape is most plausibly the denoiser latent tensor:

- rank is expected to be 4 for the current denoiser-first path
- shape should be static

If there is more than one plausible latent input, fail with `input_mapping_failed`.

### `cond` and `uncond`

Map to the pair of inputs that:

- share the same element type
- share the same shape
- are plausibly the conditioning embeddings

If the pair exists but the runtime cannot confidently distinguish which is `cond` vs `uncond` from names, fail with `input_mapping_failed`.

The runtime must not silently swap or guess.

## Output Selection Rules

If the session returns exactly one output tensor, treat it as the denoiser output and write it to `denoiser_output.raw`.

If the session returns multiple outputs, the runtime may select the output only if there is one clearly primary tensor by metadata or expected denoiser naming. Otherwise fail explicitly instead of writing the wrong tensor.

## Supported Tensor Types

The first version of this stage should intentionally support only a narrow set of ORT tensor element types that are realistic for the exported denoiser path.

At minimum, the runtime should be prepared to accept:

- `float32`
- `int64` if the timestep input requires it

If the session exposes unsupported element types, fail with `unsupported_tensor_type`.

## Failure Model

This stage adds execute-path failures beyond the previously added session failures.

New execute-stage failures should include:

- `INPUT_METADATA_UNAVAILABLE`
  - the runtime could not read usable input metadata from the session
- `DYNAMIC_SHAPE_UNSUPPORTED`
  - an input shape contains unknown dimensions that this stage cannot size-check reliably
- `UNSUPPORTED_TENSOR_TYPE`
  - an input or output tensor type is not supported in this stage
- `INPUT_MAPPING_FAILED`
  - the runtime cannot confidently map session inputs to `latent`, `timestep`, `cond`, and `uncond`
- `INPUT_FILE_SIZE_MISMATCH`
  - the raw file byte size does not match the expected tensor size
- `SESSION_EXECUTE_FAILED`
  - the ORT session throws or reports failure while executing
- `OUTPUT_WRITE_FAILED`
  - execution succeeded but writing the output raw file failed

These should map into the existing `GenerationResult.failureReason` surface rather than hiding behind vague exceptions.

## API Changes

### `OrtJavaSession`

The Android ORT bridge should expose enough metadata and execute functionality for the session handle to:

- read input metadata
- run inference
- read output data

This may require expanding the bridge abstractions beyond “create session” only.

### `OrtSessionHandle`

The handle should become a real execute boundary rather than a placeholder. It should be responsible for returning structured execute outcomes and preserving the execute-stage failure category.

### `OrtRuntimeFailure`

Extend the enum to cover the new execute-stage failures listed above.

## Testing Strategy

Keep this stage JVM-test-driven.

### Required tests

- metadata normalization exposes input names, shapes, and types in a stable internal form
- static-size calculation detects expected byte counts correctly
- unsupported element types are rejected
- dynamic shapes are rejected
- ambiguous `cond/uncond` mapping is rejected
- incorrect raw file size is rejected before execute
- successful fake execute writes `denoiser_output.raw`
- multi-output ambiguity fails explicitly
- engine maps execute failures into `GenerationResult.failureReason`

### Intentional test boundary

This stage should not require a physical device to prove the new runtime behavior. The bridge and session handle should be testable with fakes that simulate session metadata and output tensors.

## Acceptance Criteria

This stage is complete when:

- the Android session handle no longer returns unconditional `EXECUTE_NOT_IMPLEMENTED`
- the runtime reads input metadata from the live ORT session boundary
- the runtime validates and binds the four existing raw input files
- one real denoiser execute path exists behind `handle.run(...)`
- the primary output tensor is written to `denoiser_output.raw`
- execute-stage failures are classified explicitly
- JVM tests cover both success and the major mapping/validation failure modes

This stage is not complete merely because the session runs. It must also prove that the runtime validates the model/input contract and fails safely when the contract is ambiguous.

## Stop-Loss Rules

### Stop-Loss 1: Input contract ambiguity

If the runtime cannot confidently map the session metadata to the four known raw files, stop and fail explicitly. Do not guess.

### Stop-Loss 2: Dynamic shapes in the exported denoiser contract

If the session input contract exposes unknown dimensions that prevent reliable file-size validation, stop and classify the issue as `dynamic_shape_unsupported` rather than trying to infer sizes.

### Stop-Loss 3: Multiple plausible outputs

If the runtime cannot uniquely identify the primary denoiser output tensor, stop and classify the issue instead of writing an arbitrary output tensor.

## Next Stage After This One

Once Android can execute one real denoiser pass from session metadata and runtime raw files, the next lowest-cost stage is to tighten the manual device validation loop and then decide whether to bring prompt-preparation logic or additional submodels onto Android.
