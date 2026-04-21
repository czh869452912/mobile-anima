# Android ORT Java + QNN Provider Gap Closure Design

**Date:** 2026-04-21

## Goal

Close the highest-value Android runtime gaps between the current denoiser-first app shell and a real `ORT Java API + QNNExecutionProvider` integration path.

This stage does not attempt to finish full prompt-to-image generation. It focuses on replacing the current fake Android ORT bridge with a real session-creation layer, explicit provider configuration, and actionable runtime error reporting.

## Why This Stage Exists

The repository already proves that:

- host-side `QNN` compile and runtime plumbing works for cheap smoke graphs
- desktop `ONNX Runtime + QNNExecutionProvider` works on Linux
- the Android app has a denoiser-first shell, artifact paths, and UI wiring

However, the Android runtime seam is still the weakest link.

The current `ReflectionOrtSessionFactory` can only probe whether `QNNExecutionProvider` is visible. It does not build a real `InferenceSession`, does not pass provider options into Android ORT, and does not provide enough error detail to distinguish missing AARs from session-creation or execute failures.

The cheapest next move is therefore not “finish all Android inference.” The cheapest next move is to harden the Android ORT/QNN adapter boundary so later tensor I/O work sits on a trustworthy session layer.

## Scope

This stage covers:

- replacing the fake `OrtSessionFactory` behavior with a real Android ORT Java session-creation adapter
- pushing `OrtQnnConfig.providerOptions()` into Android `SessionOptions`
- making provider/session/execute failures distinguishable in runtime results
- keeping the runtime layer JVM-testable without requiring the real Android ORT AAR during unit tests
- preserving the existing denoiser-first orchestration boundary

This stage does not cover:

- full prompt-to-image generation
- text encoder or VAE execution on Android
- final real tensor marshaling for all denoiser inputs
- Android instrumentation tests on physical devices
- model export, quantization, or QNN context generation
- performance tuning

## User Decisions Captured

- Continue on the Android main line instead of expanding desktop validation first
- Prioritize the `ORT Java API + QNN` adapter gap before broader denoiser execution work
- Use a staged path:
  - first make Android session creation and provider configuration real
  - then add explicit runtime diagnostics
  - then layer full tensor execution on top of that stable seam

## Approaches Considered

### Option 1: Keep reflection only and patch the existing class inline

**Recommendation:** No

**Pros**

- smallest diff count
- fast to start coding

**Cons**

- reflection details leak into business logic
- hard to test failure modes cleanly
- session lifecycle and error mapping stay muddy

### Option 2: Add a thin ORT bridge under `OrtSessionFactory`

**Recommendation:** Yes

**Pros**

- preserves current architecture
- isolates `ai.onnxruntime.*` interaction in one place
- makes provider/session/execute behavior testable via fakes
- reduces risk before real tensor I/O work

**Cons**

- introduces a small extra abstraction layer
- still requires careful mapping from bridge failures to app-visible results

### Option 3: Skip Java API cleanup and jump directly to full tensor execution

**Recommendation:** Not yet

**Pros**

- aims closer to end-state behavior

**Cons**

- combines too many unknowns at once
- makes debugging session problems much harder
- raises the chance of masking provider misconfiguration behind tensor bugs

## Recommended Strategy

Use a thin Android ORT bridge beneath `OrtSessionFactory`.

The runtime layer should keep `OrtQnnDenoiserEngine` as the orchestration entrypoint, but move all ORT Java details into a focused adapter responsible for:

- discovering ORT classes
- creating the environment
- creating session options
- attaching `QNNExecutionProvider` with explicit provider options
- constructing the denoiser session
- reporting failures at the correct boundary

This design keeps the business layer simple while making the ORT boundary explicit and testable.

## System Architecture

This stage uses three bounded Android runtime layers.

### 1. Engine Layer

Primary file:

- `android/app/src/main/java/com/example/animanpu/runtime/OrtQnnDenoiserEngine.kt`

Responsibilities:

- validate required runtime artifacts exist before session creation
- assemble `OrtQnnConfig`
- request a denoiser session from the factory
- convert session/execute outcomes into `GenerationResult`

The engine must not know how ORT Java reflection works.

### 2. Session Factory Layer

Primary file:

- `android/app/src/main/java/com/example/animanpu/runtime/OrtSessionFactory.kt`

Responsibilities:

- expose `providerVisible()`
- create a real denoiser session
- translate low-level ORT bridge failures into stable runtime error categories

This layer is the contract between the engine and ORT Java.

### 3. ORT Bridge Layer

New Android runtime file(s):

- a small bridge or adapter under `android/app/src/main/java/com/example/animanpu/runtime/`

Responsibilities:

- isolate `ai.onnxruntime.*` reflection and invocation
- create `OrtEnvironment`
- build `SessionOptions`
- add the QNN provider with provider options
- create `InferenceSession`

This layer should be thin and mechanical, not business-aware.

## Data and Control Flow

1. `MainActivity` constructs `OrtQnnDenoiserEngine` with an `OrtSessionFactory`.
2. `OrtQnnDenoiserEngine.generate()` checks for required runtime files.
3. The engine builds `OrtQnnConfig` with:
   - `backend_path`
   - profiling options
   - `session.disable_cpu_ep_fallback=1`
4. `OrtSessionFactory.createDenoiserSession()` asks the ORT bridge to:
   - resolve ORT classes
   - resolve available providers
   - create `SessionOptions`
   - add the `QNNExecutionProvider`
   - create the denoiser session
5. The created handle reports either:
   - session ready
   - session create failure
   - execute failure

The key rule is that provider configuration and session creation happen before any future tensor marshaling logic is layered in.

## Error Model

The Android runtime must stop returning only a vague boolean meaning.

This stage standardizes these failure classes:

- `ort_api_unavailable`
  - ORT classes or required methods are missing
  - likely AAR mismatch, missing AAR, or incompatible Java API surface
- `qnn_provider_unavailable`
  - ORT loads, but `QNNExecutionProvider` is not visible
- `session_create_failed`
  - provider exists, but denoiser session construction fails
- `execute_not_implemented`
  - temporary stage-only placeholder for a created session whose real tensor execution path is not wired yet
- `execute_failed`
  - future real execution path throws or reports failure

The current stage may still use placeholder execution beneath a created session, but it must identify that state explicitly instead of pretending success or returning a silent `false`.

## API and Model Changes

### `GenerationResult`

Add a lightweight failure reason field so the UI and orchestrator can distinguish categories without parsing exception text.

### `OrtSessionHandle`

Keep the interface small, but evolve it so it can express more than a bare success boolean. The handle must be able to preserve session lifecycle state and return an explicit execution outcome.

### `OrtSessionFactory`

Keep the public contract minimal and stable. It should remain the only runtime entrypoint that knows how to create the denoiser session.

## Testing Strategy

Use JVM unit tests first.

### Required test coverage

- provider visible when the bridge reports `QNNExecutionProvider`
- `ort_api_unavailable` when ORT classes or methods are absent
- `qnn_provider_unavailable` when ORT loads but QNN is not present
- `session_create_failed` when session creation throws
- engine maps missing artifacts before any session work
- engine preserves profiling and output paths in failure results
- engine distinguishes session creation success from execute-not-implemented behavior

### Explicit non-goal for this stage

- do not require physical device or instrumentation tests to prove the adapter API shape

Those belong to the next stage after the adapter seam is real.

## Acceptance Criteria

This stage is complete when:

- the fake `createDenoiserSession()` placeholder is gone
- Android runtime code contains a real ORT Java session-construction path
- provider options are passed into Android ORT session setup through one defined adapter seam
- runtime results clearly distinguish ORT API absence, QNN provider absence, session-create failure, and execute-stage failure
- JVM tests cover the adapter and engine error mapping without requiring a real Android ORT AAR

This stage is not complete merely because the code compiles.

## Stop-Loss Rules

### Stop-Loss 1: ORT API surface mismatch

If the Android ORT Java API does not expose the expected provider-configuration surface, stop and classify the issue as ORT Android API mismatch rather than hiding it behind generic runtime failure.

### Stop-Loss 2: Provider visible but provider options cannot be attached

Do not continue toward full tensor work if `QNNExecutionProvider` is visible but `backend_path` and no-fallback options cannot be wired in reliably.

### Stop-Loss 3: Session creation remains ambiguous

If failures still require reading stack traces to know whether ORT was missing or QNN was missing, the adapter boundary is not complete yet.

## Next Stage After This One

Once the Android ORT/QNN adapter seam is real and testable, the next lowest-cost stage is to wire actual denoiser tensor inputs and outputs through that seam and then validate on device with profiling evidence.
