# Android Anima Denoiser Minimal Loop Design

**Date:** 2026-04-20

## Goal

Build the smallest Android-side runtime loop that assumes the `Anima` denoiser export artifacts already exist and proves that the app can execute one real local `ORT + QNN` denoiser inference on-device, capture profiling evidence, and surface the result in the UI.

This stage does not attempt to complete the entire text-to-image pipeline. It exists to turn the current Android shell into a real Android denoiser runtime loop that stays aligned with the main project goal: **Android minimum viable local NPU deployment of the Anima model family.**

## Why This Stage Exists

The project has already proven the host-side prerequisites in increasing confidence order:

- cheap QNN smoke graphs work
- desktop ORT + QNN works
- packaging/bootstrap exists for moving that desktop path to a stronger host later

However, the Android app itself is still only a UI shell and fake runtime bridge. The current runtime layer does not yet create sessions, load artifacts, run local inference, or capture profiling.

The next lowest-cost move on the main line is not full text-to-image. The next move is a denoiser-first Android closure.

## Scope

This stage covers:

- replacing the fake Android `GenerationEngine` with a real denoiser runtime engine
- loading already-exported denoiser runtime artifacts from Android app storage or packaged assets
- creating an Android `ORT + QNN` session
- running one real denoiser inference with fixed or precomputed inputs
- writing output and profiling artifacts to app-accessible storage
- surfacing runtime status, timing, and QNN evidence in the UI

This stage does not cover:

- complete prompt-to-image generation
- live text encoder execution on Android
- live VAE decode on Android
- Android-side scheduler correctness for the full diffusion loop
- model export work
- device performance tuning

## User Decisions Captured

- Continue on the Android main line, not more desktop work
- Assume `Anima` export has already succeeded for the denoiser artifacts
- Use a staged approach:
  - first achieve a real Android denoiser loop
  - then expose it through the UI
- Treat text encoder and VAE as temporarily replaceable by precomputed or placeholder inputs

## Recommended Strategy

Use a denoiser-first Android runtime loop with precomputed inputs.

### Option 1: Full prompt-to-image immediately

**Recommendation:** No

**Pros**

- closest to the final product shape

**Cons**

- drags in too many simultaneous unknowns: text encoder, VAE, scheduler, latent/image conversion, and denoiser runtime all at once
- much larger debugging surface

### Option 2: Fixed-input denoiser-only runtime loop

**Recommendation:** Good baseline, but incomplete by itself

**Pros**

- cheapest way to prove Android runtime integration
- isolates ORT/QNN/device integration from the rest of the pipeline

**Cons**

- does not yet connect to the app experience strongly enough on its own

### Option 3: Denoiser-first runtime loop with UI closure

**Recommendation:** Yes

**Pros**

- still isolates the hardest runtime risk
- delivers a real on-device app interaction instead of just a test harness
- keeps the next step toward full generation obvious

**Cons**

- requires slightly more wiring than a pure test harness

## System Architecture

The Android minimal loop is split into four bounded layers.

### 1. UI Layer

Files:

- `GenerationScreen`
- `GenerationViewModel`

Responsibilities:

- collect user input fields
- start generation
- display success/failure status
- display output path, timing, profiling path, and `qnnActive`

This layer must not call ORT APIs directly.

### 2. Orchestration Layer

Primary file:

- `GenerationOrchestrator`

Responsibilities:

- translate UI requests into runtime requests
- call the denoiser runtime engine
- normalize runtime success and failure into a `GenerationResult`

This is the main control point for the Android minimal loop.

### 3. Runtime Layer

Primary files:

- `OrtQnnConfig`
- `OrtSessionFactory`
- new `OrtQnnDenoiserEngine`

Responsibilities:

- validate that required Android ORT/QNN assets are present
- create an `ORT + QNN` session with CPU fallback disabled
- load denoiser inputs
- execute one inference
- persist output and profiling artifacts
- report whether QNN was truly active

### 4. Artifact Management Layer

Primary file:

- `ArtifactManager`

Responsibilities:

- resolve denoiser model/context files
- resolve profiling output path
- resolve fixed or precomputed denoiser input assets
- resolve inference output location

This layer should isolate all path and file assumptions from the rest of runtime code.

## Data Boundary

### UI Inputs

The UI can keep these fields:

- `prompt`
- `negative prompt`
- `steps`
- `cfg`

But in this stage they are allowed to function primarily as preserved request fields rather than all driving the true runtime path.

### Runtime Inputs

The actual denoiser runtime may use:

- fixed demo latent input
- fixed timestep input
- precomputed `cond` embeddings
- precomputed `uncond` embeddings

These may be packaged as Android assets or copied runtime files.

### Runtime Outputs

The runtime must produce at least:

- one output tensor/raw file or equivalent saved inference output
- one profiling file or equivalent provider evidence
- structured metadata for the UI:
  - `totalDurationMs`
  - `denoiseDurationMs`
  - `profilingPath`
  - `qnnActive`
  - `imagePath` or equivalent output location

## Key New Runtime Component

The most important new implementation should be a concrete engine:

- `OrtQnnDenoiserEngine`

This object implements `GenerationEngine` and is responsible for:

- loading denoiser runtime artifacts
- creating the QNN-backed ORT session
- executing the denoiser once
- returning a structured `GenerationResult`

This allows the current fake engine in `MainActivity` to be replaced cleanly with a real runtime engine.

## Execution Order

### Step 1: Replace the fake engine with a real denoiser engine

Do not begin with full text-to-image. Begin with one real Android local denoiser session.

### Step 2: Run fixed or precomputed denoiser inputs

Do not depend on text encoder availability yet. Use fixed or precomputed input assets to prove:

- session creation
- one execute path
- profiling generation

### Step 3: Surface runtime results in the UI

The app must display:

- output location
- total time
- denoise time
- profiling location
- whether QNN was active

Failures must be explicit, not a vague “generation failed”.

### Step 4: Keep prompt fields wired as future-facing placeholders

The request object should continue to preserve prompt-related UI input, even if the runtime path temporarily uses fixed or precomputed denoiser inputs.

## Stop-Loss Rules

### Stop-Loss 1: Missing Android runtime artifacts

If the required AAR, Qualcomm `.so`, or denoiser artifacts are absent, stop immediately and turn the problem into a clear missing-files error.

### Stop-Loss 2: Provider unavailable

If `QNNExecutionProvider` is not visible on Android, stop before inference and classify the problem as Android ORT/QNN integration.

### Stop-Loss 3: Session succeeds but execute fails

If the session exists but inference fails, classify it as runtime input, artifact, or provider execution failure.

### Stop-Loss 4: Execute succeeds but QNN evidence is missing

Do not call the stage successful if the app runs but cannot provide profiling or equivalent evidence that QNN was actually used with CPU fallback disabled.

## Acceptance Criteria

### M1: Android Denoiser Runtime Loop

On the device, one button press must:

- create a local ORT + QNN session
- run one real denoiser inference
- persist an output artifact
- update the UI with success metadata

### M2: QNN Evidence

The run must demonstrate:

- `QNNExecutionProvider` availability
- CPU fallback disabled for the denoiser session
- profiling file or equivalent proof of QNN activity

### M3: UI Closure

The app must:

- accept user input
- trigger the real runtime engine
- show success or failure state
- no longer depend on a fake generation engine

## File Plan

Likely files for this stage:

- modify `android/app/src/main/java/com/example/animanpu/MainActivity.kt`
- modify `android/app/src/main/java/com/example/animanpu/ui/GenerationScreen.kt`
- modify `android/app/src/main/java/com/example/animanpu/ui/GenerationViewModel.kt`
- modify `android/app/src/main/java/com/example/animanpu/runtime/GenerationOrchestrator.kt`
- modify `android/app/src/main/java/com/example/animanpu/runtime/ArtifactManager.kt`
- modify `android/app/src/main/java/com/example/animanpu/runtime/OrtSessionFactory.kt`
- modify `android/app/src/main/java/com/example/animanpu/runtime/GenerationResult.kt` if more runtime evidence fields are needed
- create `android/app/src/main/java/com/example/animanpu/runtime/OrtQnnDenoiserEngine.kt`
- optionally create a small runtime input spec/helper for precomputed assets
- update Android tests for the new engine and orchestrator behavior
- update `docs/manual/android-validation.md`

## Success Definition

This stage is complete when Android has a real local denoiser-first runtime loop that:

- executes one ORT+QNN denoiser inference on-device
- produces profiling or equivalent provider evidence
- is triggered from the app UI
- reports success or failure clearly

This is still aligned with the main project objective because it is the smallest Android-side proof of local NPU-backed Anima inference without pretending the entire text-to-image pipeline is already finished.
