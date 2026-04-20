# Anima Snapdragon 8 Elite NPU PoC Design

**Status:** Approved for planning

**Owner:** Codex + user

**Date:** 2026-04-20

## Goal

Build the lowest-cost Android proof of concept that can generate images on-device from the `circlestone-labs/Anima` model family while proving that Snapdragon 8 Elite's Qualcomm NPU/HTP is actually involved in inference.

The first milestone is not a product-ready app. The first milestone is a reproducible, minimal, end-to-end PoC with a crude Android UI and hard evidence that the most expensive denoising compute runs through the Qualcomm execution stack.

## Non-Goals

- Full productization
- Broad device compatibility beyond the initial Snapdragon 8 Elite target
- Dynamic resolution support
- Dynamic prompt length support
- Full-pipeline NPU offload in v1
- Best possible latency or image quality tuning in v1

## User-Facing Scope

The Android app may be visually simple. It only needs to provide:

- Resolution selector with a small fixed set of presets
- `prompt`
- `negative prompt`
- `steps`
- `cfg`
- Generate action
- Result image preview and save/export capability
- Basic timing/profiling display

## Primary Constraints

### Business Constraint

The user wants the lowest-cost feasible path, not the cleanest long-term architecture.

### Technical Constraints

- Target hardware is Snapdragon 8 Elite based Android devices
- Qualcomm official toolchain is acceptable
- Static shapes are acceptable in the PoC
- A mixed-backend pipeline is acceptable in the PoC
- Android UI can be minimal

### Deployment Constraints

- Resolution is fixed to a short allowlist in v1
- Prompt max length is fixed in v1
- Batch layout for CFG is fixed in v1
- Seeded, deterministic reference runs are required for validation

## Recommended Approach

Use a mixed-backend pipeline.

- Run `scheduler`, tokenization, CFG orchestration, and general control logic on the host side
- Run the `Anima` denoiser/transformer on Qualcomm NPU through `QNN`
- Keep the text encoder and VAE on CPU or GPU for the first milestone

This is the recommended approach because it maximizes the chance of proving real NPU participation while minimizing integration and model-conversion scope.

## Alternatives Considered

### Option 1: Mixed backend with only core denoiser on NPU

**Recommendation:** Yes

**Pros**

- Lowest engineering cost
- Fastest route to a working PoC
- Most concentrated validation of the hard part
- Limits initial model conversion risk to one critical subgraph

**Cons**

- Not full-pipeline NPU acceleration
- Requires multiple runtime paths in one generation flow

### Option 2: Full pipeline through ONNX Runtime + QNN EP

**Recommendation:** Not for v1

**Pros**

- Cleaner eventual app architecture
- Consistent runtime abstraction across submodels

**Cons**

- Much higher conversion risk up front
- More simultaneous failure points
- Larger debugging surface

### Option 3: Native QNN/QAIRT-only integration

**Recommendation:** Not for v1

**Pros**

- Maximum control and potentially best performance

**Cons**

- Highest engineering cost
- More JNI/C++ integration work
- Slower time to first proof

## System Architecture

The PoC is split into three layers.

### 1. Android Shell Layer

Responsibilities:

- Simple UI for input parameters
- Start/cancel generation
- Render output image
- Display status, latency, and profiling summary

### 2. Inference Orchestration Layer

Responsibilities:

- Convert user inputs into fixed-shape runtime inputs
- Tokenize prompts
- Run text encoding
- Prepare `cond` and `uncond` batches for CFG
- Execute iterative denoising loop
- Run VAE decode
- Return output bitmap

### 3. Runtime Layer

Responsibilities:

- Manage sessions for text encoder, denoiser, and VAE
- Bind the denoiser session to Qualcomm `QNN`
- Manage model files, context binaries, and profiling artifacts

## Model Partitioning

The initial pipeline is intentionally split into three model units.

### Text Encoder

- Input: tokenized prompt and negative prompt
- Output: conditioning embeddings
- Runtime target in v1: CPU or GPU
- Reason: runs once per request, lower payoff than denoiser

### Denoiser / Transformer

- Input: latent, timestep, conditioning embeddings
- Output: denoised latent update
- Runtime target in v1: Qualcomm `QNN/HTP`
- Reason: repeated across sampling steps and contains the dominant compute cost

### VAE Decoder

- Input: final latent
- Output: RGB image tensor / bitmap
- Runtime target in v1: CPU or GPU
- Reason: lower priority than denoiser for proving NPU value

## Fixed PoC Runtime Parameters

To minimize conversion and runtime risk, v1 intentionally constrains inputs.

- Supported resolutions: `1024x1024`, `768x1024`, `1024x768`
- Default validation resolution: `1024x1024`
- Fixed max prompt length, for example `256` tokens
- Fixed CFG batch organization using paired `cond/uncond`
- `steps` exposed only in a small bounded set, such as `8`, `12`, and `20`
- `cfg` constrained to a bounded safe range, such as `3.0` to `7.0`

If multiple resolutions are too expensive during conversion, the fallback v1 scope is a single fixed resolution: `1024x1024`.

## Model Conversion Strategy

The conversion path should start from a minimal Python reference implementation, not directly from a ComfyUI graph.

### Reference Implementation Phase

Create a minimal Python inference path that performs:

1. Tokenization
2. Text encoding
3. CFG batch preparation
4. Iterative denoising
5. VAE decode

This reference path is the correctness baseline for all later exports.

### Export Phase

Export only the denoiser first.

Requirements:

- Static input shapes
- Explicit input and output boundaries
- Deterministic test inputs
- No dynamic resolution logic in the exported graph

### Quantization Phase

Start with QDQ-style quantization aimed at `int8` execution where supported.

Acceptance for this phase is not final image perfection. Acceptance is that the quantized denoiser remains semantically usable and can be executed through the Qualcomm path.

### QNN Compilation Phase

Compile the denoiser graph for Qualcomm execution and produce deployable context artifacts.

Expected artifact families:

- ONNX model for validation
- Quantized ONNX model
- QNN context representation or equivalent compiled runtime artifact
- Profiling outputs generated during test runs

## Android Runtime Design

### Runtime Choice

Use `ONNX Runtime` on Android with the `QNN Execution Provider` for the first milestone.

Reasoning:

- Lower app-side engineering cost than a native-first QNN integration
- Cleaner Kotlin/Java integration path
- Easier mixed backend orchestration

### Important Runtime Rule

The denoiser session must be configured to avoid silent CPU fallback during validation.

This is necessary because the PoC's main claim is NPU participation. A fallback-heavy run does not satisfy that goal.

### Session Allocation

- `textEncoderSession`: CPU or GPU
- `denoiserSession`: `QNN`
- `vaeSession`: CPU or GPU

### Provider Behavior Requirements

The runtime configuration for the denoiser must support:

- Explicit `QNN` backend selection
- Profiling output generation
- CPU fallback disablement during validation runs
- Context binary generation or loading when available

## Android App Flow

For one generation request, the app performs:

1. Validate UI inputs against fixed allowlist rules
2. Tokenize `prompt` and `negative prompt`
3. Run text encoding once
4. Create paired conditioning tensors for CFG
5. Initialize latent from fixed-shape setup
6. Execute `steps` denoiser calls in a loop using the QNN-backed session
7. Decode the final latent through VAE
8. Render and save the result
9. Save or display profiling summary

## Evidence Requirements

The PoC is only considered successful if it proves real Qualcomm NPU path usage.

Required evidence:

- The denoiser session is created with `QNN`
- CPU fallback is disabled during at least one validation run
- Profiling output is generated and retained
- The profile or runtime logs show Qualcomm/QNN/HTP execution involvement

Optional but useful evidence:

- Recorded screen capture of a complete generation run
- Side-by-side timing with and without the Qualcomm execution path

## Validation Strategy

### Stage 0: Reference Baseline

- Run one fixed prompt end-to-end on PC
- Save the reference image
- Save representative intermediate tensor statistics if feasible

### Stage 1: Denoiser Export Validation

- Export the denoiser to static-shape ONNX
- Run the exported model against deterministic test inputs
- Compare outputs numerically against the Python baseline within acceptable tolerance

### Stage 2: Qualcomm Runtime Validation

- Run the denoiser through the Qualcomm-targeted path
- Ensure session creation succeeds without CPU fallback
- Produce and retain profiling artifacts

### Stage 3: Android End-to-End Validation

- Execute one complete generation on target hardware
- Verify that the app remains stable across at least three runs
- Capture total time, denoising time, and profiling outputs

## Acceptance Criteria

The PoC is accepted when all of the following are true:

- A fixed prompt can generate an image end-to-end on a Snapdragon 8 Elite Android device
- The app provides the required minimal controls: resolution, prompt, negative prompt, steps, and cfg
- The denoiser executes through the Qualcomm path with CPU fallback disabled for the validation run
- Profiling artifacts show Qualcomm/QNN/HTP participation
- The generation flow is repeatable across at least three runs without crashes

## Main Risks

### Risk 1: Denoiser graph does not map cleanly to QNN

This is the highest-risk item. Unsupported ops, graph complexity, or export quirks may block deployment before Android integration even begins.

### Risk 2: Static-shape graph is still too large for practical device initialization

Even if the graph compiles, initialization time or memory use may make the PoC impractical without additional scope reduction.

### Risk 3: Quantization causes unacceptable semantic drift

The image does not need to be perfect in v1, but it must remain recognizable and consistent enough for a credible demo.

### Risk 4: False confidence caused by silent fallback

If fallback is enabled, the system may appear successful while not actually meeting the NPU proof objective.

## Stop-Loss Gates

Stop work on the Android shell and reassess if any of the following happens:

- The denoiser cannot be exported to a usable static-shape ONNX graph
- The Qualcomm path cannot initialize the denoiser without fallback
- Runtime memory pressure or initialization latency makes the denoiser unusable on the target phone
- Quantized output quality collapses to an unusable level

## Cheapest Execution Order

1. Build and validate the Python reference pipeline
2. Isolate and export only the denoiser
3. Prove denoiser execution through the Qualcomm path on desktop tooling
4. Build the thinnest possible Android shell around that path
5. Add VAE or text-encoder optimizations only after successful real-device proof

## Out of Scope for v1

- Device matrix testing
- Scheduler experimentation beyond what is needed for a stable demo
- Full prompt parser feature set
- Image-to-image, control inputs, or batch generation
- Background generation service
- Production telemetry, analytics, or cloud fallback

## Decision Summary

- Use a mixed backend architecture
- Prioritize only the denoiser for NPU offload in v1
- Use static shapes and hard constraints to reduce risk
- Use Android `ONNX Runtime + QNN EP` as the lowest-cost integration path
- Require hard evidence of Qualcomm execution through no-fallback validation and profiling artifacts

## Open Follow-Up Work

The next artifact after this spec is a detailed implementation plan that breaks the work into:

- Python reference pipeline setup
- Denoiser extraction and export
- Quantization and Qualcomm runtime validation
- Android runtime integration
- Minimal UI and end-to-end validation
