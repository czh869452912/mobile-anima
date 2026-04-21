# Android Anima Denoiser Minimal Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the current Android shell into a real denoiser-first local runtime loop that assumes exported `Anima` denoiser artifacts already exist, runs one on-device `ORT + QNN` inference, captures QNN evidence, and surfaces runtime results in the UI.

**Architecture:** Keep the Android closure split into four layers: UI (`GenerationScreen`, `GenerationViewModel`), orchestration (`GenerationOrchestrator`), runtime (`OrtQnnDenoiserEngine`, `OrtSessionFactory`, `OrtQnnConfig`), and artifact management (`ArtifactManager` plus a small fixed-input asset descriptor). Use fixed/precomputed denoiser inputs first, not full prompt-to-image. The plan must replace the fake `GenerationEngine` in `MainActivity` without pretending text encoder or VAE are complete.

**Tech Stack:** Kotlin, Android app module, Compose, JUnit4, coroutine test, local `onnxruntime-android-qnn.aar`, Qualcomm `arm64-v8a` runtime `.so` files, Markdown

---

## Prerequisites

Before Task 1, verify these assumptions.

- Android project already builds with the current shell code
- Android tests currently pass
- The app still depends on a fake engine in `MainActivity`
- The denoiser runtime artifacts will be provided externally, not generated in this stage
- `android/app/libs/README.md` remains the source of truth for required AAR and Qualcomm libraries

Run:

```bash
mkdir -p android/app/src/main/assets/runtime android/app/src/test/java/com/example/animanpu/runtime
```

Expected: runtime asset directory and Android unit test directory structure are present.

## File Structure

- Modify: `android/app/build.gradle.kts` — wire the local ORT AAR and asset packaging expectations
- Modify: `android/app/src/main/java/com/example/animanpu/MainActivity.kt` — replace the fake engine with the real denoiser engine wiring
- Modify: `android/app/src/main/java/com/example/animanpu/ui/GenerationScreen.kt` — show richer runtime result and failure state
- Modify: `android/app/src/main/java/com/example/animanpu/ui/GenerationViewModel.kt` — support richer status and engine-driven results
- Modify: `android/app/src/main/java/com/example/animanpu/runtime/GenerationOrchestrator.kt` — orchestrate real runtime results and failure mapping
- Modify: `android/app/src/main/java/com/example/animanpu/runtime/GenerationResult.kt` — add output/profiling/provider evidence fields as needed
- Modify: `android/app/src/main/java/com/example/animanpu/runtime/ArtifactManager.kt` — resolve denoiser assets, profiling output, and output file paths
- Modify: `android/app/src/main/java/com/example/animanpu/runtime/OrtSessionFactory.kt` — build real ORT session options and create sessions via injectable abstractions
- Modify: `android/app/src/main/java/com/example/animanpu/runtime/OrtQnnConfig.kt` — keep provider options builder aligned with Android runtime needs
- Create: `android/app/src/main/java/com/example/animanpu/runtime/DenoiserInputAssetSpec.kt` — fixed/precomputed input asset descriptor
- Create: `android/app/src/main/java/com/example/animanpu/runtime/OrtQnnDenoiserEngine.kt` — concrete runtime engine for one denoiser inference
- Modify: `android/app/src/test/java/com/example/animanpu/runtime/GenerationOrchestratorTest.kt`
- Modify: `android/app/src/test/java/com/example/animanpu/runtime/OrtQnnConfigTest.kt`
- Create: `android/app/src/test/java/com/example/animanpu/runtime/ArtifactManagerTest.kt`
- Create: `android/app/src/test/java/com/example/animanpu/runtime/OrtQnnDenoiserEngineTest.kt`
- Optionally create: `android/app/src/test/java/com/example/animanpu/ui/GenerationViewModelTest.kt`
- Modify: `android/app/libs/README.md` — document required denoiser runtime artifacts and fixed input assets
- Modify: `docs/manual/android-validation.md` — update manual validation for denoiser-first loop

### Task 1: Add artifact/input descriptors and output metadata

**Files:**
- Create: `android/app/src/main/java/com/example/animanpu/runtime/DenoiserInputAssetSpec.kt`
- Modify: `android/app/src/main/java/com/example/animanpu/runtime/ArtifactManager.kt`
- Modify: `android/app/src/main/java/com/example/animanpu/runtime/GenerationResult.kt`
- Create: `android/app/src/test/java/com/example/animanpu/runtime/ArtifactManagerTest.kt`

- [ ] **Step 1: Write the failing artifact tests**

```kotlin
// android/app/src/test/java/com/example/animanpu/runtime/ArtifactManagerTest.kt
package com.example.animanpu.runtime

import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ArtifactManagerTest {
    @Test
    fun resolves_denoiser_runtime_inputs_and_outputs_under_runtime_directory() {
        val root = createTempDir(prefix = "anima-runtime-")
        val manager = ArtifactManager(root)

        assertEquals(File(root, "runtime/denoiser_ctx.onnx"), manager.denoiserContextOnnx())
        assertEquals(File(root, "runtime/denoiser_qnn.bin"), manager.denoiserContextBin())
        assertEquals(File(root, "runtime/inputs/latent.raw"), manager.latentInput())
        assertEquals(File(root, "runtime/inputs/timestep.raw"), manager.timestepInput())
        assertEquals(File(root, "runtime/inputs/cond.raw"), manager.condInput())
        assertEquals(File(root, "runtime/inputs/uncond.raw"), manager.uncondInput())
        assertEquals(File(root, "runtime/outputs/denoiser_output.raw"), manager.outputTensor())
        assertEquals(File(root, "runtime/denoiser_profile.csv"), manager.profilingCsv())
    }

    @Test
    fun ensures_runtime_parent_directories_exist() {
        val root = createTempDir(prefix = "anima-runtime-")
        val manager = ArtifactManager(root)

        manager.ensureRuntimeDirectories()

        assertTrue(File(root, "runtime/inputs").isDirectory)
        assertTrue(File(root, "runtime/outputs").isDirectory)
    }
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
cd android && ./gradlew :app:testDebugUnitTest --tests com.example.animanpu.runtime.ArtifactManagerTest
```

Expected: FAIL because the new `ArtifactManager` methods do not exist yet.

- [ ] **Step 3: Write the minimal implementation**

Add a small fixed-input descriptor:

```kotlin
// android/app/src/main/java/com/example/animanpu/runtime/DenoiserInputAssetSpec.kt
package com.example.animanpu.runtime

data class DenoiserInputAssetSpec(
    val latentFileName: String = "latent.raw",
    val timestepFileName: String = "timestep.raw",
    val condFileName: String = "cond.raw",
    val uncondFileName: String = "uncond.raw",
)
```

Expand `ArtifactManager` to resolve:

- `denoiserContextOnnx()`
- `denoiserContextBin()`
- `latentInput()`
- `timestepInput()`
- `condInput()`
- `uncondInput()`
- `outputTensor()`
- `profilingCsv()`
- `ensureRuntimeDirectories()`

Update `GenerationResult` to carry the minimum runtime evidence:

```kotlin
data class GenerationResult(
    val imagePath: String,
    val totalDurationMs: Long,
    val denoiseDurationMs: Long,
    val profilingPath: String,
    val qnnActive: Boolean,
    val sessionCreated: Boolean = false,
    val outputTensorPath: String = "",
)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
cd android && ./gradlew :app:testDebugUnitTest --tests com.example.animanpu.runtime.ArtifactManagerTest
```

Expected: `BUILD SUCCESSFUL`.

- [ ] **Step 5: Commit**

```bash
git add android/app/src/main/java/com/example/animanpu/runtime/DenoiserInputAssetSpec.kt \
  android/app/src/main/java/com/example/animanpu/runtime/ArtifactManager.kt \
  android/app/src/main/java/com/example/animanpu/runtime/GenerationResult.kt \
  android/app/src/test/java/com/example/animanpu/runtime/ArtifactManagerTest.kt
git commit -m "test: add android denoiser artifact management"
```

### Task 2: Add real ORT session abstractions and QNN runtime engine tests

**Files:**
- Modify: `android/app/src/main/java/com/example/animanpu/runtime/OrtSessionFactory.kt`
- Create: `android/app/src/main/java/com/example/animanpu/runtime/OrtQnnDenoiserEngine.kt`
- Create: `android/app/src/test/java/com/example/animanpu/runtime/OrtQnnDenoiserEngineTest.kt`
- Modify: `android/app/src/test/java/com/example/animanpu/runtime/OrtQnnConfigTest.kt`

- [ ] **Step 1: Write the failing engine tests**

```kotlin
// android/app/src/test/java/com/example/animanpu/runtime/OrtQnnDenoiserEngineTest.kt
package com.example.animanpu.runtime

import java.io.File
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class OrtQnnDenoiserEngineTest {
    @Test
    fun returns_missing_artifact_failure_before_session_creation() = runTest {
        val root = createTempDir(prefix = "anima-engine-")
        val artifacts = ArtifactManager(root)
        artifacts.ensureRuntimeDirectories()

        val engine = OrtQnnDenoiserEngine(
            artifactManager = artifacts,
            sessionFactory = FakeOrtSessionFactory(providerVisible = true, executeSucceeds = true),
            backendPath = "/data/local/tmp/libQnnHtp.so",
        )

        val result = engine.generate(
            GenerationRequest(1024, 1024, "cat astronaut", "blurry", 8, 5.0f)
        )

        assertFalse(result.sessionCreated)
        assertFalse(result.qnnActive)
    }

    @Test
    fun returns_success_when_fake_session_executes() = runTest {
        val root = createTempDir(prefix = "anima-engine-")
        val artifacts = ArtifactManager(root)
        artifacts.ensureRuntimeDirectories()
        artifacts.denoiserContextOnnx().writeText("ctx")
        artifacts.denoiserContextBin().writeText("bin")
        artifacts.latentInput().writeText("latent")
        artifacts.timestepInput().writeText("time")
        artifacts.condInput().writeText("cond")
        artifacts.uncondInput().writeText("uncond")

        val engine = OrtQnnDenoiserEngine(
            artifactManager = artifacts,
            sessionFactory = FakeOrtSessionFactory(providerVisible = true, executeSucceeds = true),
            backendPath = "/data/local/tmp/libQnnHtp.so",
        )

        val result = engine.generate(
            GenerationRequest(1024, 1024, "cat astronaut", "blurry", 8, 5.0f)
        )

        assertTrue(result.sessionCreated)
        assertTrue(result.qnnActive)
        assertEquals(artifacts.outputTensor().path, result.outputTensorPath)
        assertEquals(artifacts.profilingCsv().path, result.profilingPath)
    }
}
```

The fake session factory used in the test should model three things:

- provider visibility
- session creation
- execute success/failure

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
cd android && ./gradlew :app:testDebugUnitTest --tests com.example.animanpu.runtime.OrtQnnDenoiserEngineTest
```

Expected: FAIL because `OrtQnnDenoiserEngine` and fakeable `OrtSessionFactory` contracts do not exist yet.

- [ ] **Step 3: Write the minimal implementation**

Refactor `OrtSessionFactory` into a testable abstraction. For example:

```kotlin
interface OrtSessionHandle {
    fun run(outputPath: File): Boolean
}

interface OrtSessionFactory {
    fun providerVisible(): Boolean
    fun createDenoiserSession(modelPath: File, config: OrtQnnConfig): OrtSessionHandle
}
```

Then add `OrtQnnDenoiserEngine` that:

- checks required artifact files first
- checks `providerVisible()`
- builds a config with CPU fallback disabled
- creates a session
- runs one inference
- writes the output/profiling file paths into `GenerationResult`
- classifies failures into explicit error messages

Do not call real ORT APIs yet in unit tests; keep the engine fakeable.

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
cd android && ./gradlew :app:testDebugUnitTest \
  --tests com.example.animanpu.runtime.OrtQnnDenoiserEngineTest \
  --tests com.example.animanpu.runtime.OrtQnnConfigTest
```

Expected: `BUILD SUCCESSFUL`.

- [ ] **Step 5: Commit**

```bash
git add android/app/src/main/java/com/example/animanpu/runtime/OrtSessionFactory.kt \
  android/app/src/main/java/com/example/animanpu/runtime/OrtQnnDenoiserEngine.kt \
  android/app/src/test/java/com/example/animanpu/runtime/OrtQnnDenoiserEngineTest.kt \
  android/app/src/test/java/com/example/animanpu/runtime/OrtQnnConfigTest.kt
git commit -m "feat: add android ort qnn denoiser engine"
```

### Task 3: Wire the real engine into orchestration and UI state

**Files:**
- Modify: `android/app/src/main/java/com/example/animanpu/runtime/GenerationOrchestrator.kt`
- Modify: `android/app/src/main/java/com/example/animanpu/ui/GenerationViewModel.kt`
- Modify: `android/app/src/main/java/com/example/animanpu/ui/GenerationScreen.kt`
- Modify: `android/app/src/main/java/com/example/animanpu/MainActivity.kt`
- Modify: `android/app/src/test/java/com/example/animanpu/runtime/GenerationOrchestratorTest.kt`
- Optional: create `android/app/src/test/java/com/example/animanpu/ui/GenerationViewModelTest.kt`

- [ ] **Step 1: Extend orchestrator tests to assert richer runtime metadata**

Update `GenerationOrchestratorTest.kt` so the expected result includes:

- `sessionCreated = true`
- `outputTensorPath`
- `qnnActive = true`

Add a failure-path test where the engine throws or returns a non-QNN result and ensure the orchestrator surfaces a meaningful failure.

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
cd android && ./gradlew :app:testDebugUnitTest --tests com.example.animanpu.runtime.GenerationOrchestratorTest
```

Expected: FAIL because the orchestrator and result shape have changed.

- [ ] **Step 3: Write the minimal implementation**

Implementation goals:

- `MainActivity` must stop constructing the fake engine
- `MainActivity` should construct:
  - `ArtifactManager(filesDir)`
  - the real `OrtSessionFactory` implementation or a bootstrap-friendly wrapper
  - `OrtQnnDenoiserEngine`
  - `GenerationOrchestrator`
  - `GenerationViewModel`
- `GenerationViewModel` must preserve prompt fields but drive the real engine path
- `GenerationScreen` must display at least:
  - output tensor path or image path
  - total duration
  - denoise duration
  - profiling path
  - `qnnActive`
  - clear runtime error text

Do not overbuild the UI. Keep it minimal and diagnostic.

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
cd android && ./gradlew :app:testDebugUnitTest \
  --tests com.example.animanpu.runtime.GenerationOrchestratorTest
```

Expected: `BUILD SUCCESSFUL`.

- [ ] **Step 5: Commit**

```bash
git add android/app/src/main/java/com/example/animanpu/MainActivity.kt \
  android/app/src/main/java/com/example/animanpu/runtime/GenerationOrchestrator.kt \
  android/app/src/main/java/com/example/animanpu/ui/GenerationViewModel.kt \
  android/app/src/main/java/com/example/animanpu/ui/GenerationScreen.kt \
  android/app/src/test/java/com/example/animanpu/runtime/GenerationOrchestratorTest.kt
git commit -m "feat: wire android denoiser loop into ui"
```

### Task 4: Wire Android build inputs and runtime-file expectations

**Files:**
- Modify: `android/app/build.gradle.kts`
- Modify: `android/app/libs/README.md`
- Modify: `docs/manual/android-validation.md`

- [ ] **Step 1: Add the local AAR dependency and document the runtime payload**

`android/app/build.gradle.kts` should:

- include the local `onnxruntime-android-qnn.aar` from `android/app/libs/`
- include any packaging rules needed so the Qualcomm `.so` files are not stripped or excluded
- remain conservative and avoid introducing unrelated Android dependencies

Minimal example shape:

```kotlin
dependencies {
    implementation(files("libs/onnxruntime-android-qnn.aar"))
}

android {
    packaging {
        jniLibs {
            useLegacyPackaging = true
        }
    }
}
```

- [ ] **Step 2: Update documentation for denoiser-only closure**

`android/app/libs/README.md` must now list the complete first-loop payload, including:

- custom ORT AAR
- Qualcomm runtime `.so` files
- denoiser context/model artifacts
- precomputed input assets

`docs/manual/android-validation.md` must be rewritten for the denoiser-first loop:

- install app
- confirm assets are present
- tap Generate
- confirm output tensor and profiling file appear
- confirm `qnnActive` evidence and no CPU fallback

- [ ] **Step 3: Run Android unit tests again**

Run:

```bash
cd android && ./gradlew testDebugUnitTest
```

Expected: `BUILD SUCCESSFUL`.

- [ ] **Step 4: Commit**

```bash
git add android/app/build.gradle.kts android/app/libs/README.md docs/manual/android-validation.md
git commit -m "docs: add android denoiser runtime prerequisites"
```

### Task 5: Final verification and device-readiness handoff

**Files:**
- No new code files required unless cleanup is needed

- [ ] **Step 1: Run fresh Android verification**

Run:

```bash
cd android && ./gradlew testDebugUnitTest
```

Expected: `BUILD SUCCESSFUL`.

- [ ] **Step 2: Recheck repo status**

Run:

```bash
git status --short
```

Expected: no uncommitted changes remain.

- [ ] **Step 3: Summarize the device-side manual loop for the user**

The final implementation handoff must include:

- which Android artifacts the user must provide manually
- where to place them
- how to run the app
- what evidence proves success
- what the remaining gap is before full text-to-image

## Self-Review

- Spec coverage checked: the plan covers the real denoiser engine, runtime artifact management, Android ORT/QNN session creation, UI closure, and device validation evidence.
- Placeholder scan checked: no unresolved placeholders or vague future tasks remain.
- Type consistency checked: `GenerationResult`, `ArtifactManager`, `OrtSessionFactory`, and `OrtQnnDenoiserEngine` responsibilities are aligned across tasks and tests.
