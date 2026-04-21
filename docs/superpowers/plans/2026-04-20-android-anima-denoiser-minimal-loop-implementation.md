# Android Anima Denoiser Minimal Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the fake Android generation loop with a real denoiser-first local runtime path that assumes exported `Anima` denoiser artifacts already exist, performs one on-device `ORT + QNN` inference, captures profiling evidence, and reports success/failure through the app UI.

**Architecture:** Preserve the four-layer split from the approved spec: UI (`GenerationScreen`, `GenerationViewModel`), orchestration (`GenerationOrchestrator`), runtime (`OrtQnnDenoiserEngine`, `OrtSessionFactory`, `OrtQnnConfig`), and artifacts (`ArtifactManager`, fixed-input asset descriptor). To keep the repo buildable without proprietary blobs, use a fakeable session-factory abstraction plus a reflection-based Android implementation. Add the local ORT QNN AAR dependency conditionally so Android unit tests still compile when the AAR is absent.

**Tech Stack:** Kotlin, Compose, Android Gradle plugin, JUnit4, kotlinx-coroutines-test, reflection-based Android runtime access to `ai.onnxruntime`, Markdown

---

## Prerequisites

Before Task 1, verify these conditions.

- The Android app currently builds as a shell with a fake engine.
- Current Android unit tests pass.
- `android/app/libs/README.md` is still the source of truth for manually supplied ORT/QNN Android artifacts.
- This stage does not require the real AAR or Qualcomm `.so` files to exist on the development host, but it must produce clear runtime errors if they are absent on-device.

Run:

```bash
mkdir -p android/app/src/main/assets/runtime android/app/src/test/java/com/example/animanpu/runtime
```

Expected: runtime asset and test directories exist.

## File Structure

- Modify: `android/app/build.gradle.kts` — conditional local AAR wiring and runtime packaging settings
- Modify: `android/app/src/main/java/com/example/animanpu/MainActivity.kt` — replace fake engine wiring with the real denoiser engine
- Modify: `android/app/src/main/java/com/example/animanpu/ui/GenerationScreen.kt` — show richer runtime evidence
- Modify: `android/app/src/main/java/com/example/animanpu/ui/GenerationViewModel.kt` — propagate richer runtime state
- Modify: `android/app/src/main/java/com/example/animanpu/runtime/GenerationOrchestrator.kt` — normalize runtime result handling
- Modify: `android/app/src/main/java/com/example/animanpu/runtime/GenerationResult.kt` — include runtime evidence fields
- Modify: `android/app/src/main/java/com/example/animanpu/runtime/ArtifactManager.kt` — resolve denoiser assets, profiling output, and output tensor path
- Modify: `android/app/src/main/java/com/example/animanpu/runtime/OrtSessionFactory.kt` — define fakeable factory/handle interfaces and a reflective Android implementation
- Modify: `android/app/src/main/java/com/example/animanpu/runtime/OrtQnnConfig.kt` — keep provider options aligned with Android QNN requirements
- Create: `android/app/src/main/java/com/example/animanpu/runtime/DenoiserInputAssetSpec.kt` — fixed/precomputed input asset names
- Create: `android/app/src/main/java/com/example/animanpu/runtime/OrtQnnDenoiserEngine.kt` — concrete denoiser runtime engine
- Create: `android/app/src/test/java/com/example/animanpu/runtime/ArtifactManagerTest.kt`
- Create: `android/app/src/test/java/com/example/animanpu/runtime/OrtQnnDenoiserEngineTest.kt`
- Modify: `android/app/src/test/java/com/example/animanpu/runtime/GenerationOrchestratorTest.kt`
- Optionally create: `android/app/src/test/java/com/example/animanpu/ui/GenerationViewModelTest.kt`
- Modify: `android/app/libs/README.md` — document denoiser artifacts and precomputed inputs needed for first run
- Modify: `docs/manual/android-validation.md` — rewrite validation for the denoiser-first loop

### Task 1: Add denoiser asset descriptors, runtime output metadata, and artifact tests

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

Add a small input descriptor:

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

Expand `ArtifactManager` to add:

- `latentInput()`
- `timestepInput()`
- `condInput()`
- `uncondInput()`
- `outputTensor()`
- `ensureRuntimeDirectories()`

Update `GenerationResult` to include Android runtime evidence:

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

### Task 2: Add fakeable ORT/QNN runtime abstractions and denoiser engine tests

**Files:**
- Modify: `android/app/src/main/java/com/example/animanpu/runtime/OrtSessionFactory.kt`
- Create: `android/app/src/main/java/com/example/animanpu/runtime/OrtQnnDenoiserEngine.kt`
- Create: `android/app/src/test/java/com/example/animanpu/runtime/OrtQnnDenoiserEngineTest.kt`
- Modify: `android/app/src/test/java/com/example/animanpu/runtime/OrtQnnConfigTest.kt`

- [ ] **Step 1: Write the failing engine tests**

```kotlin
// android/app/src/test/java/com/example/animanpu/runtime/OrtQnnDenoiserEngineTest.kt
package com.example.animanpu.runtime

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

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
cd android && ./gradlew :app:testDebugUnitTest --tests com.example.animanpu.runtime.OrtQnnDenoiserEngineTest
```

Expected: FAIL because `OrtQnnDenoiserEngine` and fakeable `OrtSessionFactory` contracts do not exist yet.

- [ ] **Step 3: Write the minimal implementation**

Refactor `OrtSessionFactory.kt` into abstractions that do not require compile-time ORT classes:

```kotlin
interface OrtSessionHandle {
    fun run(outputPath: File): Boolean
}

interface OrtSessionFactory {
    fun providerVisible(): Boolean
    fun createDenoiserSession(modelPath: File, config: OrtQnnConfig): OrtSessionHandle
}
```

Add a real Android implementation that uses reflection against `ai.onnxruntime` so the app module can still compile when the local AAR is absent on the development host.

Add `OrtQnnDenoiserEngine` that:

- ensures runtime directories exist
- checks artifact presence before session creation
- checks `providerVisible()`
- builds `OrtQnnConfig` with CPU fallback disabled
- creates a session
- runs one inference into `outputTensor()`
- returns a populated `GenerationResult`
- throws or classifies a clear runtime error when session creation or execute fails

Update `OrtQnnConfigTest` if provider option expectations need to include any additional denoiser-loop fields.

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

### Task 3: Wire the real engine into orchestration, view-model, and UI

**Files:**
- Modify: `android/app/src/main/java/com/example/animanpu/runtime/GenerationOrchestrator.kt`
- Modify: `android/app/src/main/java/com/example/animanpu/ui/GenerationViewModel.kt`
- Modify: `android/app/src/main/java/com/example/animanpu/ui/GenerationScreen.kt`
- Modify: `android/app/src/main/java/com/example/animanpu/MainActivity.kt`
- Modify: `android/app/src/test/java/com/example/animanpu/runtime/GenerationOrchestratorTest.kt`
- Optional: create `android/app/src/test/java/com/example/animanpu/ui/GenerationViewModelTest.kt`

- [ ] **Step 1: Extend orchestrator tests with richer runtime metadata**

Update `GenerationOrchestratorTest.kt` to assert:

- `sessionCreated = true`
- `outputTensorPath` is preserved
- `qnnActive = true`

Add a failure-path test where the engine raises a meaningful runtime error and confirm the orchestrator does not erase it.

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
cd android && ./gradlew :app:testDebugUnitTest --tests com.example.animanpu.runtime.GenerationOrchestratorTest
```

Expected: FAIL because the orchestrator/result shape has changed.

- [ ] **Step 3: Write the minimal implementation**

Implementation goals:

- `MainActivity` must stop creating the fake engine
- `MainActivity` should build:
  - `ArtifactManager(filesDir)`
  - the reflection-backed Android `OrtSessionFactory`
  - `OrtQnnDenoiserEngine`
  - `GenerationOrchestrator`
  - `GenerationViewModel`
- `GenerationViewModel` must keep prompt-related fields but drive the real denoiser loop
- `GenerationScreen` must display at least:
  - output tensor path or image/output location
  - total duration
  - denoise duration
  - profiling path
  - `qnnActive`
  - clear runtime error text

Keep the UI intentionally diagnostic and minimal.

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
cd android && ./gradlew :app:testDebugUnitTest --tests com.example.animanpu.runtime.GenerationOrchestratorTest
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

### Task 4: Wire Android build config and runtime payload expectations

**Files:**
- Modify: `android/app/build.gradle.kts`
- Modify: `android/app/libs/README.md`
- Modify: `docs/manual/android-validation.md`

- [ ] **Step 1: Add conditional local AAR wiring and packaging rules**

Update `android/app/build.gradle.kts` to:

- include the local `onnxruntime-android-qnn.aar` **only if the file exists**
- keep the module buildable without proprietary artifacts on the host
- set conservative JNI packaging options so Qualcomm `.so` files are preserved when present

Recommended pattern:

```kotlin
val qnnAar = file("libs/onnxruntime-android-qnn.aar")
if (qnnAar.exists()) {
    dependencies {
        implementation(files(qnnAar))
    }
}

android {
    packaging {
        jniLibs {
            useLegacyPackaging = true
        }
    }
}
```

If Gradle syntax requires moving the conditional differently, keep the intent the same.

- [ ] **Step 2: Update runtime payload docs**

`android/app/libs/README.md` must list the first-loop payload clearly:

- custom ORT AAR
- Qualcomm runtime `.so` files
- denoiser context/model artifacts
- fixed/precomputed denoiser input assets

`docs/manual/android-validation.md` must be rewritten for the denoiser-first loop:

- install the app
- ensure runtime assets are present
- tap Generate once
- confirm output tensor/output artifact and profiling file
- confirm QNN evidence and disabled CPU fallback

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

- which Android artifacts must be provided manually
- where to place them
- how to run the app
- what evidence proves success
- what gap still remains before full text-to-image

## Self-Review

- Spec coverage checked: the plan covers the real denoiser engine, artifact management, fakeable Android ORT/QNN session creation, UI closure, and updated device validation.
- Placeholder scan checked: no unresolved placeholders or vague future tasks remain.
- Type consistency checked: `GenerationResult`, `ArtifactManager`, `OrtSessionFactory`, and `OrtQnnDenoiserEngine` responsibilities are aligned across tasks and tests.
