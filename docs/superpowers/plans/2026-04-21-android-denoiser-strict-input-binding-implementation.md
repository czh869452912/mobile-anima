# Android Denoiser Strict Input Binding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current mostly name-driven Android denoiser input binding with strict metadata-first heuristics that only accept unambiguous `timestep`, `latent`, `cond`, and `uncond` inputs.

**Architecture:** Keep the existing Android ORT bridge and session handle intact, but tighten `DenoiserInputBinding.kt` so candidate selection is driven first by static shape and supported element type, then by explicit conditioning names only where disambiguation is still needed. Prove the stricter behavior with focused unit tests and a small runtime integration test that exercises the live session-handle path with fake ORT sessions.

**Tech Stack:** Kotlin, Android runtime module, JVM unit tests via JUnit4 and coroutines test, local `/tmp/run_android_runtime_tests.sh` helper for sandbox-safe verification

---

### File Structure

- `android/app/src/main/java/com/example/animanpu/runtime/DenoiserInputBinding.kt`
  - owns strict candidate filtering and final binding for `latent`, `timestep`, `cond`, and `uncond`
- `android/app/src/test/java/com/example/animanpu/runtime/DenoiserInputBindingTest.kt`
  - owns unit tests for binding heuristics and ambiguity rejection
- `android/app/src/test/java/com/example/animanpu/runtime/StrictDenoiserBindingRuntimeTest.kt`
  - owns runtime-path tests that prove the stricter binding is actually used by `ReflectionOrtSessionFactory`

### Task 1: Make `latent` metadata-first and require explicit conditioning names

**Files:**
- Modify: `android/app/src/main/java/com/example/animanpu/runtime/DenoiserInputBinding.kt`
- Modify: `android/app/src/test/java/com/example/animanpu/runtime/DenoiserInputBindingTest.kt`

- [ ] **Step 1: Write the failing binding tests**

```kotlin
package com.example.animanpu.runtime

import org.junit.Assert.assertEquals
import org.junit.Test

class DenoiserInputBindingTest {
    @Test
    fun calculates_expected_float32_byte_size_for_static_shape() {
        val info = OrtJavaTensorInfo(
            elementType = OrtJavaElementType.FLOAT,
            shape = longArrayOf(1, 4, 128, 128),
        )

        val bytes = expectedByteCount(info)

        assertEquals(1L * 4L * 128L * 128L * 4L, bytes)
    }

    @Test
    fun rejects_dynamic_shapes() {
        val info = OrtJavaTensorInfo(
            elementType = OrtJavaElementType.FLOAT,
            shape = longArrayOf(1, -1, 128, 128),
        )

        try {
            expectedByteCount(info)
            throw AssertionError("Expected DenoiserBindingException")
        } catch (expected: DenoiserBindingException) {
            assertEquals(OrtRuntimeFailure.DYNAMIC_SHAPE_UNSUPPORTED, expected.failure)
        }
    }

    @Test
    fun rejects_unsupported_tensor_type() {
        val info = OrtJavaTensorInfo(
            elementType = OrtJavaElementType.UNKNOWN,
            shape = longArrayOf(1),
        )

        try {
            expectedByteCount(info)
            throw AssertionError("Expected DenoiserBindingException")
        } catch (expected: DenoiserBindingException) {
            assertEquals(OrtRuntimeFailure.UNSUPPORTED_TENSOR_TYPE, expected.failure)
        }
    }

    @Test
    fun binds_unique_4d_float_latent_even_without_latent_in_the_name() {
        val inputs = listOf(
            OrtJavaNamedTensorInfo("sample", OrtJavaTensorInfo(OrtJavaElementType.FLOAT, longArrayOf(1, 4, 128, 128))),
            OrtJavaNamedTensorInfo("timestep", OrtJavaTensorInfo(OrtJavaElementType.INT64, longArrayOf(1))),
            OrtJavaNamedTensorInfo("positive_cond", OrtJavaTensorInfo(OrtJavaElementType.FLOAT, longArrayOf(1, 256, 4096))),
            OrtJavaNamedTensorInfo("negative_uncond", OrtJavaTensorInfo(OrtJavaElementType.FLOAT, longArrayOf(1, 256, 4096))),
        )

        val binding = bindDenoiserInputs(inputs)

        assertEquals("sample", binding.latent.name)
        assertEquals("timestep", binding.timestep.name)
        assertEquals("positive_cond", binding.cond.name)
        assertEquals("negative_uncond", binding.uncond.name)
    }

    @Test
    fun rejects_opaque_conditioning_pair() {
        val inputs = listOf(
            OrtJavaNamedTensorInfo("sample", OrtJavaTensorInfo(OrtJavaElementType.FLOAT, longArrayOf(1, 4, 128, 128))),
            OrtJavaNamedTensorInfo("timestep", OrtJavaTensorInfo(OrtJavaElementType.INT64, longArrayOf(1))),
            OrtJavaNamedTensorInfo("context_a", OrtJavaTensorInfo(OrtJavaElementType.FLOAT, longArrayOf(1, 256, 4096))),
            OrtJavaNamedTensorInfo("context_b", OrtJavaTensorInfo(OrtJavaElementType.FLOAT, longArrayOf(1, 256, 4096))),
        )

        try {
            bindDenoiserInputs(inputs)
            throw AssertionError("Expected DenoiserBindingException")
        } catch (expected: DenoiserBindingException) {
            assertEquals(OrtRuntimeFailure.INPUT_MAPPING_FAILED, expected.failure)
        }
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
/tmp/run_android_runtime_tests.sh com.example.animanpu.runtime.DenoiserInputBindingTest
```

Expected: FAIL because the current implementation requires `latent` in the latent name and does not accept `positive_cond` / `negative_uncond` as an explicit pair.

- [ ] **Step 3: Write the minimal implementation**

```kotlin
// android/app/src/main/java/com/example/animanpu/runtime/DenoiserInputBinding.kt
package com.example.animanpu.runtime

class DenoiserBindingException(
    val failure: OrtRuntimeFailure,
) : RuntimeException(failure.code)

data class DenoiserInputBinding(
    val latent: OrtJavaNamedTensorInfo,
    val timestep: OrtJavaNamedTensorInfo,
    val cond: OrtJavaNamedTensorInfo,
    val uncond: OrtJavaNamedTensorInfo,
)

private enum class ConditioningRole {
    COND,
    UNCOND,
}

private fun isStaticShape(info: OrtJavaTensorInfo): Boolean = info.shape.all { it > 0L }

private fun elementCount(shape: LongArray): Long = shape.fold(1L) { acc, dim -> acc * dim }

private fun conditioningRole(name: String): ConditioningRole? {
    val lower = name.lowercase()
    return when {
        lower.contains("uncond") || lower.contains("negative") -> ConditioningRole.UNCOND
        (lower.contains("cond") && !lower.contains("uncond")) || lower.contains("positive") -> ConditioningRole.COND
        else -> null
    }
}

fun expectedByteCount(info: OrtJavaTensorInfo): Long {
    val elementWidth = when (info.elementType) {
        OrtJavaElementType.FLOAT -> 4L
        OrtJavaElementType.INT64 -> 8L
        else -> throw DenoiserBindingException(OrtRuntimeFailure.UNSUPPORTED_TENSOR_TYPE)
    }
    if (!isStaticShape(info)) {
        throw DenoiserBindingException(OrtRuntimeFailure.DYNAMIC_SHAPE_UNSUPPORTED)
    }
    return elementCount(info.shape) * elementWidth
}

fun bindDenoiserInputs(inputs: List<OrtJavaNamedTensorInfo>): DenoiserInputBinding {
    inputs.forEach { input -> expectedByteCount(input.info) }

    val timestep = inputs.singleOrNull { input ->
        input.name.contains("time", ignoreCase = true) &&
            elementCount(input.info.shape) == 1L &&
            input.info.elementType in setOf(OrtJavaElementType.FLOAT, OrtJavaElementType.INT64)
    } ?: throw DenoiserBindingException(OrtRuntimeFailure.INPUT_MAPPING_FAILED)

    val latentCandidates = inputs.filter { input ->
        input != timestep &&
            input.info.elementType == OrtJavaElementType.FLOAT &&
            input.info.shape.size == 4 &&
            isStaticShape(input.info)
    }
    val latent = latentCandidates.singleOrNull()
        ?: throw DenoiserBindingException(OrtRuntimeFailure.INPUT_MAPPING_FAILED)

    val remaining = inputs.filter { input -> input != timestep && input != latent }
    if (remaining.size != 2) {
        throw DenoiserBindingException(OrtRuntimeFailure.INPUT_MAPPING_FAILED)
    }

    val cond = remaining.singleOrNull { input ->
        conditioningRole(input.name) == ConditioningRole.COND && input.info.elementType == OrtJavaElementType.FLOAT
    } ?: throw DenoiserBindingException(OrtRuntimeFailure.INPUT_MAPPING_FAILED)

    val uncond = remaining.singleOrNull { input ->
        conditioningRole(input.name) == ConditioningRole.UNCOND && input.info.elementType == OrtJavaElementType.FLOAT
    } ?: throw DenoiserBindingException(OrtRuntimeFailure.INPUT_MAPPING_FAILED)

    if (!cond.info.shape.contentEquals(uncond.info.shape)) {
        throw DenoiserBindingException(OrtRuntimeFailure.INPUT_MAPPING_FAILED)
    }

    return DenoiserInputBinding(
        latent = latent,
        timestep = timestep,
        cond = cond,
        uncond = uncond,
    )
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run:

```bash
/tmp/run_android_runtime_tests.sh com.example.animanpu.runtime.DenoiserInputBindingTest
```

Expected: PASS with `OK (5 tests)`.

- [ ] **Step 5: Commit**

```bash
git add android/app/src/main/java/com/example/animanpu/runtime/DenoiserInputBinding.kt android/app/src/test/java/com/example/animanpu/runtime/DenoiserInputBindingTest.kt
git commit -m "test: bind android denoiser inputs from metadata"
```

### Task 2: Make `timestep` metadata-first and reject ambiguous scalar/latent candidates

**Files:**
- Modify: `android/app/src/main/java/com/example/animanpu/runtime/DenoiserInputBinding.kt`
- Modify: `android/app/src/test/java/com/example/animanpu/runtime/DenoiserInputBindingTest.kt`
- Create: `android/app/src/test/java/com/example/animanpu/runtime/StrictDenoiserBindingRuntimeTest.kt`

- [ ] **Step 1: Write the failing unit and runtime tests**

```kotlin
// android/app/src/test/java/com/example/animanpu/runtime/DenoiserInputBindingTest.kt
package com.example.animanpu.runtime

import org.junit.Assert.assertEquals
import org.junit.Test

class DenoiserInputBindingTest {
    @Test
    fun calculates_expected_float32_byte_size_for_static_shape() {
        val info = OrtJavaTensorInfo(
            elementType = OrtJavaElementType.FLOAT,
            shape = longArrayOf(1, 4, 128, 128),
        )

        val bytes = expectedByteCount(info)

        assertEquals(1L * 4L * 128L * 128L * 4L, bytes)
    }

    @Test
    fun rejects_dynamic_shapes() {
        val info = OrtJavaTensorInfo(
            elementType = OrtJavaElementType.FLOAT,
            shape = longArrayOf(1, -1, 128, 128),
        )

        try {
            expectedByteCount(info)
            throw AssertionError("Expected DenoiserBindingException")
        } catch (expected: DenoiserBindingException) {
            assertEquals(OrtRuntimeFailure.DYNAMIC_SHAPE_UNSUPPORTED, expected.failure)
        }
    }

    @Test
    fun rejects_unsupported_tensor_type() {
        val info = OrtJavaTensorInfo(
            elementType = OrtJavaElementType.UNKNOWN,
            shape = longArrayOf(1),
        )

        try {
            expectedByteCount(info)
            throw AssertionError("Expected DenoiserBindingException")
        } catch (expected: DenoiserBindingException) {
            assertEquals(OrtRuntimeFailure.UNSUPPORTED_TENSOR_TYPE, expected.failure)
        }
    }

    @Test
    fun binds_unique_scalar_like_timestep_without_time_in_the_name() {
        val inputs = listOf(
            OrtJavaNamedTensorInfo("sample", OrtJavaTensorInfo(OrtJavaElementType.FLOAT, longArrayOf(1, 4, 128, 128))),
            OrtJavaNamedTensorInfo("sigma", OrtJavaTensorInfo(OrtJavaElementType.INT64, longArrayOf(1))),
            OrtJavaNamedTensorInfo("positive_cond", OrtJavaTensorInfo(OrtJavaElementType.FLOAT, longArrayOf(1, 256, 4096))),
            OrtJavaNamedTensorInfo("negative_uncond", OrtJavaTensorInfo(OrtJavaElementType.FLOAT, longArrayOf(1, 256, 4096))),
        )

        val binding = bindDenoiserInputs(inputs)

        assertEquals("sigma", binding.timestep.name)
        assertEquals("sample", binding.latent.name)
    }

    @Test
    fun rejects_multiple_scalar_like_candidates() {
        val inputs = listOf(
            OrtJavaNamedTensorInfo("sample", OrtJavaTensorInfo(OrtJavaElementType.FLOAT, longArrayOf(1, 4, 128, 128))),
            OrtJavaNamedTensorInfo("timestep", OrtJavaTensorInfo(OrtJavaElementType.INT64, longArrayOf(1))),
            OrtJavaNamedTensorInfo("sigma", OrtJavaTensorInfo(OrtJavaElementType.FLOAT, longArrayOf(1))),
            OrtJavaNamedTensorInfo("positive_cond", OrtJavaTensorInfo(OrtJavaElementType.FLOAT, longArrayOf(1, 256, 4096))),
            OrtJavaNamedTensorInfo("negative_uncond", OrtJavaTensorInfo(OrtJavaElementType.FLOAT, longArrayOf(1, 256, 4096))),
        )

        try {
            bindDenoiserInputs(inputs)
            throw AssertionError("Expected DenoiserBindingException")
        } catch (expected: DenoiserBindingException) {
            assertEquals(OrtRuntimeFailure.INPUT_MAPPING_FAILED, expected.failure)
        }
    }

    @Test
    fun rejects_multiple_4d_float_candidates() {
        val inputs = listOf(
            OrtJavaNamedTensorInfo("sample", OrtJavaTensorInfo(OrtJavaElementType.FLOAT, longArrayOf(1, 4, 128, 128))),
            OrtJavaNamedTensorInfo("residual", OrtJavaTensorInfo(OrtJavaElementType.FLOAT, longArrayOf(1, 4, 128, 128))),
            OrtJavaNamedTensorInfo("sigma", OrtJavaTensorInfo(OrtJavaElementType.INT64, longArrayOf(1))),
            OrtJavaNamedTensorInfo("positive_cond", OrtJavaTensorInfo(OrtJavaElementType.FLOAT, longArrayOf(1, 256, 4096))),
            OrtJavaNamedTensorInfo("negative_uncond", OrtJavaTensorInfo(OrtJavaElementType.FLOAT, longArrayOf(1, 256, 4096))),
        )

        try {
            bindDenoiserInputs(inputs)
            throw AssertionError("Expected DenoiserBindingException")
        } catch (expected: DenoiserBindingException) {
            assertEquals(OrtRuntimeFailure.INPUT_MAPPING_FAILED, expected.failure)
        }
    }

    @Test
    fun rejects_opaque_conditioning_pair() {
        val inputs = listOf(
            OrtJavaNamedTensorInfo("sample", OrtJavaTensorInfo(OrtJavaElementType.FLOAT, longArrayOf(1, 4, 128, 128))),
            OrtJavaNamedTensorInfo("sigma", OrtJavaTensorInfo(OrtJavaElementType.INT64, longArrayOf(1))),
            OrtJavaNamedTensorInfo("context_a", OrtJavaTensorInfo(OrtJavaElementType.FLOAT, longArrayOf(1, 256, 4096))),
            OrtJavaNamedTensorInfo("context_b", OrtJavaTensorInfo(OrtJavaElementType.FLOAT, longArrayOf(1, 256, 4096))),
        )

        try {
            bindDenoiserInputs(inputs)
            throw AssertionError("Expected DenoiserBindingException")
        } catch (expected: DenoiserBindingException) {
            assertEquals(OrtRuntimeFailure.INPUT_MAPPING_FAILED, expected.failure)
        }
    }
}
```

```kotlin
// android/app/src/test/java/com/example/animanpu/runtime/StrictDenoiserBindingRuntimeTest.kt
package com.example.animanpu.runtime

import java.io.File
import kotlin.io.path.createTempDirectory
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

private class StrictFakeOrtJavaTensorValue : OrtJavaTensorValue {
    override fun close() = Unit
}

private class StrictFakeOrtJavaValue(
    private val bytes: ByteArray,
) : OrtJavaValue {
    override fun readRawBytes(): ByteArray = bytes
    override fun close() = Unit
}

private class StrictFakeExecuteSession(
    private val inputs: List<OrtJavaNamedTensorInfo>,
    private val outputs: Map<String, ByteArray>,
) : OrtJavaSession {
    override fun inputInfos(): List<OrtJavaNamedTensorInfo> = inputs

    override fun run(inputs: Map<String, OrtJavaTensorValue>): Map<String, OrtJavaValue> =
        outputs.mapValues { StrictFakeOrtJavaValue(it.value) }

    override fun close() = Unit
}

private class StrictFakeBridge(
    private val session: OrtJavaSession,
) : OrtJavaBridge {
    override fun availableProviders(): Set<String> = setOf("QNNExecutionProvider")

    override fun createSession(modelPath: File, config: OrtQnnConfig): OrtJavaSession = session

    override fun tensorFromRaw(info: OrtJavaTensorInfo, rawBytes: ByteArray): OrtJavaTensorValue = StrictFakeOrtJavaTensorValue()
}

class StrictDenoiserBindingRuntimeTest {
    @Test
    fun execute_accepts_metadata_first_latent_and_explicit_conditioning_names() = runTest {
        val root = createTempDirectory(prefix = "anima-engine-").toFile()
        val artifacts = ArtifactManager(root)
        artifacts.ensureRuntimeDirectories()
        artifacts.denoiserContextOnnx().writeText("ctx")
        artifacts.denoiserContextBin().writeText("bin")
        artifacts.latentInput().writeBytes(ByteArray(1 * 4 * 128 * 128 * 4))
        artifacts.timestepInput().writeBytes(ByteArray(8))
        artifacts.condInput().writeBytes(ByteArray(1 * 256 * 4096 * 4))
        artifacts.uncondInput().writeBytes(ByteArray(1 * 256 * 4096 * 4))

        val session = StrictFakeExecuteSession(
            inputs = listOf(
                OrtJavaNamedTensorInfo("sample", OrtJavaTensorInfo(OrtJavaElementType.FLOAT, longArrayOf(1, 4, 128, 128))),
                OrtJavaNamedTensorInfo("sigma", OrtJavaTensorInfo(OrtJavaElementType.INT64, longArrayOf(1))),
                OrtJavaNamedTensorInfo("positive_cond", OrtJavaTensorInfo(OrtJavaElementType.FLOAT, longArrayOf(1, 256, 4096))),
                OrtJavaNamedTensorInfo("negative_uncond", OrtJavaTensorInfo(OrtJavaElementType.FLOAT, longArrayOf(1, 256, 4096))),
            ),
            outputs = mapOf("output" to byteArrayOf(7, 6, 5, 4)),
        )

        val engine = OrtQnnDenoiserEngine(
            artifactManager = artifacts,
            sessionFactory = ReflectionOrtSessionFactory(StrictFakeBridge(session)),
            backendPath = "/data/local/tmp/libQnnHtp.so",
        )

        val result = engine.generate(
            GenerationRequest(1024, 1024, "cat astronaut", "blurry", 8, 5.0f),
        )

        assertTrue(result.sessionCreated)
        assertTrue(result.qnnActive)
        assertNull(result.failureReason)
        assertTrue(artifacts.outputTensor().readBytes().contentEquals(byteArrayOf(7, 6, 5, 4)))
    }

    @Test
    fun execute_rejects_multiple_scalar_like_candidates() = runTest {
        val root = createTempDirectory(prefix = "anima-engine-").toFile()
        val artifacts = ArtifactManager(root)
        artifacts.ensureRuntimeDirectories()
        artifacts.denoiserContextOnnx().writeText("ctx")
        artifacts.denoiserContextBin().writeText("bin")
        artifacts.latentInput().writeBytes(ByteArray(1 * 4 * 128 * 128 * 4))
        artifacts.timestepInput().writeBytes(ByteArray(8))
        artifacts.condInput().writeBytes(ByteArray(1 * 256 * 4096 * 4))
        artifacts.uncondInput().writeBytes(ByteArray(1 * 256 * 4096 * 4))

        val session = StrictFakeExecuteSession(
            inputs = listOf(
                OrtJavaNamedTensorInfo("sample", OrtJavaTensorInfo(OrtJavaElementType.FLOAT, longArrayOf(1, 4, 128, 128))),
                OrtJavaNamedTensorInfo("timestep", OrtJavaTensorInfo(OrtJavaElementType.INT64, longArrayOf(1))),
                OrtJavaNamedTensorInfo("sigma", OrtJavaTensorInfo(OrtJavaElementType.FLOAT, longArrayOf(1))),
                OrtJavaNamedTensorInfo("positive_cond", OrtJavaTensorInfo(OrtJavaElementType.FLOAT, longArrayOf(1, 256, 4096))),
                OrtJavaNamedTensorInfo("negative_uncond", OrtJavaTensorInfo(OrtJavaElementType.FLOAT, longArrayOf(1, 256, 4096))),
            ),
            outputs = mapOf("output" to byteArrayOf(1)),
        )

        val engine = OrtQnnDenoiserEngine(
            artifactManager = artifacts,
            sessionFactory = ReflectionOrtSessionFactory(StrictFakeBridge(session)),
            backendPath = "/data/local/tmp/libQnnHtp.so",
        )

        val result = engine.generate(
            GenerationRequest(1024, 1024, "cat astronaut", "blurry", 8, 5.0f),
        )

        assertTrue(result.sessionCreated)
        assertFalse(result.qnnActive)
        assertEquals(OrtRuntimeFailure.INPUT_MAPPING_FAILED, result.failureReason)
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
/tmp/run_android_runtime_tests.sh com.example.animanpu.runtime.DenoiserInputBindingTest com.example.animanpu.runtime.StrictDenoiserBindingRuntimeTest
```

Expected: FAIL because the current implementation still requires `timestep` to contain `time` in the name and does not reject ambiguity by metadata-first rules.

- [ ] **Step 3: Write the minimal implementation**

```kotlin
// android/app/src/main/java/com/example/animanpu/runtime/DenoiserInputBinding.kt
package com.example.animanpu.runtime

class DenoiserBindingException(
    val failure: OrtRuntimeFailure,
) : RuntimeException(failure.code)

data class DenoiserInputBinding(
    val latent: OrtJavaNamedTensorInfo,
    val timestep: OrtJavaNamedTensorInfo,
    val cond: OrtJavaNamedTensorInfo,
    val uncond: OrtJavaNamedTensorInfo,
)

private enum class ConditioningRole {
    COND,
    UNCOND,
}

private fun isStaticShape(info: OrtJavaTensorInfo): Boolean = info.shape.all { it > 0L }

private fun elementCount(shape: LongArray): Long = shape.fold(1L) { acc, dim -> acc * dim }

private fun isScalarLikeCandidate(info: OrtJavaTensorInfo): Boolean =
    isStaticShape(info) &&
        elementCount(info.shape) == 1L &&
        info.elementType in setOf(OrtJavaElementType.FLOAT, OrtJavaElementType.INT64)

private fun isLatentCandidate(info: OrtJavaTensorInfo): Boolean =
    isStaticShape(info) &&
        info.elementType == OrtJavaElementType.FLOAT &&
        info.shape.size == 4

private fun conditioningRole(name: String): ConditioningRole? {
    val lower = name.lowercase()
    return when {
        lower.contains("uncond") || lower.contains("negative") -> ConditioningRole.UNCOND
        (lower.contains("cond") && !lower.contains("uncond")) || lower.contains("positive") -> ConditioningRole.COND
        else -> null
    }
}

fun expectedByteCount(info: OrtJavaTensorInfo): Long {
    val elementWidth = when (info.elementType) {
        OrtJavaElementType.FLOAT -> 4L
        OrtJavaElementType.INT64 -> 8L
        else -> throw DenoiserBindingException(OrtRuntimeFailure.UNSUPPORTED_TENSOR_TYPE)
    }
    if (!isStaticShape(info)) {
        throw DenoiserBindingException(OrtRuntimeFailure.DYNAMIC_SHAPE_UNSUPPORTED)
    }
    return elementCount(info.shape) * elementWidth
}

fun bindDenoiserInputs(inputs: List<OrtJavaNamedTensorInfo>): DenoiserInputBinding {
    inputs.forEach { input -> expectedByteCount(input.info) }

    val timestepCandidates = inputs.filter { input -> isScalarLikeCandidate(input.info) }
    val timestep = timestepCandidates.singleOrNull()
        ?: throw DenoiserBindingException(OrtRuntimeFailure.INPUT_MAPPING_FAILED)

    val latentCandidates = inputs.filter { input -> input != timestep && isLatentCandidate(input.info) }
    val latent = latentCandidates.singleOrNull()
        ?: throw DenoiserBindingException(OrtRuntimeFailure.INPUT_MAPPING_FAILED)

    val remaining = inputs.filter { input -> input != timestep && input != latent }
    if (remaining.size != 2) {
        throw DenoiserBindingException(OrtRuntimeFailure.INPUT_MAPPING_FAILED)
    }

    val first = remaining[0]
    val second = remaining[1]
    if (first.info.elementType != second.info.elementType || !first.info.shape.contentEquals(second.info.shape)) {
        throw DenoiserBindingException(OrtRuntimeFailure.INPUT_MAPPING_FAILED)
    }

    val cond = remaining.singleOrNull { input -> conditioningRole(input.name) == ConditioningRole.COND }
        ?: throw DenoiserBindingException(OrtRuntimeFailure.INPUT_MAPPING_FAILED)
    val uncond = remaining.singleOrNull { input -> conditioningRole(input.name) == ConditioningRole.UNCOND }
        ?: throw DenoiserBindingException(OrtRuntimeFailure.INPUT_MAPPING_FAILED)

    return DenoiserInputBinding(
        latent = latent,
        timestep = timestep,
        cond = cond,
        uncond = uncond,
    )
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
/tmp/run_android_runtime_tests.sh com.example.animanpu.runtime.DenoiserInputBindingTest com.example.animanpu.runtime.StrictDenoiserBindingRuntimeTest
```

Expected: PASS with `OK (8 tests)`.

- [ ] **Step 5: Commit**

```bash
git add android/app/src/main/java/com/example/animanpu/runtime/DenoiserInputBinding.kt android/app/src/test/java/com/example/animanpu/runtime/DenoiserInputBindingTest.kt android/app/src/test/java/com/example/animanpu/runtime/StrictDenoiserBindingRuntimeTest.kt
git commit -m "feat: tighten android denoiser input binding"
```

### Task 3: Run the full Android runtime suite with the stricter binding

**Files:**
- Test: `android/app/src/test/java/com/example/animanpu/runtime/GenerationOrchestratorTest.kt`
- Test: `android/app/src/test/java/com/example/animanpu/runtime/OrtRuntimeFailureTest.kt`
- Test: `android/app/src/test/java/com/example/animanpu/runtime/OrtSessionFactoryTest.kt`
- Test: `android/app/src/test/java/com/example/animanpu/runtime/DenoiserInputBindingTest.kt`
- Test: `android/app/src/test/java/com/example/animanpu/runtime/OrtQnnDenoiserEngineTest.kt`
- Test: `android/app/src/test/java/com/example/animanpu/runtime/AndroidValidationDocTest.kt`
- Test: `android/app/src/test/java/com/example/animanpu/runtime/StrictDenoiserBindingRuntimeTest.kt`

- [ ] **Step 1: Run the targeted runtime suite**

Run:

```bash
/tmp/run_android_runtime_tests.sh \
  com.example.animanpu.runtime.GenerationOrchestratorTest \
  com.example.animanpu.runtime.OrtRuntimeFailureTest \
  com.example.animanpu.runtime.OrtSessionFactoryTest \
  com.example.animanpu.runtime.DenoiserInputBindingTest \
  com.example.animanpu.runtime.OrtQnnDenoiserEngineTest \
  com.example.animanpu.runtime.AndroidValidationDocTest \
  com.example.animanpu.runtime.StrictDenoiserBindingRuntimeTest
```

Expected: PASS with all Android runtime tests green.

- [ ] **Step 2: Commit**

```bash
git commit --allow-empty -m "test: verify strict android denoiser binding suite"
```
