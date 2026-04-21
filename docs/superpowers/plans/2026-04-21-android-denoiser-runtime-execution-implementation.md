# Android Denoiser Runtime Execution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the Android denoiser session handle into a real one-shot execution path that reads session metadata, validates the four runtime `raw` inputs, executes once, and writes `denoiser_output.raw`.

**Architecture:** Extend the Android ORT bridge to expose session metadata and fakeable execute primitives, then implement a deterministic input-binding layer inside the session handle that maps metadata onto `latent`, `timestep`, `cond`, and `uncond`. Keep failure handling explicit through `OrtRuntimeFailure` and prove behavior with JVM tests using fake sessions and fake output tensors.

**Tech Stack:** Kotlin, Android app module, Android ORT Java API via reflection, JVM unit tests via Gradle/JUnit4

---

### Task 1: Extend runtime failures for execute-stage validation

**Files:**
- Modify: `android/app/src/main/java/com/example/animanpu/runtime/OrtRuntimeFailure.kt`
- Create: `android/app/src/test/java/com/example/animanpu/runtime/OrtRuntimeFailureTest.kt`

- [ ] **Step 1: Write the failing test for the new execute-stage failure codes**

```kotlin
package com.example.animanpu.runtime

import org.junit.Assert.assertEquals
import org.junit.Test

class OrtRuntimeFailureTest {
    @Test
    fun exposes_execute_stage_failure_codes() {
        assertEquals("input_metadata_unavailable", OrtRuntimeFailure.INPUT_METADATA_UNAVAILABLE.code)
        assertEquals("dynamic_shape_unsupported", OrtRuntimeFailure.DYNAMIC_SHAPE_UNSUPPORTED.code)
        assertEquals("unsupported_tensor_type", OrtRuntimeFailure.UNSUPPORTED_TENSOR_TYPE.code)
        assertEquals("input_mapping_failed", OrtRuntimeFailure.INPUT_MAPPING_FAILED.code)
        assertEquals("input_file_size_mismatch", OrtRuntimeFailure.INPUT_FILE_SIZE_MISMATCH.code)
        assertEquals("session_execute_failed", OrtRuntimeFailure.SESSION_EXECUTE_FAILED.code)
        assertEquals("output_write_failed", OrtRuntimeFailure.OUTPUT_WRITE_FAILED.code)
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cd android && /opt/gradle-8.10.2/bin/gradle testDebugUnitTest --tests com.example.animanpu.runtime.OrtRuntimeFailureTest --console=plain
```

Expected: FAIL with unresolved enum constants such as `INPUT_METADATA_UNAVAILABLE`.

- [ ] **Step 3: Write the minimal implementation**

```kotlin
// android/app/src/main/java/com/example/animanpu/runtime/OrtRuntimeFailure.kt
package com.example.animanpu.runtime

enum class OrtRuntimeFailure(
    val code: String,
) {
    MISSING_RUNTIME_ARTIFACT("missing_runtime_artifact"),
    ORT_API_UNAVAILABLE("ort_api_unavailable"),
    QNN_PROVIDER_UNAVAILABLE("qnn_provider_unavailable"),
    SESSION_CREATE_FAILED("session_create_failed"),
    EXECUTE_NOT_IMPLEMENTED("execute_not_implemented"),
    EXECUTE_FAILED("execute_failed"),
    INPUT_METADATA_UNAVAILABLE("input_metadata_unavailable"),
    DYNAMIC_SHAPE_UNSUPPORTED("dynamic_shape_unsupported"),
    UNSUPPORTED_TENSOR_TYPE("unsupported_tensor_type"),
    INPUT_MAPPING_FAILED("input_mapping_failed"),
    INPUT_FILE_SIZE_MISMATCH("input_file_size_mismatch"),
    SESSION_EXECUTE_FAILED("session_execute_failed"),
    OUTPUT_WRITE_FAILED("output_write_failed"),
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run:

```bash
cd android && /opt/gradle-8.10.2/bin/gradle testDebugUnitTest --tests com.example.animanpu.runtime.OrtRuntimeFailureTest --console=plain
```

Expected: PASS with `1 test completed, 0 failed`.

- [ ] **Step 5: Commit**

```bash
git add android/app/src/main/java/com/example/animanpu/runtime/OrtRuntimeFailure.kt android/app/src/test/java/com/example/animanpu/runtime/OrtRuntimeFailureTest.kt
git commit -m "test: add android execute-stage runtime failures"
```

### Task 2: Add fakeable session metadata and tensor primitives to the ORT bridge

**Files:**
- Modify: `android/app/src/main/java/com/example/animanpu/runtime/OrtJavaBridge.kt`
- Test: `android/app/src/test/java/com/example/animanpu/runtime/OrtSessionFactoryTest.kt`

- [ ] **Step 1: Write the failing metadata and output tests**

```kotlin
package com.example.animanpu.runtime

import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

private class FakeOrtJavaValue(
    private val bytes: ByteArray,
) : OrtJavaValue {
    override fun readRawBytes(): ByteArray = bytes
    override fun close() = Unit
}

private class FakeOrtJavaSessionWithMetadata(
    private val inputs: List<OrtJavaTensorInfo>,
    private val outputNames: List<String>,
) : OrtJavaSession {
    override fun inputInfos(): List<OrtJavaNamedTensorInfo> =
        inputs.mapIndexed { index, info ->
            OrtJavaNamedTensorInfo(name = listOf("latent", "timestep", "cond", "uncond")[index], info = info)
        }

    override fun run(inputs: Map<String, OrtJavaTensorValue>): Map<String, OrtJavaValue> =
        mapOf(outputNames.first() to FakeOrtJavaValue(byteArrayOf(1, 2, 3, 4)))

    override fun close() = Unit
}

class OrtSessionFactoryTest {
    @Test
    fun session_exposes_named_input_metadata() {
        val session = FakeOrtJavaSessionWithMetadata(
            inputs = listOf(
                OrtJavaTensorInfo(elementType = OrtJavaElementType.FLOAT, shape = longArrayOf(1, 4, 128, 128)),
                OrtJavaTensorInfo(elementType = OrtJavaElementType.INT64, shape = longArrayOf(1)),
                OrtJavaTensorInfo(elementType = OrtJavaElementType.FLOAT, shape = longArrayOf(1, 256, 4096)),
                OrtJavaTensorInfo(elementType = OrtJavaElementType.FLOAT, shape = longArrayOf(1, 256, 4096)),
            ),
            outputNames = listOf("output"),
        )

        val infos = session.inputInfos()

        assertEquals(4, infos.size)
        assertEquals("latent", infos[0].name)
        assertEquals(OrtJavaElementType.FLOAT, infos[0].info.elementType)
        assertTrue(infos[0].info.shape.contentEquals(longArrayOf(1, 4, 128, 128)))
    }

    @Test
    fun session_run_returns_named_outputs() {
        val session = FakeOrtJavaSessionWithMetadata(
            inputs = listOf(
                OrtJavaTensorInfo(elementType = OrtJavaElementType.FLOAT, shape = longArrayOf(1, 4, 128, 128)),
                OrtJavaTensorInfo(elementType = OrtJavaElementType.INT64, shape = longArrayOf(1)),
                OrtJavaTensorInfo(elementType = OrtJavaElementType.FLOAT, shape = longArrayOf(1, 256, 4096)),
                OrtJavaTensorInfo(elementType = OrtJavaElementType.FLOAT, shape = longArrayOf(1, 256, 4096)),
            ),
            outputNames = listOf("output"),
        )

        val outputs = session.run(emptyMap())

        assertEquals(listOf("output"), outputs.keys.toList())
        assertTrue(outputs.getValue("output").readRawBytes().contentEquals(byteArrayOf(1, 2, 3, 4)))
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cd android && /opt/gradle-8.10.2/bin/gradle testDebugUnitTest --tests com.example.animanpu.runtime.OrtSessionFactoryTest --console=plain
```

Expected: FAIL with unresolved references for `OrtJavaTensorInfo`, `OrtJavaNamedTensorInfo`, `OrtJavaElementType`, `OrtJavaTensorValue`, `OrtJavaValue`, `inputInfos`, or `run`.

- [ ] **Step 3: Write the minimal implementation**

```kotlin
// android/app/src/main/java/com/example/animanpu/runtime/OrtJavaBridge.kt
package com.example.animanpu.runtime

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

enum class OrtJavaElementType {
    FLOAT,
    INT64,
    UNKNOWN,
}

data class OrtJavaTensorInfo(
    val elementType: OrtJavaElementType,
    val shape: LongArray,
)

data class OrtJavaNamedTensorInfo(
    val name: String,
    val info: OrtJavaTensorInfo,
)

interface OrtJavaTensorValue : AutoCloseable {
    override fun close()
}

interface OrtJavaValue : AutoCloseable {
    fun readRawBytes(): ByteArray
    override fun close()
}

interface OrtJavaSession : AutoCloseable {
    fun inputInfos(): List<OrtJavaNamedTensorInfo>
    fun run(inputs: Map<String, OrtJavaTensorValue>): Map<String, OrtJavaValue>
    override fun close()
}

interface OrtJavaBridge {
    fun availableProviders(): Set<String>
    fun createSession(modelPath: File, config: OrtQnnConfig): OrtJavaSession
    fun tensorFromRaw(info: OrtJavaTensorInfo, rawBytes: ByteArray): OrtJavaTensorValue
}
```

```kotlin
// android/app/src/main/java/com/example/animanpu/runtime/OrtJavaBridge.kt
package com.example.animanpu.runtime

import java.io.File
import java.nio.Buffer
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.LongBuffer

private class ReflectionOrtJavaTensorValue(
    val inner: Any,
) : OrtJavaTensorValue {
    override fun close() {
        runCatching {
            inner.javaClass.getMethod("close").invoke(inner)
        }
    }
}

private class ReflectionOrtJavaValue(
    private val inner: Any,
) : OrtJavaValue {
    override fun readRawBytes(): ByteArray {
        val optional = inner.javaClass.getMethod("getBufferRef").invoke(inner)
        val buffer = optional.javaClass.getMethod("get").invoke(optional) as Buffer
        return when (buffer) {
            is ByteBuffer -> {
                val dup = buffer.duplicate().order(ByteOrder.nativeOrder())
                val out = ByteArray(dup.remaining())
                dup.get(out)
                out
            }
            is FloatBuffer -> {
                val dup = buffer.duplicate()
                val out = ByteBuffer.allocate(dup.remaining() * 4).order(ByteOrder.nativeOrder())
                while (dup.hasRemaining()) {
                    out.putFloat(dup.get())
                }
                out.array()
            }
            is LongBuffer -> {
                val dup = buffer.duplicate()
                val out = ByteBuffer.allocate(dup.remaining() * 8).order(ByteOrder.nativeOrder())
                while (dup.hasRemaining()) {
                    out.putLong(dup.get())
                }
                out.array()
            }
            else -> throw IllegalStateException("Unsupported output buffer: ${buffer.javaClass.name}")
        }
    }

    override fun close() {
        runCatching {
            inner.javaClass.getMethod("close").invoke(inner)
        }
    }
}

private fun toOrtJavaElementType(valueInfo: Any): OrtJavaElementType {
    val typeName = valueInfo.javaClass.getField("type").get(valueInfo).toString()
    return when (typeName) {
        "FLOAT" -> OrtJavaElementType.FLOAT
        "INT64" -> OrtJavaElementType.INT64
        else -> OrtJavaElementType.UNKNOWN
    }
}

private fun extractNamedTensorInfo(nodeInfo: Any): OrtJavaNamedTensorInfo? {
    val name = nodeInfo.javaClass.getMethod("getName").invoke(nodeInfo)?.toString() ?: return null
    val valueInfo = nodeInfo.javaClass.getMethod("getInfo").invoke(nodeInfo) ?: return null
    if (valueInfo.javaClass.simpleName != "TensorInfo") {
        return null
    }
    val shape = valueInfo.javaClass.getMethod("getShape").invoke(valueInfo) as LongArray
    return OrtJavaNamedTensorInfo(
        name = name,
        info = OrtJavaTensorInfo(
            elementType = toOrtJavaElementType(valueInfo),
            shape = shape,
        ),
    )
}

class ReflectionOrtJavaBridge : OrtJavaBridge {
    override fun availableProviders(): Set<String> {
        return try {
            val environmentClass = Class.forName("ai.onnxruntime.OrtEnvironment")
            val providers = environmentClass.getMethod("getAvailableProviders").invoke(null) as Iterable<*>
            normalizeProviderNames(providers)
        } catch (error: Throwable) {
            throw OrtJavaApiUnavailableException(error)
        }
    }

    override fun createSession(modelPath: File, config: OrtQnnConfig): OrtJavaSession {
        try {
            val environmentClass = Class.forName("ai.onnxruntime.OrtEnvironment")
            val sessionOptionsClass = Class.forName("ai.onnxruntime.OrtSession\$SessionOptions")
            val environment = environmentClass.getMethod("getEnvironment").invoke(null)
            val sessionOptions = sessionOptionsClass.getConstructor().newInstance()
            val qnnOptions = config.providerOptions().filterKeys { it != "session.disable_cpu_ep_fallback" }
            sessionOptionsClass.getMethod("addQnn", Map::class.java).invoke(sessionOptions, qnnOptions)
            sessionOptionsClass.getMethod("addConfigEntry", String::class.java, String::class.java).invoke(
                sessionOptions,
                "session.disable_cpu_ep_fallback",
                if (config.disableCpuFallback) "1" else "0",
            )
            sessionOptionsClass.getMethod("enableProfiling", String::class.java).invoke(sessionOptions, config.profilingPath)
            val session = environmentClass.getMethod("createSession", String::class.java, sessionOptionsClass)
                .invoke(environment, modelPath.path, sessionOptions)

            return object : OrtJavaSession {
                override fun inputInfos(): List<OrtJavaNamedTensorInfo> {
                    val inputMap = session.javaClass.getMethod("getInputInfo").invoke(session) as Map<*, *>
                    return inputMap.values.mapNotNull { node -> extractNamedTensorInfo(node!!) }
                }

                override fun run(inputs: Map<String, OrtJavaTensorValue>): Map<String, OrtJavaValue> {
                    val ortInputs = inputs.mapValues { (_, value) -> (value as ReflectionOrtJavaTensorValue).inner }
                    val result = session.javaClass.getMethod("run", Map::class.java).invoke(session, ortInputs)
                    val iterator = result.javaClass.getMethod("iterator").invoke(result) as Iterator<*>
                    val outputs = linkedMapOf<String, OrtJavaValue>()
                    while (iterator.hasNext()) {
                        val entry = iterator.next()
                        val key = entry.javaClass.getMethod("getKey").invoke(entry).toString()
                        val value = entry.javaClass.getMethod("getValue").invoke(entry)
                        outputs[key] = ReflectionOrtJavaValue(value)
                    }
                    return outputs
                }

                override fun close() {
                    runCatching { session.javaClass.getMethod("close").invoke(session) }
                    runCatching { sessionOptionsClass.getMethod("close").invoke(sessionOptions) }
                }
            }
        } catch (error: RuntimeException) {
            throw error
        } catch (error: Throwable) {
            throw OrtJavaSessionCreateException(error)
        }
    }

    override fun tensorFromRaw(info: OrtJavaTensorInfo, rawBytes: ByteArray): OrtJavaTensorValue {
        try {
            val environmentClass = Class.forName("ai.onnxruntime.OrtEnvironment")
            val onnxTensorClass = Class.forName("ai.onnxruntime.OnnxTensor")
            val environment = environmentClass.getMethod("getEnvironment").invoke(null)
            val nativeBuffer = ByteBuffer.wrap(rawBytes).order(ByteOrder.nativeOrder())
            val tensor = when (info.elementType) {
                OrtJavaElementType.FLOAT -> {
                    val floatBuffer = nativeBuffer.asFloatBuffer()
                    onnxTensorClass.getMethod(
                        "createTensor",
                        environmentClass,
                        FloatBuffer::class.java,
                        LongArray::class.java,
                    ).invoke(null, environment, floatBuffer, info.shape)
                }
                OrtJavaElementType.INT64 -> {
                    val longBuffer = nativeBuffer.asLongBuffer()
                    onnxTensorClass.getMethod(
                        "createTensor",
                        environmentClass,
                        LongBuffer::class.java,
                        LongArray::class.java,
                    ).invoke(null, environment, longBuffer, info.shape)
                }
                else -> throw IllegalArgumentException("Unsupported tensor type: ${info.elementType}")
            }
            return ReflectionOrtJavaTensorValue(tensor)
        } catch (error: RuntimeException) {
            throw error
        } catch (error: Throwable) {
            throw OrtJavaSessionCreateException(error)
        }
    }
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run:

```bash
cd android && /opt/gradle-8.10.2/bin/gradle testDebugUnitTest --tests com.example.animanpu.runtime.OrtSessionFactoryTest --console=plain
```

Expected: PASS with the metadata-oriented tests included in the suite.

- [ ] **Step 5: Commit**

```bash
git add android/app/src/main/java/com/example/animanpu/runtime/OrtJavaBridge.kt android/app/src/test/java/com/example/animanpu/runtime/OrtSessionFactoryTest.kt
git commit -m "test: add android ort session metadata seam"
```

### Task 3: Add deterministic input mapping and byte-size validation helpers

**Files:**
- Create: `android/app/src/main/java/com/example/animanpu/runtime/DenoiserInputBinding.kt`
- Test: `android/app/src/test/java/com/example/animanpu/runtime/DenoiserInputBindingTest.kt`

- [ ] **Step 1: Write the failing input-binding tests**

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
    fun maps_named_inputs_to_runtime_files() {
        val inputs = listOf(
            OrtJavaNamedTensorInfo("latent", OrtJavaTensorInfo(OrtJavaElementType.FLOAT, longArrayOf(1, 4, 128, 128))),
            OrtJavaNamedTensorInfo("timestep", OrtJavaTensorInfo(OrtJavaElementType.INT64, longArrayOf(1))),
            OrtJavaNamedTensorInfo("cond", OrtJavaTensorInfo(OrtJavaElementType.FLOAT, longArrayOf(1, 256, 4096))),
            OrtJavaNamedTensorInfo("uncond", OrtJavaTensorInfo(OrtJavaElementType.FLOAT, longArrayOf(1, 256, 4096))),
        )

        val binding = bindDenoiserInputs(inputs)

        assertEquals("latent", binding.latent.name)
        assertEquals("timestep", binding.timestep.name)
        assertEquals("cond", binding.cond.name)
        assertEquals("uncond", binding.uncond.name)
    }

    @Test
    fun rejects_ambiguous_conditioning_pair() {
        val inputs = listOf(
            OrtJavaNamedTensorInfo("latent", OrtJavaTensorInfo(OrtJavaElementType.FLOAT, longArrayOf(1, 4, 128, 128))),
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
cd android && /opt/gradle-8.10.2/bin/gradle testDebugUnitTest --tests com.example.animanpu.runtime.DenoiserInputBindingTest --console=plain
```

Expected: FAIL with unresolved references such as `expectedByteCount`, `bindDenoiserInputs`, and `DenoiserBindingException`.

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

fun expectedByteCount(info: OrtJavaTensorInfo): Long {
    val elementWidth = when (info.elementType) {
        OrtJavaElementType.FLOAT -> 4L
        OrtJavaElementType.INT64 -> 8L
        else -> throw DenoiserBindingException(OrtRuntimeFailure.UNSUPPORTED_TENSOR_TYPE)
    }
    if (info.shape.any { it <= 0L }) {
        throw DenoiserBindingException(OrtRuntimeFailure.DYNAMIC_SHAPE_UNSUPPORTED)
    }
    return info.shape.fold(1L) { acc, dim -> acc * dim } * elementWidth
}

fun bindDenoiserInputs(inputs: List<OrtJavaNamedTensorInfo>): DenoiserInputBinding {
    val latent = inputs.singleOrNull { it.name.contains("latent", ignoreCase = true) }
        ?: throw DenoiserBindingException(OrtRuntimeFailure.INPUT_MAPPING_FAILED)
    val timestep = inputs.singleOrNull { it.name.contains("time", ignoreCase = true) }
        ?: throw DenoiserBindingException(OrtRuntimeFailure.INPUT_MAPPING_FAILED)
    val cond = inputs.singleOrNull { it.name.equals("cond", ignoreCase = true) }
        ?: throw DenoiserBindingException(OrtRuntimeFailure.INPUT_MAPPING_FAILED)
    val uncond = inputs.singleOrNull { it.name.equals("uncond", ignoreCase = true) }
        ?: throw DenoiserBindingException(OrtRuntimeFailure.INPUT_MAPPING_FAILED)
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
cd android && /opt/gradle-8.10.2/bin/gradle testDebugUnitTest --tests com.example.animanpu.runtime.DenoiserInputBindingTest --console=plain
```

Expected: PASS with `5 tests completed, 0 failed`.

- [ ] **Step 5: Commit**

```bash
git add android/app/src/main/java/com/example/animanpu/runtime/DenoiserInputBinding.kt android/app/src/test/java/com/example/animanpu/runtime/DenoiserInputBindingTest.kt
git commit -m "test: add android denoiser input binding"
```

### Task 4: Execute one fake denoiser run and persist the output raw file

**Files:**
- Modify: `android/app/src/main/java/com/example/animanpu/runtime/OrtSessionFactory.kt`
- Modify: `android/app/src/main/java/com/example/animanpu/runtime/OrtJavaBridge.kt`
- Modify: `android/app/src/test/java/com/example/animanpu/runtime/OrtQnnDenoiserEngineTest.kt`
- Test: `android/app/src/test/java/com/example/animanpu/runtime/OrtQnnDenoiserEngineTest.kt`

- [ ] **Step 1: Write the failing execute-path tests**

```kotlin
package com.example.animanpu.runtime

import java.io.File
import kotlin.io.path.createTempDirectory
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

private class FakeOrtJavaTensorValue : OrtJavaTensorValue {
    override fun close() = Unit
}

private class FakeOrtJavaValue(
    private val bytes: ByteArray,
) : OrtJavaValue {
    override fun readRawBytes(): ByteArray = bytes
    override fun close() = Unit
}

private class FakeExecuteOrtJavaSession(
    private val inputs: List<OrtJavaNamedTensorInfo>,
    private val outputs: Map<String, ByteArray>,
    private val executeFails: Boolean = false,
) : OrtJavaSession {
    override fun inputInfos(): List<OrtJavaNamedTensorInfo> = inputs

    override fun run(inputs: Map<String, OrtJavaTensorValue>): Map<String, OrtJavaValue> {
        if (executeFails) {
            throw IllegalStateException("execute failed")
        }
        return outputs.mapValues { FakeOrtJavaValue(it.value) }
    }

    override fun close() = Unit
}

private class FakeExecuteOrtJavaBridge(
    private val session: OrtJavaSession,
) : OrtJavaBridge {
    override fun availableProviders(): Set<String> = setOf("QNNExecutionProvider")

    override fun createSession(modelPath: File, config: OrtQnnConfig): OrtJavaSession = session

    override fun tensorFromRaw(info: OrtJavaTensorInfo, rawBytes: ByteArray): OrtJavaTensorValue = FakeOrtJavaTensorValue()
}

class OrtQnnDenoiserEngineTest {
    @Test
    fun successful_execute_writes_output_raw_file() = runTest {
        val root = createTempDirectory(prefix = "anima-engine-").toFile()
        val artifacts = ArtifactManager(root)
        artifacts.ensureRuntimeDirectories()
        artifacts.denoiserContextOnnx().writeText("ctx")
        artifacts.denoiserContextBin().writeText("bin")
        artifacts.latentInput().writeBytes(ByteArray(1 * 4 * 128 * 128 * 4))
        artifacts.timestepInput().writeBytes(ByteArray(8))
        artifacts.condInput().writeBytes(ByteArray(1 * 256 * 4096 * 4))
        artifacts.uncondInput().writeBytes(ByteArray(1 * 256 * 4096 * 4))

        val session = FakeExecuteOrtJavaSession(
            inputs = listOf(
                OrtJavaNamedTensorInfo("latent", OrtJavaTensorInfo(OrtJavaElementType.FLOAT, longArrayOf(1, 4, 128, 128))),
                OrtJavaNamedTensorInfo("timestep", OrtJavaTensorInfo(OrtJavaElementType.INT64, longArrayOf(1))),
                OrtJavaNamedTensorInfo("cond", OrtJavaTensorInfo(OrtJavaElementType.FLOAT, longArrayOf(1, 256, 4096))),
                OrtJavaNamedTensorInfo("uncond", OrtJavaTensorInfo(OrtJavaElementType.FLOAT, longArrayOf(1, 256, 4096))),
            ),
            outputs = mapOf("output" to byteArrayOf(9, 8, 7, 6)),
        )

        val engine = OrtQnnDenoiserEngine(
            artifactManager = artifacts,
            sessionFactory = ReflectionOrtSessionFactory(FakeExecuteOrtJavaBridge(session)),
            backendPath = "/data/local/tmp/libQnnHtp.so",
        )

        val result = engine.generate(
            GenerationRequest(1024, 1024, "cat astronaut", "blurry", 8, 5.0f),
        )

        assertTrue(result.sessionCreated)
        assertTrue(result.qnnActive)
        assertNull(result.failureReason)
        assertTrue(artifacts.outputTensor().readBytes().contentEquals(byteArrayOf(9, 8, 7, 6)))
    }

    @Test
    fun rejects_input_file_size_mismatch_before_execute() = runTest {
        val root = createTempDirectory(prefix = "anima-engine-").toFile()
        val artifacts = ArtifactManager(root)
        artifacts.ensureRuntimeDirectories()
        artifacts.denoiserContextOnnx().writeText("ctx")
        artifacts.denoiserContextBin().writeText("bin")
        artifacts.latentInput().writeBytes(ByteArray(3))
        artifacts.timestepInput().writeBytes(ByteArray(8))
        artifacts.condInput().writeBytes(ByteArray(1 * 256 * 4096 * 4))
        artifacts.uncondInput().writeBytes(ByteArray(1 * 256 * 4096 * 4))

        val session = FakeExecuteOrtJavaSession(
            inputs = listOf(
                OrtJavaNamedTensorInfo("latent", OrtJavaTensorInfo(OrtJavaElementType.FLOAT, longArrayOf(1, 4, 128, 128))),
                OrtJavaNamedTensorInfo("timestep", OrtJavaTensorInfo(OrtJavaElementType.INT64, longArrayOf(1))),
                OrtJavaNamedTensorInfo("cond", OrtJavaTensorInfo(OrtJavaElementType.FLOAT, longArrayOf(1, 256, 4096))),
                OrtJavaNamedTensorInfo("uncond", OrtJavaTensorInfo(OrtJavaElementType.FLOAT, longArrayOf(1, 256, 4096))),
            ),
            outputs = mapOf("output" to byteArrayOf(1)),
        )

        val engine = OrtQnnDenoiserEngine(
            artifactManager = artifacts,
            sessionFactory = ReflectionOrtSessionFactory(FakeExecuteOrtJavaBridge(session)),
            backendPath = "/data/local/tmp/libQnnHtp.so",
        )

        val result = engine.generate(
            GenerationRequest(1024, 1024, "cat astronaut", "blurry", 8, 5.0f),
        )

        assertTrue(result.sessionCreated)
        assertFalse(result.qnnActive)
        assertEquals(OrtRuntimeFailure.INPUT_FILE_SIZE_MISMATCH, result.failureReason)
    }

    @Test
    fun rejects_ambiguous_multiple_outputs() = runTest {
        val root = createTempDirectory(prefix = "anima-engine-").toFile()
        val artifacts = ArtifactManager(root)
        artifacts.ensureRuntimeDirectories()
        artifacts.denoiserContextOnnx().writeText("ctx")
        artifacts.denoiserContextBin().writeText("bin")
        artifacts.latentInput().writeBytes(ByteArray(1 * 4 * 128 * 128 * 4))
        artifacts.timestepInput().writeBytes(ByteArray(8))
        artifacts.condInput().writeBytes(ByteArray(1 * 256 * 4096 * 4))
        artifacts.uncondInput().writeBytes(ByteArray(1 * 256 * 4096 * 4))

        val session = FakeExecuteOrtJavaSession(
            inputs = listOf(
                OrtJavaNamedTensorInfo("latent", OrtJavaTensorInfo(OrtJavaElementType.FLOAT, longArrayOf(1, 4, 128, 128))),
                OrtJavaNamedTensorInfo("timestep", OrtJavaTensorInfo(OrtJavaElementType.INT64, longArrayOf(1))),
                OrtJavaNamedTensorInfo("cond", OrtJavaTensorInfo(OrtJavaElementType.FLOAT, longArrayOf(1, 256, 4096))),
                OrtJavaNamedTensorInfo("uncond", OrtJavaTensorInfo(OrtJavaElementType.FLOAT, longArrayOf(1, 256, 4096))),
            ),
            outputs = mapOf("output_a" to byteArrayOf(1), "output_b" to byteArrayOf(2)),
        )

        val engine = OrtQnnDenoiserEngine(
            artifactManager = artifacts,
            sessionFactory = ReflectionOrtSessionFactory(FakeExecuteOrtJavaBridge(session)),
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
cd android && /opt/gradle-8.10.2/bin/gradle testDebugUnitTest --tests com.example.animanpu.runtime.OrtQnnDenoiserEngineTest --console=plain
```

Expected: FAIL because the current session handle still returns `EXECUTE_NOT_IMPLEMENTED` rather than performing metadata-driven execution.

- [ ] **Step 3: Write the minimal implementation**

```kotlin
// android/app/src/main/java/com/example/animanpu/runtime/OrtSessionFactory.kt
package com.example.animanpu.runtime

import java.io.File

data class OrtProviderProbeResult(
    val qnnAvailable: Boolean,
    val failureReason: OrtRuntimeFailure? = null,
)

data class OrtSessionExecutionResult(
    val qnnActive: Boolean,
    val failureReason: OrtRuntimeFailure? = null,
)

data class OrtSessionCreationResult(
    val sessionCreated: Boolean,
    val handle: OrtSessionHandle? = null,
    val failureReason: OrtRuntimeFailure? = null,
)

interface OrtSessionHandle {
    fun run(outputPath: File): OrtSessionExecutionResult
}

interface OrtSessionFactory {
    fun probeQnnProvider(): OrtProviderProbeResult
    fun providerVisible(): Boolean = probeQnnProvider().qnnAvailable
    fun createDenoiserSession(modelPath: File, config: OrtQnnConfig): OrtSessionCreationResult
}

private fun runtimeInputFile(outputPath: File, fileName: String): File {
    val runtimeRoot = outputPath.parentFile?.parentFile ?: error("Output path missing runtime root")
    return File(runtimeRoot, "inputs/$fileName")
}

class ReflectionOrtSessionFactory(
    private val bridge: OrtJavaBridge = ReflectionOrtJavaBridge(),
) : OrtSessionFactory {
    override fun probeQnnProvider(): OrtProviderProbeResult {
        return try {
            val providers = bridge.availableProviders()
            if (providers.contains("QNNExecutionProvider")) {
                OrtProviderProbeResult(qnnAvailable = true, failureReason = null)
            } else {
                OrtProviderProbeResult(false, OrtRuntimeFailure.QNN_PROVIDER_UNAVAILABLE)
            }
        } catch (_: OrtJavaApiUnavailableException) {
            OrtProviderProbeResult(false, OrtRuntimeFailure.ORT_API_UNAVAILABLE)
        }
    }

    override fun createDenoiserSession(modelPath: File, config: OrtQnnConfig): OrtSessionCreationResult {
        return try {
            val session = bridge.createSession(modelPath, config)
            OrtSessionCreationResult(
                sessionCreated = true,
                handle = object : OrtSessionHandle {
                    override fun run(outputPath: File): OrtSessionExecutionResult {
                        val tensors = linkedMapOf<String, OrtJavaTensorValue>()
                        val outputs = linkedMapOf<String, OrtJavaValue>()
                        try {
                            val inputInfos = session.inputInfos()
                            if (inputInfos.isEmpty()) {
                                return OrtSessionExecutionResult(false, OrtRuntimeFailure.INPUT_METADATA_UNAVAILABLE)
                            }
                            val binding = bindDenoiserInputs(inputInfos)
                            val latentBytes = runtimeInputFile(outputPath, "latent.raw").readBytes()
                            val timestepBytes = runtimeInputFile(outputPath, "timestep.raw").readBytes()
                            val condBytes = runtimeInputFile(outputPath, "cond.raw").readBytes()
                            val uncondBytes = runtimeInputFile(outputPath, "uncond.raw").readBytes()
                            if (latentBytes.size.toLong() != expectedByteCount(binding.latent.info) ||
                                timestepBytes.size.toLong() != expectedByteCount(binding.timestep.info) ||
                                condBytes.size.toLong() != expectedByteCount(binding.cond.info) ||
                                uncondBytes.size.toLong() != expectedByteCount(binding.uncond.info)) {
                                return OrtSessionExecutionResult(false, OrtRuntimeFailure.INPUT_FILE_SIZE_MISMATCH)
                            }
                            tensors[binding.latent.name] = bridge.tensorFromRaw(binding.latent.info, latentBytes)
                            tensors[binding.timestep.name] = bridge.tensorFromRaw(binding.timestep.info, timestepBytes)
                            tensors[binding.cond.name] = bridge.tensorFromRaw(binding.cond.info, condBytes)
                            tensors[binding.uncond.name] = bridge.tensorFromRaw(binding.uncond.info, uncondBytes)
                            outputs.putAll(session.run(tensors))
                            if (outputs.size != 1) {
                                return OrtSessionExecutionResult(false, OrtRuntimeFailure.INPUT_MAPPING_FAILED)
                            }
                            val raw = outputs.values.single().readRawBytes()
                            outputPath.parentFile?.mkdirs()
                            outputPath.writeBytes(raw)
                            return OrtSessionExecutionResult(true, null)
                        } catch (error: DenoiserBindingException) {
                            return OrtSessionExecutionResult(false, error.failure)
                        } catch (error: java.io.IOException) {
                            return OrtSessionExecutionResult(false, OrtRuntimeFailure.OUTPUT_WRITE_FAILED)
                        } catch (_: Throwable) {
                            return OrtSessionExecutionResult(false, OrtRuntimeFailure.SESSION_EXECUTE_FAILED)
                        } finally {
                            outputs.values.forEach { value -> runCatching { value.close() } }
                            tensors.values.forEach { value -> runCatching { value.close() } }
                            runCatching { session.close() }
                        }
                    }
                },
                failureReason = null,
            )
        } catch (_: OrtJavaApiUnavailableException) {
            OrtSessionCreationResult(false, null, OrtRuntimeFailure.ORT_API_UNAVAILABLE)
        } catch (_: OrtJavaSessionCreateException) {
            OrtSessionCreationResult(false, null, OrtRuntimeFailure.SESSION_CREATE_FAILED)
        }
    }
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run:

```bash
cd android && /opt/gradle-8.10.2/bin/gradle testDebugUnitTest --tests com.example.animanpu.runtime.OrtQnnDenoiserEngineTest --console=plain
```

Expected: PASS with the new execute-path tests and the previous engine tests all green.

- [ ] **Step 5: Commit**

```bash
git add android/app/src/main/java/com/example/animanpu/runtime/OrtSessionFactory.kt android/app/src/main/java/com/example/animanpu/runtime/OrtJavaBridge.kt android/app/src/test/java/com/example/animanpu/runtime/OrtQnnDenoiserEngineTest.kt
git commit -m "feat: execute android denoiser runtime from session metadata"
```

### Task 5: Run the targeted Android runtime test suite and refresh docs

**Files:**
- Modify: `docs/manual/android-validation.md`
- Test: `android/app/src/test/java/com/example/animanpu/runtime/GenerationOrchestratorTest.kt`
- Test: `android/app/src/test/java/com/example/animanpu/runtime/OrtSessionFactoryTest.kt`
- Test: `android/app/src/test/java/com/example/animanpu/runtime/DenoiserInputBindingTest.kt`
- Test: `android/app/src/test/java/com/example/animanpu/runtime/OrtQnnDenoiserEngineTest.kt`
- Test: `android/app/src/test/java/com/example/animanpu/runtime/OrtRuntimeFailureTest.kt`

- [ ] **Step 1: Write the failing documentation test**

```kotlin
package com.example.animanpu.runtime

import java.nio.file.Files
import java.nio.file.Paths
import org.junit.Assert.assertTrue
import org.junit.Test

class AndroidValidationDocTest {
    @Test
    fun validation_doc_mentions_real_denoiser_execute_and_output_raw() {
        val text = Files.readString(Paths.get("docs/manual/android-validation.md"))

        assertTrue(text.contains("denoiser_output.raw"))
        assertTrue(text.contains("real denoiser execute"))
        assertTrue(text.contains("input size mismatch"))
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cd android && /opt/gradle-8.10.2/bin/gradle testDebugUnitTest --tests com.example.animanpu.runtime.AndroidValidationDocTest --console=plain
```

Expected: FAIL because the current manual validation doc does not yet mention the new execute-stage details.

- [ ] **Step 3: Update the manual validation doc**

```markdown
# Android Denoiser-First Validation Checklist

## Validation Run

1. Install the debug app on the phone.
2. Launch the app and enter the fixed denoiser-first parameters.
3. Start generation.
4. Confirm the app performs one real denoiser execute instead of stopping at session creation.
5. Confirm the app writes `denoiser_output.raw` to the runtime output directory.
6. Confirm the app writes profiling output for the same run.
7. If a runtime file is malformed, confirm the UI surfaces an explicit execute-stage failure such as input size mismatch.
8. Repeat the run three times.
```

- [ ] **Step 4: Run the targeted runtime suite**

Run:

```bash
cd android && /opt/gradle-8.10.2/bin/gradle testDebugUnitTest \
  --tests com.example.animanpu.runtime.GenerationOrchestratorTest \
  --tests com.example.animanpu.runtime.OrtRuntimeFailureTest \
  --tests com.example.animanpu.runtime.OrtSessionFactoryTest \
  --tests com.example.animanpu.runtime.DenoiserInputBindingTest \
  --tests com.example.animanpu.runtime.OrtQnnDenoiserEngineTest \
  --tests com.example.animanpu.runtime.AndroidValidationDocTest \
  --console=plain
```

Expected: PASS with all targeted Android runtime tests green.

- [ ] **Step 5: Commit**

```bash
git add docs/manual/android-validation.md android/app/src/test/java/com/example/animanpu/runtime/AndroidValidationDocTest.kt
git commit -m "docs: update android denoiser execute validation"
```
