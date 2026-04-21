# Android ORT Java + QNN Provider Gap Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the fake Android ORT bridge with a real `ORT Java + QNN` session-construction seam, explicit runtime failure categories, and JVM-testable engine/factory behavior.

**Architecture:** Keep `OrtQnnDenoiserEngine` as the runtime entrypoint, but move ORT Java interaction behind a small bridge and make session creation return an explicit outcome instead of a bare handle. Preserve the current Android app shell and denoiser-first flow while adding enough structure to distinguish `ORT API`, `QNN provider`, `session creation`, and `execute-stage` failures.

**Tech Stack:** Kotlin, Android app module, JVM unit tests via Gradle/JUnit4, ONNX Runtime Java API accessed via reflection

---

### Task 1: Add explicit Android runtime failure types

**Files:**
- Create: `android/app/src/main/java/com/example/animanpu/runtime/OrtRuntimeFailure.kt`
- Modify: `android/app/src/main/java/com/example/animanpu/runtime/GenerationResult.kt`
- Modify: `android/app/src/main/java/com/example/animanpu/runtime/GenerationOrchestrator.kt`
- Test: `android/app/src/test/java/com/example/animanpu/runtime/GenerationOrchestratorTest.kt`

- [ ] **Step 1: Write the failing orchestrator test**

```kotlin
package com.example.animanpu.runtime

import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Test

class GenerationOrchestratorTest {
    @Test
    fun delegatesToEngineAndReturnsResult() = runTest {
        val expected = GenerationResult(
            imagePath = "/tmp/output.png",
            totalDurationMs = 1234,
            denoiseDurationMs = 1100,
            profilingPath = "/tmp/profile.csv",
            qnnActive = true,
            sessionCreated = true,
            outputTensorPath = "/tmp/output.raw",
            failureReason = null,
        )
        val engine = object : GenerationEngine {
            override suspend fun generate(request: GenerationRequest): GenerationResult = expected
        }
        val orchestrator = GenerationOrchestrator(engine)

        val result = orchestrator.generate(
            GenerationRequest(
                width = 1024,
                height = 1024,
                prompt = "cat astronaut",
                negativePrompt = "blurry",
                steps = 8,
                cfg = 5.0f,
            )
        )

        assertEquals(expected, result)
    }

    @Test
    fun rejects_result_when_qnn_is_not_active_and_includes_failure_code() = runTest {
        val engine = object : GenerationEngine {
            override suspend fun generate(request: GenerationRequest): GenerationResult {
                return GenerationResult(
                    imagePath = "",
                    totalDurationMs = 0,
                    denoiseDurationMs = 0,
                    profilingPath = "/tmp/profile.csv",
                    qnnActive = false,
                    sessionCreated = false,
                    outputTensorPath = "/tmp/output.raw",
                    failureReason = OrtRuntimeFailure.QNN_PROVIDER_UNAVAILABLE,
                )
            }
        }
        val orchestrator = GenerationOrchestrator(engine)

        try {
            orchestrator.generate(
                GenerationRequest(
                    width = 1024,
                    height = 1024,
                    prompt = "cat astronaut",
                    negativePrompt = "blurry",
                    steps = 8,
                    cfg = 5.0f,
                )
            )
            throw AssertionError("Expected IllegalStateException")
        } catch (expected: IllegalStateException) {
            assertEquals(
                "Denoiser runtime did not activate QNN (qnn_provider_unavailable)",
                expected.message,
            )
        }
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cd android && ./gradlew testDebugUnitTest --tests com.example.animanpu.runtime.GenerationOrchestratorTest --console=plain
```

Expected: FAIL with unresolved `OrtRuntimeFailure` and/or `failureReason` constructor argument errors.

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
}
```

```kotlin
// android/app/src/main/java/com/example/animanpu/runtime/GenerationResult.kt
package com.example.animanpu.runtime

data class GenerationResult(
    val imagePath: String,
    val totalDurationMs: Long,
    val denoiseDurationMs: Long,
    val profilingPath: String,
    val qnnActive: Boolean,
    val sessionCreated: Boolean = false,
    val outputTensorPath: String = "",
    val failureReason: OrtRuntimeFailure? = null,
)
```

```kotlin
// android/app/src/main/java/com/example/animanpu/runtime/GenerationOrchestrator.kt
package com.example.animanpu.runtime

class GenerationOrchestrator(
    private val engine: GenerationEngine,
) {
    suspend fun generate(request: GenerationRequest): GenerationResult {
        val result = engine.generate(request)
        if (!result.sessionCreated || !result.qnnActive) {
            val detail = result.failureReason?.code ?: "unknown_runtime_failure"
            throw IllegalStateException("Denoiser runtime did not activate QNN ($detail)")
        }
        return result
    }
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
cd android && ./gradlew testDebugUnitTest --tests com.example.animanpu.runtime.GenerationOrchestratorTest --console=plain
```

Expected: PASS with `2 tests completed, 0 failed`.

- [ ] **Step 5: Commit**

```bash
git add \
  android/app/src/main/java/com/example/animanpu/runtime/OrtRuntimeFailure.kt \
  android/app/src/main/java/com/example/animanpu/runtime/GenerationResult.kt \
  android/app/src/main/java/com/example/animanpu/runtime/GenerationOrchestrator.kt \
  android/app/src/test/java/com/example/animanpu/runtime/GenerationOrchestratorTest.kt
git commit -m "test: add android runtime failure categories"
```

### Task 2: Introduce a fakeable ORT Java bridge and explicit session-creation outcomes

**Files:**
- Create: `android/app/src/main/java/com/example/animanpu/runtime/OrtJavaBridge.kt`
- Modify: `android/app/src/main/java/com/example/animanpu/runtime/OrtSessionFactory.kt`
- Test: `android/app/src/test/java/com/example/animanpu/runtime/OrtSessionFactoryTest.kt`

- [ ] **Step 1: Write the failing factory tests**

```kotlin
package com.example.animanpu.runtime

import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

private class NamedProvider(
    private val providerName: String,
) {
    fun getName(): String = providerName

    override fun toString(): String = "ignored"
}

private class LegacyProvider(
    private val text: String,
) {
    override fun toString(): String = text
}

private class FakeOrtJavaSession : OrtJavaSession {
    override fun close() = Unit
}

private class FakeOrtJavaBridge(
    private val providers: Set<String> = emptySet(),
    private val failureMode: String? = null,
) : OrtJavaBridge {
    override fun availableProviders(): Set<String> {
        if (failureMode == "api") {
            throw OrtJavaApiUnavailableException(IllegalStateException("api unavailable"))
        }
        return providers
    }

    override fun createSession(modelPath: File, config: OrtQnnConfig): OrtJavaSession {
        return when (failureMode) {
            "api" -> throw OrtJavaApiUnavailableException(IllegalStateException("api unavailable"))
            "session" -> throw OrtJavaSessionCreateException(IllegalStateException("session failed"))
            else -> FakeOrtJavaSession()
        }
    }
}

class OrtSessionFactoryTest {
    @Test
    fun normalize_provider_names_prefers_get_name_when_present() {
        val providers = normalizeProviderNames(listOf(NamedProvider("QNNExecutionProvider"), NamedProvider("CPUExecutionProvider")))

        assertEquals(setOf("QNNExecutionProvider", "CPUExecutionProvider"), providers)
    }

    @Test
    fun normalize_provider_names_falls_back_to_to_string_when_get_name_is_absent() {
        val providers = normalizeProviderNames(listOf(LegacyProvider("QNNExecutionProvider")))

        assertEquals(setOf("QNNExecutionProvider"), providers)
    }

    @Test
    fun reports_qnn_provider_visibility_from_bridge_provider_names() {
        val factory = ReflectionOrtSessionFactory(
            bridge = FakeOrtJavaBridge(providers = setOf("QNNExecutionProvider", "CPUExecutionProvider")),
        )

        assertTrue(factory.providerVisible())
    }

    @Test
    fun returns_false_when_qnn_provider_is_not_visible() {
        val factory = ReflectionOrtSessionFactory(
            bridge = FakeOrtJavaBridge(providers = setOf("CPUExecutionProvider")),
        )

        assertFalse(factory.providerVisible())
    }

    @Test
    fun returns_api_unavailable_creation_result_when_bridge_cannot_load_ort() {
        val factory = ReflectionOrtSessionFactory(
            bridge = FakeOrtJavaBridge(failureMode = "api"),
        )

        val result = factory.createDenoiserSession(
            modelPath = File("/tmp/denoiser_ctx.onnx"),
            config = OrtQnnConfig(
                backendPath = "/data/local/tmp/libQnnHtp.so",
                profilingPath = "/tmp/profile.csv",
                profilingLevel = "detailed",
                disableCpuFallback = true,
            ),
        )

        assertFalse(result.sessionCreated)
        assertEquals(OrtRuntimeFailure.ORT_API_UNAVAILABLE, result.failureReason)
        assertNull(result.handle)
    }

    @Test
    fun returns_session_create_failed_when_bridge_cannot_create_session() {
        val factory = ReflectionOrtSessionFactory(
            bridge = FakeOrtJavaBridge(
                providers = setOf("QNNExecutionProvider"),
                failureMode = "session",
            ),
        )

        val result = factory.createDenoiserSession(
            modelPath = File("/tmp/denoiser_ctx.onnx"),
            config = OrtQnnConfig(
                backendPath = "/data/local/tmp/libQnnHtp.so",
                profilingPath = "/tmp/profile.csv",
                profilingLevel = "detailed",
                disableCpuFallback = true,
            ),
        )

        assertFalse(result.sessionCreated)
        assertEquals(OrtRuntimeFailure.SESSION_CREATE_FAILED, result.failureReason)
        assertNull(result.handle)
    }

    @Test
    fun returns_session_handle_when_bridge_creates_session() {
        val factory = ReflectionOrtSessionFactory(
            bridge = FakeOrtJavaBridge(providers = setOf("QNNExecutionProvider")),
        )

        val result = factory.createDenoiserSession(
            modelPath = File("/tmp/denoiser_ctx.onnx"),
            config = OrtQnnConfig(
                backendPath = "/data/local/tmp/libQnnHtp.so",
                profilingPath = "/tmp/profile.csv",
                profilingLevel = "detailed",
                disableCpuFallback = true,
            ),
        )

        assertTrue(result.sessionCreated)
        assertNull(result.failureReason)
        assertNotNull(result.handle)

        val execution = result.handle!!.run(File("/tmp/denoiser_output.raw"))
        assertFalse(execution.qnnActive)
        assertEquals(OrtRuntimeFailure.EXECUTE_NOT_IMPLEMENTED, execution.failureReason)
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cd android && ./gradlew testDebugUnitTest --tests com.example.animanpu.runtime.OrtSessionFactoryTest --console=plain
```

Expected: FAIL with unresolved references for `OrtJavaBridge`, `OrtJavaSession`, `OrtJavaApiUnavailableException`, `OrtJavaSessionCreateException`, and the new factory result types.

- [ ] **Step 3: Write the minimal implementation**

```kotlin
// android/app/src/main/java/com/example/animanpu/runtime/OrtJavaBridge.kt
package com.example.animanpu.runtime

import java.io.File

interface OrtJavaSession : AutoCloseable {
    override fun close()
}

interface OrtJavaBridge {
    fun availableProviders(): Set<String>
    fun createSession(modelPath: File, config: OrtQnnConfig): OrtJavaSession
}

class OrtJavaApiUnavailableException(
    cause: Throwable,
) : RuntimeException(cause)

class OrtJavaSessionCreateException(
    cause: Throwable,
) : RuntimeException(cause)

internal fun normalizeProviderNames(rawProviders: Iterable<*>): Set<String> =
    rawProviders.mapNotNull { provider ->
        if (provider == null) {
            null
        } else {
            runCatching {
                provider.javaClass.getMethod("getName").invoke(provider)?.toString()
            }.getOrNull() ?: provider.toString()
        }
    }.toSet()

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
            sessionOptionsClass.getMethod("addQnn", Map::class.java).invoke(sessionOptions, config.providerOptions())
            val session = environmentClass
                .getMethod("createSession", String::class.java, sessionOptionsClass)
                .invoke(environment, modelPath.path, sessionOptions)

            return object : OrtJavaSession {
                private var closed = false

                override fun close() {
                    if (closed) {
                        return
                    }
                    runCatching {
                        session.javaClass.getMethod("close").invoke(session)
                    }
                    runCatching {
                        sessionOptionsClass.getMethod("close").invoke(sessionOptions)
                    }
                    closed = true
                }
            }
        } catch (error: RuntimeException) {
            throw error
        } catch (error: Throwable) {
            throw OrtJavaSessionCreateException(error)
        }
    }
}
```

```kotlin
// android/app/src/main/java/com/example/animanpu/runtime/OrtSessionFactory.kt
package com.example.animanpu.runtime

import java.io.File

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
    fun providerVisible(): Boolean
    fun createDenoiserSession(modelPath: File, config: OrtQnnConfig): OrtSessionCreationResult
}

class ReflectionOrtSessionFactory(
    private val bridge: OrtJavaBridge = ReflectionOrtJavaBridge(),
) : OrtSessionFactory {
    override fun providerVisible(): Boolean {
        return runCatching {
            bridge.availableProviders().contains("QNNExecutionProvider")
        }.getOrDefault(false)
    }

    override fun createDenoiserSession(modelPath: File, config: OrtQnnConfig): OrtSessionCreationResult {
        return try {
            val session = bridge.createSession(modelPath, config)
            OrtSessionCreationResult(
                sessionCreated = true,
                handle = object : OrtSessionHandle {
                    override fun run(outputPath: File): OrtSessionExecutionResult {
                        outputPath.parentFile?.mkdirs()
                        session.close()
                        return OrtSessionExecutionResult(
                            qnnActive = false,
                            failureReason = OrtRuntimeFailure.EXECUTE_NOT_IMPLEMENTED,
                        )
                    }
                },
                failureReason = null,
            )
        } catch (_: OrtJavaApiUnavailableException) {
            OrtSessionCreationResult(
                sessionCreated = false,
                handle = null,
                failureReason = OrtRuntimeFailure.ORT_API_UNAVAILABLE,
            )
        } catch (_: OrtJavaSessionCreateException) {
            OrtSessionCreationResult(
                sessionCreated = false,
                handle = null,
                failureReason = OrtRuntimeFailure.SESSION_CREATE_FAILED,
            )
        }
    }
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
cd android && ./gradlew testDebugUnitTest --tests com.example.animanpu.runtime.OrtSessionFactoryTest --console=plain
```

Expected: PASS with `7 tests completed, 0 failed`.

- [ ] **Step 5: Commit**

```bash
git add \
  android/app/src/main/java/com/example/animanpu/runtime/OrtJavaBridge.kt \
  android/app/src/main/java/com/example/animanpu/runtime/OrtSessionFactory.kt \
  android/app/src/test/java/com/example/animanpu/runtime/OrtSessionFactoryTest.kt
git commit -m "test: add android ort session factory seam"
```

### Task 3: Map engine outcomes to explicit runtime failures

**Files:**
- Modify: `android/app/src/main/java/com/example/animanpu/runtime/OrtQnnDenoiserEngine.kt`
- Modify: `android/app/src/test/java/com/example/animanpu/runtime/OrtQnnDenoiserEngineTest.kt`
- Test: `android/app/src/test/java/com/example/animanpu/runtime/OrtQnnDenoiserEngineTest.kt`

- [ ] **Step 1: Write the failing engine tests**

```kotlin
package com.example.animanpu.runtime

import java.io.File
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

private class FakeOrtSessionHandle(
    private val executionResult: OrtSessionExecutionResult,
) : OrtSessionHandle {
    override fun run(outputPath: File): OrtSessionExecutionResult {
        if (executionResult.qnnActive) {
            outputPath.parentFile?.mkdirs()
            outputPath.writeText("ok")
        }
        return executionResult
    }
}

private class FakeOrtSessionFactory(
    private val providerVisible: Boolean,
    private val creationResult: OrtSessionCreationResult,
) : OrtSessionFactory {
    override fun providerVisible(): Boolean = providerVisible

    override fun createDenoiserSession(modelPath: File, config: OrtQnnConfig): OrtSessionCreationResult {
        return creationResult
    }
}

class OrtQnnDenoiserEngineTest {
    @Test
    fun returns_missing_artifact_failure_before_session_creation() = runTest {
        val root = createTempDir(prefix = "anima-engine-")
        val artifacts = ArtifactManager(root)
        artifacts.ensureRuntimeDirectories()

        val engine = OrtQnnDenoiserEngine(
            artifactManager = artifacts,
            sessionFactory = FakeOrtSessionFactory(
                providerVisible = true,
                creationResult = OrtSessionCreationResult(sessionCreated = true),
            ),
            backendPath = "/data/local/tmp/libQnnHtp.so",
        )

        val result = engine.generate(
            GenerationRequest(1024, 1024, "cat astronaut", "blurry", 8, 5.0f),
        )

        assertFalse(result.sessionCreated)
        assertFalse(result.qnnActive)
        assertEquals(OrtRuntimeFailure.MISSING_RUNTIME_ARTIFACT, result.failureReason)
    }

    @Test
    fun returns_provider_unavailable_failure_before_session_creation() = runTest {
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
            sessionFactory = FakeOrtSessionFactory(
                providerVisible = false,
                creationResult = OrtSessionCreationResult(sessionCreated = true),
            ),
            backendPath = "/data/local/tmp/libQnnHtp.so",
        )

        val result = engine.generate(
            GenerationRequest(1024, 1024, "cat astronaut", "blurry", 8, 5.0f),
        )

        assertFalse(result.sessionCreated)
        assertFalse(result.qnnActive)
        assertEquals(OrtRuntimeFailure.QNN_PROVIDER_UNAVAILABLE, result.failureReason)
    }

    @Test
    fun returns_session_create_failed_when_factory_cannot_create_session() = runTest {
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
            sessionFactory = FakeOrtSessionFactory(
                providerVisible = true,
                creationResult = OrtSessionCreationResult(
                    sessionCreated = false,
                    handle = null,
                    failureReason = OrtRuntimeFailure.SESSION_CREATE_FAILED,
                ),
            ),
            backendPath = "/data/local/tmp/libQnnHtp.so",
        )

        val result = engine.generate(
            GenerationRequest(1024, 1024, "cat astronaut", "blurry", 8, 5.0f),
        )

        assertFalse(result.sessionCreated)
        assertFalse(result.qnnActive)
        assertEquals(OrtRuntimeFailure.SESSION_CREATE_FAILED, result.failureReason)
    }

    @Test
    fun returns_execute_not_implemented_when_session_exists_but_execution_is_placeholder() = runTest {
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
            sessionFactory = FakeOrtSessionFactory(
                providerVisible = true,
                creationResult = OrtSessionCreationResult(
                    sessionCreated = true,
                    handle = FakeOrtSessionHandle(
                        OrtSessionExecutionResult(
                            qnnActive = false,
                            failureReason = OrtRuntimeFailure.EXECUTE_NOT_IMPLEMENTED,
                        ),
                    ),
                    failureReason = null,
                ),
            ),
            backendPath = "/data/local/tmp/libQnnHtp.so",
        )

        val result = engine.generate(
            GenerationRequest(1024, 1024, "cat astronaut", "blurry", 8, 5.0f),
        )

        assertTrue(result.sessionCreated)
        assertFalse(result.qnnActive)
        assertEquals(OrtRuntimeFailure.EXECUTE_NOT_IMPLEMENTED, result.failureReason)
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
            sessionFactory = FakeOrtSessionFactory(
                providerVisible = true,
                creationResult = OrtSessionCreationResult(
                    sessionCreated = true,
                    handle = FakeOrtSessionHandle(
                        OrtSessionExecutionResult(
                            qnnActive = true,
                            failureReason = null,
                        ),
                    ),
                    failureReason = null,
                ),
            ),
            backendPath = "/data/local/tmp/libQnnHtp.so",
        )

        val result = engine.generate(
            GenerationRequest(1024, 1024, "cat astronaut", "blurry", 8, 5.0f),
        )

        assertTrue(result.sessionCreated)
        assertTrue(result.qnnActive)
        assertEquals(artifacts.outputTensor().path, result.outputTensorPath)
        assertEquals(artifacts.profilingCsv().path, result.profilingPath)
        assertNull(result.failureReason)
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cd android && ./gradlew testDebugUnitTest --tests com.example.animanpu.runtime.OrtQnnDenoiserEngineTest --console=plain
```

Expected: FAIL with signature mismatches against the old `OrtSessionFactory` and `OrtSessionHandle` contracts.

- [ ] **Step 3: Write the minimal implementation**

```kotlin
// android/app/src/main/java/com/example/animanpu/runtime/OrtQnnDenoiserEngine.kt
package com.example.animanpu.runtime

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

class OrtQnnDenoiserEngine(
    private val artifactManager: ArtifactManager,
    private val sessionFactory: OrtSessionFactory,
    private val backendPath: String,
) : GenerationEngine {
    override suspend fun generate(request: GenerationRequest): GenerationResult = withContext(Dispatchers.IO) {
        artifactManager.ensureRuntimeDirectories()

        val required = listOf(
            artifactManager.denoiserContextOnnx(),
            artifactManager.denoiserContextBin(),
            artifactManager.latentInput(),
            artifactManager.timestepInput(),
            artifactManager.condInput(),
            artifactManager.uncondInput(),
        )

        if (required.any { !it.exists() }) {
            return@withContext GenerationResult(
                imagePath = "",
                totalDurationMs = 0,
                denoiseDurationMs = 0,
                profilingPath = artifactManager.profilingCsv().path,
                qnnActive = false,
                sessionCreated = false,
                outputTensorPath = artifactManager.outputTensor().path,
                failureReason = OrtRuntimeFailure.MISSING_RUNTIME_ARTIFACT,
            )
        }

        if (!sessionFactory.providerVisible()) {
            return@withContext GenerationResult(
                imagePath = "",
                totalDurationMs = 0,
                denoiseDurationMs = 0,
                profilingPath = artifactManager.profilingCsv().path,
                qnnActive = false,
                sessionCreated = false,
                outputTensorPath = artifactManager.outputTensor().path,
                failureReason = OrtRuntimeFailure.QNN_PROVIDER_UNAVAILABLE,
            )
        }

        val config = OrtQnnConfig(
            backendPath = backendPath,
            profilingPath = artifactManager.profilingCsv().path,
            profilingLevel = "detailed",
            disableCpuFallback = true,
        )
        val creation = sessionFactory.createDenoiserSession(artifactManager.denoiserContextOnnx(), config)
        if (!creation.sessionCreated || creation.handle == null) {
            return@withContext GenerationResult(
                imagePath = "",
                totalDurationMs = 0,
                denoiseDurationMs = 0,
                profilingPath = artifactManager.profilingCsv().path,
                qnnActive = false,
                sessionCreated = false,
                outputTensorPath = artifactManager.outputTensor().path,
                failureReason = creation.failureReason,
            )
        }

        val startedAt = System.currentTimeMillis()
        val executed = creation.handle.run(artifactManager.outputTensor())
        val endedAt = System.currentTimeMillis()

        GenerationResult(
            imagePath = artifactManager.outputTensor().path,
            totalDurationMs = endedAt - startedAt,
            denoiseDurationMs = endedAt - startedAt,
            profilingPath = artifactManager.profilingCsv().path,
            qnnActive = executed.qnnActive,
            sessionCreated = true,
            outputTensorPath = artifactManager.outputTensor().path,
            failureReason = executed.failureReason,
        )
    }
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
cd android && ./gradlew testDebugUnitTest --tests com.example.animanpu.runtime.OrtQnnDenoiserEngineTest --console=plain
```

Expected: PASS with `5 tests completed, 0 failed`.

- [ ] **Step 5: Commit**

```bash
git add \
  android/app/src/main/java/com/example/animanpu/runtime/OrtQnnDenoiserEngine.kt \
  android/app/src/test/java/com/example/animanpu/runtime/OrtQnnDenoiserEngineTest.kt
git commit -m "test: map android denoiser runtime failures"
```
