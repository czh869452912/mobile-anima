package com.example.animanpu.runtime

import java.io.File
import kotlin.io.path.createTempDirectory
import kotlin.io.path.pathString
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
    private val providerProbe: OrtProviderProbeResult,
    private val creationResult: OrtSessionCreationResult,
) : OrtSessionFactory {
    override fun probeQnnProvider(): OrtProviderProbeResult = providerProbe

    override fun createDenoiserSession(modelPath: File, config: OrtQnnConfig): OrtSessionCreationResult {
        return creationResult
    }
}

private class FakeExecuteOrtJavaTensorValue : OrtJavaTensorValue {
    override fun close() = Unit
}

private class FakeExecuteOrtJavaValue(
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
        return outputs.mapValues { FakeExecuteOrtJavaValue(it.value) }
    }

    override fun close() = Unit
}

private class FakeExecuteOrtJavaBridge(
    private val session: OrtJavaSession,
) : OrtJavaBridge {
    override fun availableProviders(): Set<String> = setOf("QNNExecutionProvider")

    override fun createSession(modelPath: File, config: OrtQnnConfig): OrtJavaSession = session

    override fun tensorFromRaw(info: OrtJavaTensorInfo, rawBytes: ByteArray): OrtJavaTensorValue = FakeExecuteOrtJavaTensorValue()
}

class OrtQnnDenoiserEngineTest {
    @Test
    fun returns_missing_artifact_failure_before_session_creation() = runTest {
        val root = createTempDirectory(prefix = "anima-engine-").toFile()
        val artifacts = ArtifactManager(root)
        artifacts.ensureRuntimeDirectories()

        val engine = OrtQnnDenoiserEngine(
            artifactManager = artifacts,
            sessionFactory = FakeOrtSessionFactory(
                providerProbe = OrtProviderProbeResult(qnnAvailable = true, failureReason = null),
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
    fun returns_ort_api_unavailable_when_provider_probe_cannot_load_ort() = runTest {
        val root = createTempDirectory(prefix = "anima-engine-").toFile()
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
                providerProbe = OrtProviderProbeResult(
                    qnnAvailable = false,
                    failureReason = OrtRuntimeFailure.ORT_API_UNAVAILABLE,
                ),
                creationResult = OrtSessionCreationResult(sessionCreated = true),
            ),
            backendPath = "/data/local/tmp/libQnnHtp.so",
        )

        val result = engine.generate(
            GenerationRequest(1024, 1024, "cat astronaut", "blurry", 8, 5.0f),
        )

        assertFalse(result.sessionCreated)
        assertFalse(result.qnnActive)
        assertEquals(OrtRuntimeFailure.ORT_API_UNAVAILABLE, result.failureReason)
    }

    @Test
    fun returns_qnn_provider_unavailable_when_probe_finds_no_qnn_provider() = runTest {
        val root = createTempDirectory(prefix = "anima-engine-").toFile()
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
                providerProbe = OrtProviderProbeResult(
                    qnnAvailable = false,
                    failureReason = OrtRuntimeFailure.QNN_PROVIDER_UNAVAILABLE,
                ),
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
        val root = createTempDirectory(prefix = "anima-engine-").toFile()
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
                providerProbe = OrtProviderProbeResult(qnnAvailable = true, failureReason = null),
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
