package com.example.animanpu.runtime

import java.io.File
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.io.path.createTempDirectory

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
    fun returns_execute_not_implemented_when_session_exists_but_execution_is_placeholder() = runTest {
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
