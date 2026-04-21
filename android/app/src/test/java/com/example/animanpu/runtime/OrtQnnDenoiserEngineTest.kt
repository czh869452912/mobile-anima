package com.example.animanpu.runtime

import java.io.File
import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

private class FakeOrtSessionHandle(
    private val executeSucceeds: Boolean,
) : OrtSessionHandle {
    override fun run(outputPath: File): Boolean {
        if (executeSucceeds) {
            outputPath.parentFile?.mkdirs()
            outputPath.writeText("ok")
        }
        return executeSucceeds
    }
}

private class FakeOrtSessionFactory(
    private val providerVisible: Boolean,
    private val executeSucceeds: Boolean,
) : OrtSessionFactory {
    override fun providerVisible(): Boolean = providerVisible

    override fun createDenoiserSession(modelPath: File, config: OrtQnnConfig): OrtSessionHandle {
        return FakeOrtSessionHandle(executeSucceeds)
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
            sessionFactory = FakeOrtSessionFactory(providerVisible = true, executeSucceeds = true),
            backendPath = "/data/local/tmp/libQnnHtp.so",
        )

        val result = engine.generate(
            GenerationRequest(1024, 1024, "cat astronaut", "blurry", 8, 5.0f),
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
            GenerationRequest(1024, 1024, "cat astronaut", "blurry", 8, 5.0f),
        )

        assertTrue(result.sessionCreated)
        assertTrue(result.qnnActive)
        assertEquals(artifacts.outputTensor().path, result.outputTensorPath)
        assertEquals(artifacts.profilingCsv().path, result.profilingPath)
    }
}
