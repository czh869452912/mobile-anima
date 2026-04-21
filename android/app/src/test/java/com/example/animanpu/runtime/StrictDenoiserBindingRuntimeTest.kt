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
