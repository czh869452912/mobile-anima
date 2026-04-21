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
        val providers = normalizeProviderNames(
            listOf(NamedProvider("QNNExecutionProvider"), NamedProvider("CPUExecutionProvider")),
        )

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

        val result = factory.probeQnnProvider()

        assertTrue(result.qnnAvailable)
        assertNull(result.failureReason)
    }

    @Test
    fun returns_qnn_provider_unavailable_when_qnn_is_not_visible() {
        val factory = ReflectionOrtSessionFactory(
            bridge = FakeOrtJavaBridge(providers = setOf("CPUExecutionProvider")),
        )

        val result = factory.probeQnnProvider()

        assertFalse(result.qnnAvailable)
        assertEquals(OrtRuntimeFailure.QNN_PROVIDER_UNAVAILABLE, result.failureReason)
    }

    @Test
    fun returns_api_unavailable_probe_result_when_bridge_cannot_load_ort() {
        val factory = ReflectionOrtSessionFactory(
            bridge = FakeOrtJavaBridge(failureMode = "api"),
        )

        val result = factory.probeQnnProvider()

        assertFalse(result.qnnAvailable)
        assertEquals(OrtRuntimeFailure.ORT_API_UNAVAILABLE, result.failureReason)
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
