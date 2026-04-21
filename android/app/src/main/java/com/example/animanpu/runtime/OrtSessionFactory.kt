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

class ReflectionOrtSessionFactory(
    private val bridge: OrtJavaBridge = ReflectionOrtJavaBridge(),
) : OrtSessionFactory {
    override fun probeQnnProvider(): OrtProviderProbeResult {
        return try {
            val providers = bridge.availableProviders()
            if (providers.contains("QNNExecutionProvider")) {
                OrtProviderProbeResult(qnnAvailable = true, failureReason = null)
            } else {
                OrtProviderProbeResult(
                    qnnAvailable = false,
                    failureReason = OrtRuntimeFailure.QNN_PROVIDER_UNAVAILABLE,
                )
            }
        } catch (_: OrtJavaApiUnavailableException) {
            OrtProviderProbeResult(
                qnnAvailable = false,
                failureReason = OrtRuntimeFailure.ORT_API_UNAVAILABLE,
            )
        }
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
