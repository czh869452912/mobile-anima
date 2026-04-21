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
