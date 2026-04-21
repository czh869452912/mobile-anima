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

        val probe = sessionFactory.probeQnnProvider()
        if (!probe.qnnAvailable) {
            return@withContext GenerationResult(
                imagePath = "",
                totalDurationMs = 0,
                denoiseDurationMs = 0,
                profilingPath = artifactManager.profilingCsv().path,
                qnnActive = false,
                sessionCreated = false,
                outputTensorPath = artifactManager.outputTensor().path,
                failureReason = probe.failureReason,
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
