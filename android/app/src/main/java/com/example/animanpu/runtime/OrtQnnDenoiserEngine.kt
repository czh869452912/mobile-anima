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
            )
        }

        val config = OrtQnnConfig(
            backendPath = backendPath,
            profilingPath = artifactManager.profilingCsv().path,
            profilingLevel = "detailed",
            disableCpuFallback = true,
        )
        val session = sessionFactory.createDenoiserSession(artifactManager.denoiserContextOnnx(), config)
        val startedAt = System.currentTimeMillis()
        val executed = session.run(artifactManager.outputTensor())
        val endedAt = System.currentTimeMillis()

        GenerationResult(
            imagePath = artifactManager.outputTensor().path,
            totalDurationMs = endedAt - startedAt,
            denoiseDurationMs = endedAt - startedAt,
            profilingPath = artifactManager.profilingCsv().path,
            qnnActive = executed,
            sessionCreated = true,
            outputTensorPath = artifactManager.outputTensor().path,
        )
    }
}
