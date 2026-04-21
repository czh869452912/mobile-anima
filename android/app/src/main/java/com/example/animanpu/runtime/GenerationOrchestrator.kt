package com.example.animanpu.runtime

class GenerationOrchestrator(
    private val engine: GenerationEngine,
) {
    suspend fun generate(request: GenerationRequest): GenerationResult {
        val result = engine.generate(request)
        if (!result.sessionCreated || !result.qnnActive) {
            throw IllegalStateException("Denoiser runtime did not activate QNN")
        }
        return result
    }
}
