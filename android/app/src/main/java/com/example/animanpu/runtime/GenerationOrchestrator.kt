package com.example.animanpu.runtime

class GenerationOrchestrator(
    private val engine: GenerationEngine,
) {
    suspend fun generate(request: GenerationRequest): GenerationResult = engine.generate(request)
}
