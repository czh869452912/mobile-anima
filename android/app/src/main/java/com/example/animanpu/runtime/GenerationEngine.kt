package com.example.animanpu.runtime

interface GenerationEngine {
    suspend fun generate(request: GenerationRequest): GenerationResult
}
