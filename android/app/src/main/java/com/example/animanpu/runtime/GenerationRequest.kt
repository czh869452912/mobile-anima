package com.example.animanpu.runtime

data class GenerationRequest(
    val width: Int,
    val height: Int,
    val prompt: String,
    val negativePrompt: String,
    val steps: Int,
    val cfg: Float,
    val maxTokens: Int = 256,
) {
    init {
        val resolution = width to height
        require(
            resolution in setOf(1024 to 1024, 768 to 1024, 1024 to 768)
        ) { "Unsupported resolution: ${width}x${height}" }
        require(steps in setOf(8, 12, 20)) { "Unsupported steps: $steps" }
        require(cfg in 3.0f..7.0f) { "cfg must be between 3.0 and 7.0" }
        require(maxTokens == 256) { "maxTokens must remain fixed at 256 for v1" }
        require(prompt.isNotBlank()) { "prompt must not be blank" }
    }
}
