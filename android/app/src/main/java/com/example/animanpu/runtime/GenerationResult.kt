package com.example.animanpu.runtime

data class GenerationResult(
    val imagePath: String,
    val totalDurationMs: Long,
    val denoiseDurationMs: Long,
    val profilingPath: String,
    val qnnActive: Boolean,
    val sessionCreated: Boolean = false,
    val outputTensorPath: String = "",
)
