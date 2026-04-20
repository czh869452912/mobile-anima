package com.example.animanpu.runtime

data class OrtQnnConfig(
    val backendPath: String,
    val profilingPath: String,
    val profilingLevel: String,
    val disableCpuFallback: Boolean,
) {
    fun providerOptions(): Map<String, String> = buildMap {
        put("backend_path", backendPath)
        put("profiling_level", profilingLevel)
        put("profiling_file_path", profilingPath)
        put("ep.context_enable", "1")
        put("ep.context_embed_mode", "0")
        put(
            "session.disable_cpu_ep_fallback",
            if (disableCpuFallback) "1" else "0",
        )
    }
}
