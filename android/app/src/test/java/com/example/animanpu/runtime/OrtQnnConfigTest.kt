package com.example.animanpu.runtime

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class OrtQnnConfigTest {
    @Test
    fun buildsProviderOptionsWithNoFallbackAndProfiling() {
        val config = OrtQnnConfig(
            backendPath = "/data/local/tmp/libQnnHtp.so",
            profilingPath = "/sdcard/Download/denoiser_profile.csv",
            profilingLevel = "detailed",
            disableCpuFallback = true,
        )

        val options = config.providerOptions()

        assertEquals("/data/local/tmp/libQnnHtp.so", options["backend_path"])
        assertEquals("detailed", options["profiling_level"])
        assertEquals("/sdcard/Download/denoiser_profile.csv", options["profiling_file_path"])
        assertEquals("1", options["session.disable_cpu_ep_fallback"])
        assertTrue(options.containsKey("ep.context_enable"))
    }
}
