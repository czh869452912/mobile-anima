package com.example.animanpu.runtime

import java.nio.file.Files
import java.nio.file.Paths
import org.junit.Assert.assertTrue
import org.junit.Test

class AndroidValidationDocTest {
    @Test
    fun validation_doc_mentions_real_denoiser_execute_and_output_raw() {
        val text = Files.readString(Paths.get("docs/manual/android-validation.md"))

        assertTrue(text.contains("denoiser_output.raw"))
        assertTrue(text.contains("real denoiser execute"))
        assertTrue(text.contains("input size mismatch"))
    }
}
