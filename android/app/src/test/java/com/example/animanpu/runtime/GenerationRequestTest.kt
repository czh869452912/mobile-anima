package com.example.animanpu.runtime

import org.junit.Assert.assertEquals
import org.junit.Test

class GenerationRequestTest {
    @Test
    fun rejectsUnsupportedResolution() {
        try {
            GenerationRequest(
                width = 640,
                height = 640,
                prompt = "cat astronaut",
                negativePrompt = "blurry",
                steps = 12,
                cfg = 5.0f,
            )
            throw AssertionError("Expected IllegalArgumentException")
        } catch (expected: IllegalArgumentException) {
            assertEquals("Unsupported resolution: 640x640", expected.message)
        }
    }

    @Test
    fun acceptsSupportedRequest() {
        val request = GenerationRequest(
            width = 1024,
            height = 1024,
            prompt = "cat astronaut",
            negativePrompt = "blurry",
            steps = 12,
            cfg = 5.0f,
        )

        assertEquals(1024, request.width)
        assertEquals(1024, request.height)
    }
}
