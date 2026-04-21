package com.example.animanpu.runtime

import kotlinx.coroutines.test.runTest
import org.junit.Assert.assertEquals
import org.junit.Test

class GenerationOrchestratorTest {
    @Test
    fun delegatesToEngineAndReturnsResult() = runTest {
        val expected = GenerationResult(
            imagePath = "/tmp/output.png",
            totalDurationMs = 1234,
            denoiseDurationMs = 1100,
            profilingPath = "/tmp/profile.csv",
            qnnActive = true,
            sessionCreated = true,
            outputTensorPath = "/tmp/output.raw",
            failureReason = null,
        )
        val engine = object : GenerationEngine {
            override suspend fun generate(request: GenerationRequest): GenerationResult = expected
        }
        val orchestrator = GenerationOrchestrator(engine)

        val result = orchestrator.generate(
            GenerationRequest(
                width = 1024,
                height = 1024,
                prompt = "cat astronaut",
                negativePrompt = "blurry",
                steps = 8,
                cfg = 5.0f,
            )
        )

        assertEquals(expected, result)
    }

    @Test
    fun rejects_result_when_qnn_is_not_active_and_includes_failure_code() = runTest {
        val engine = object : GenerationEngine {
            override suspend fun generate(request: GenerationRequest): GenerationResult {
                return GenerationResult(
                    imagePath = "",
                    totalDurationMs = 0,
                    denoiseDurationMs = 0,
                    profilingPath = "/tmp/profile.csv",
                    qnnActive = false,
                    sessionCreated = false,
                    outputTensorPath = "/tmp/output.raw",
                    failureReason = OrtRuntimeFailure.QNN_PROVIDER_UNAVAILABLE,
                )
            }
        }
        val orchestrator = GenerationOrchestrator(engine)

        try {
            orchestrator.generate(
                GenerationRequest(
                    width = 1024,
                    height = 1024,
                    prompt = "cat astronaut",
                    negativePrompt = "blurry",
                    steps = 8,
                    cfg = 5.0f,
                )
            )
            throw AssertionError("Expected IllegalStateException")
        } catch (expected: IllegalStateException) {
            assertEquals(
                "Denoiser runtime did not activate QNN (qnn_provider_unavailable)",
                expected.message,
            )
        }
    }
}
