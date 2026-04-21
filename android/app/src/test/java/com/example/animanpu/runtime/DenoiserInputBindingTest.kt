package com.example.animanpu.runtime

import org.junit.Assert.assertEquals
import org.junit.Test

class DenoiserInputBindingTest {
    @Test
    fun calculates_expected_float32_byte_size_for_static_shape() {
        val info = OrtJavaTensorInfo(
            elementType = OrtJavaElementType.FLOAT,
            shape = longArrayOf(1, 4, 128, 128),
        )

        val bytes = expectedByteCount(info)

        assertEquals(1L * 4L * 128L * 128L * 4L, bytes)
    }

    @Test
    fun rejects_dynamic_shapes() {
        val info = OrtJavaTensorInfo(
            elementType = OrtJavaElementType.FLOAT,
            shape = longArrayOf(1, -1, 128, 128),
        )

        try {
            expectedByteCount(info)
            throw AssertionError("Expected DenoiserBindingException")
        } catch (expected: DenoiserBindingException) {
            assertEquals(OrtRuntimeFailure.DYNAMIC_SHAPE_UNSUPPORTED, expected.failure)
        }
    }

    @Test
    fun rejects_unsupported_tensor_type() {
        val info = OrtJavaTensorInfo(
            elementType = OrtJavaElementType.UNKNOWN,
            shape = longArrayOf(1),
        )

        try {
            expectedByteCount(info)
            throw AssertionError("Expected DenoiserBindingException")
        } catch (expected: DenoiserBindingException) {
            assertEquals(OrtRuntimeFailure.UNSUPPORTED_TENSOR_TYPE, expected.failure)
        }
    }

    @Test
    fun maps_named_inputs_to_runtime_files() {
        val inputs = listOf(
            OrtJavaNamedTensorInfo("latent", OrtJavaTensorInfo(OrtJavaElementType.FLOAT, longArrayOf(1, 4, 128, 128))),
            OrtJavaNamedTensorInfo("timestep", OrtJavaTensorInfo(OrtJavaElementType.INT64, longArrayOf(1))),
            OrtJavaNamedTensorInfo("cond", OrtJavaTensorInfo(OrtJavaElementType.FLOAT, longArrayOf(1, 256, 4096))),
            OrtJavaNamedTensorInfo("uncond", OrtJavaTensorInfo(OrtJavaElementType.FLOAT, longArrayOf(1, 256, 4096))),
        )

        val binding = bindDenoiserInputs(inputs)

        assertEquals("latent", binding.latent.name)
        assertEquals("timestep", binding.timestep.name)
        assertEquals("cond", binding.cond.name)
        assertEquals("uncond", binding.uncond.name)
    }

    @Test
    fun rejects_ambiguous_conditioning_pair() {
        val inputs = listOf(
            OrtJavaNamedTensorInfo("latent", OrtJavaTensorInfo(OrtJavaElementType.FLOAT, longArrayOf(1, 4, 128, 128))),
            OrtJavaNamedTensorInfo("timestep", OrtJavaTensorInfo(OrtJavaElementType.INT64, longArrayOf(1))),
            OrtJavaNamedTensorInfo("context_a", OrtJavaTensorInfo(OrtJavaElementType.FLOAT, longArrayOf(1, 256, 4096))),
            OrtJavaNamedTensorInfo("context_b", OrtJavaTensorInfo(OrtJavaElementType.FLOAT, longArrayOf(1, 256, 4096))),
        )

        try {
            bindDenoiserInputs(inputs)
            throw AssertionError("Expected DenoiserBindingException")
        } catch (expected: DenoiserBindingException) {
            assertEquals(OrtRuntimeFailure.INPUT_MAPPING_FAILED, expected.failure)
        }
    }
}
