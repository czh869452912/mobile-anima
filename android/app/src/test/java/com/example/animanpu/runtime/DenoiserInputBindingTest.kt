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
    fun binds_unique_4d_float_latent_even_without_latent_in_the_name() {
        val inputs = listOf(
            OrtJavaNamedTensorInfo("sample", OrtJavaTensorInfo(OrtJavaElementType.FLOAT, longArrayOf(1, 4, 128, 128))),
            OrtJavaNamedTensorInfo("timestep", OrtJavaTensorInfo(OrtJavaElementType.INT64, longArrayOf(1))),
            OrtJavaNamedTensorInfo("positive_cond", OrtJavaTensorInfo(OrtJavaElementType.FLOAT, longArrayOf(1, 256, 4096))),
            OrtJavaNamedTensorInfo("negative_uncond", OrtJavaTensorInfo(OrtJavaElementType.FLOAT, longArrayOf(1, 256, 4096))),
        )

        val binding = bindDenoiserInputs(inputs)

        assertEquals("sample", binding.latent.name)
        assertEquals("timestep", binding.timestep.name)
        assertEquals("positive_cond", binding.cond.name)
        assertEquals("negative_uncond", binding.uncond.name)
    }

    @Test
    fun rejects_multiple_scalar_like_candidates() {
        val inputs = listOf(
            OrtJavaNamedTensorInfo("sample", OrtJavaTensorInfo(OrtJavaElementType.FLOAT, longArrayOf(1, 4, 128, 128))),
            OrtJavaNamedTensorInfo("timestep", OrtJavaTensorInfo(OrtJavaElementType.INT64, longArrayOf(1))),
            OrtJavaNamedTensorInfo("sigma", OrtJavaTensorInfo(OrtJavaElementType.FLOAT, longArrayOf(1))),
            OrtJavaNamedTensorInfo("positive_cond", OrtJavaTensorInfo(OrtJavaElementType.FLOAT, longArrayOf(1, 256, 4096))),
            OrtJavaNamedTensorInfo("negative_uncond", OrtJavaTensorInfo(OrtJavaElementType.FLOAT, longArrayOf(1, 256, 4096))),
        )

        try {
            bindDenoiserInputs(inputs)
            throw AssertionError("Expected DenoiserBindingException")
        } catch (expected: DenoiserBindingException) {
            assertEquals(OrtRuntimeFailure.INPUT_MAPPING_FAILED, expected.failure)
        }
    }

    @Test
    fun rejects_multiple_4d_float_candidates() {
        val inputs = listOf(
            OrtJavaNamedTensorInfo("sample", OrtJavaTensorInfo(OrtJavaElementType.FLOAT, longArrayOf(1, 4, 128, 128))),
            OrtJavaNamedTensorInfo("residual", OrtJavaTensorInfo(OrtJavaElementType.FLOAT, longArrayOf(1, 4, 128, 128))),
            OrtJavaNamedTensorInfo("sigma", OrtJavaTensorInfo(OrtJavaElementType.INT64, longArrayOf(1))),
            OrtJavaNamedTensorInfo("positive_cond", OrtJavaTensorInfo(OrtJavaElementType.FLOAT, longArrayOf(1, 256, 4096))),
            OrtJavaNamedTensorInfo("negative_uncond", OrtJavaTensorInfo(OrtJavaElementType.FLOAT, longArrayOf(1, 256, 4096))),
        )

        try {
            bindDenoiserInputs(inputs)
            throw AssertionError("Expected DenoiserBindingException")
        } catch (expected: DenoiserBindingException) {
            assertEquals(OrtRuntimeFailure.INPUT_MAPPING_FAILED, expected.failure)
        }
    }

    @Test
    fun rejects_opaque_conditioning_pair() {
        val inputs = listOf(
            OrtJavaNamedTensorInfo("sample", OrtJavaTensorInfo(OrtJavaElementType.FLOAT, longArrayOf(1, 4, 128, 128))),
            OrtJavaNamedTensorInfo("sigma", OrtJavaTensorInfo(OrtJavaElementType.INT64, longArrayOf(1))),
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
