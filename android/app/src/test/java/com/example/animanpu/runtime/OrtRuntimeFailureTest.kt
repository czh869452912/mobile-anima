package com.example.animanpu.runtime

import org.junit.Assert.assertEquals
import org.junit.Test

class OrtRuntimeFailureTest {
    @Test
    fun exposes_execute_stage_failure_codes() {
        assertEquals("input_metadata_unavailable", OrtRuntimeFailure.INPUT_METADATA_UNAVAILABLE.code)
        assertEquals("dynamic_shape_unsupported", OrtRuntimeFailure.DYNAMIC_SHAPE_UNSUPPORTED.code)
        assertEquals("unsupported_tensor_type", OrtRuntimeFailure.UNSUPPORTED_TENSOR_TYPE.code)
        assertEquals("input_mapping_failed", OrtRuntimeFailure.INPUT_MAPPING_FAILED.code)
        assertEquals("input_file_size_mismatch", OrtRuntimeFailure.INPUT_FILE_SIZE_MISMATCH.code)
        assertEquals("session_execute_failed", OrtRuntimeFailure.SESSION_EXECUTE_FAILED.code)
        assertEquals("output_write_failed", OrtRuntimeFailure.OUTPUT_WRITE_FAILED.code)
    }
}
