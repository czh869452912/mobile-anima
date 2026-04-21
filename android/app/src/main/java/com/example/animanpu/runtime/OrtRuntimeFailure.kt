package com.example.animanpu.runtime

enum class OrtRuntimeFailure(
    val code: String,
) {
    MISSING_RUNTIME_ARTIFACT("missing_runtime_artifact"),
    ORT_API_UNAVAILABLE("ort_api_unavailable"),
    QNN_PROVIDER_UNAVAILABLE("qnn_provider_unavailable"),
    SESSION_CREATE_FAILED("session_create_failed"),
    EXECUTE_NOT_IMPLEMENTED("execute_not_implemented"),
    EXECUTE_FAILED("execute_failed"),
    INPUT_METADATA_UNAVAILABLE("input_metadata_unavailable"),
    DYNAMIC_SHAPE_UNSUPPORTED("dynamic_shape_unsupported"),
    UNSUPPORTED_TENSOR_TYPE("unsupported_tensor_type"),
    INPUT_MAPPING_FAILED("input_mapping_failed"),
    INPUT_FILE_SIZE_MISMATCH("input_file_size_mismatch"),
    SESSION_EXECUTE_FAILED("session_execute_failed"),
    OUTPUT_WRITE_FAILED("output_write_failed"),
}
