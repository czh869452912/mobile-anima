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
}
