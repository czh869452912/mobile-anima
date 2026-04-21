package com.example.animanpu.runtime

class DenoiserBindingException(
    val failure: OrtRuntimeFailure,
) : RuntimeException(failure.code)

data class DenoiserInputBinding(
    val latent: OrtJavaNamedTensorInfo,
    val timestep: OrtJavaNamedTensorInfo,
    val cond: OrtJavaNamedTensorInfo,
    val uncond: OrtJavaNamedTensorInfo,
)

fun expectedByteCount(info: OrtJavaTensorInfo): Long {
    val elementWidth = when (info.elementType) {
        OrtJavaElementType.FLOAT -> 4L
        OrtJavaElementType.INT64 -> 8L
        else -> throw DenoiserBindingException(OrtRuntimeFailure.UNSUPPORTED_TENSOR_TYPE)
    }
    if (info.shape.any { it <= 0L }) {
        throw DenoiserBindingException(OrtRuntimeFailure.DYNAMIC_SHAPE_UNSUPPORTED)
    }
    return info.shape.fold(1L) { acc, dim -> acc * dim } * elementWidth
}

fun bindDenoiserInputs(inputs: List<OrtJavaNamedTensorInfo>): DenoiserInputBinding {
    val latent = inputs.singleOrNull { it.name.contains("latent", ignoreCase = true) }
        ?: throw DenoiserBindingException(OrtRuntimeFailure.INPUT_MAPPING_FAILED)
    val timestep = inputs.singleOrNull { it.name.contains("time", ignoreCase = true) }
        ?: throw DenoiserBindingException(OrtRuntimeFailure.INPUT_MAPPING_FAILED)
    val cond = inputs.singleOrNull { it.name.equals("cond", ignoreCase = true) }
        ?: throw DenoiserBindingException(OrtRuntimeFailure.INPUT_MAPPING_FAILED)
    val uncond = inputs.singleOrNull { it.name.equals("uncond", ignoreCase = true) }
        ?: throw DenoiserBindingException(OrtRuntimeFailure.INPUT_MAPPING_FAILED)
    return DenoiserInputBinding(
        latent = latent,
        timestep = timestep,
        cond = cond,
        uncond = uncond,
    )
}
