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

private enum class ConditioningRole {
    COND,
    UNCOND,
}

private fun isStaticShape(info: OrtJavaTensorInfo): Boolean = info.shape.all { it > 0L }

private fun elementCount(shape: LongArray): Long = shape.fold(1L) { acc, dim -> acc * dim }

private fun isScalarLikeCandidate(info: OrtJavaTensorInfo): Boolean =
    isStaticShape(info) &&
        elementCount(info.shape) == 1L &&
        info.elementType in setOf(OrtJavaElementType.FLOAT, OrtJavaElementType.INT64)

private fun isLatentCandidate(info: OrtJavaTensorInfo): Boolean =
    isStaticShape(info) &&
        info.elementType == OrtJavaElementType.FLOAT &&
        info.shape.size == 4

private fun conditioningRole(name: String): ConditioningRole? {
    val lower = name.lowercase()
    return when {
        lower.contains("uncond") || lower.contains("negative") -> ConditioningRole.UNCOND
        (lower.contains("cond") && !lower.contains("uncond")) || lower.contains("positive") -> ConditioningRole.COND
        else -> null
    }
}

fun expectedByteCount(info: OrtJavaTensorInfo): Long {
    val elementWidth = when (info.elementType) {
        OrtJavaElementType.FLOAT -> 4L
        OrtJavaElementType.INT64 -> 8L
        else -> throw DenoiserBindingException(OrtRuntimeFailure.UNSUPPORTED_TENSOR_TYPE)
    }
    if (!isStaticShape(info)) {
        throw DenoiserBindingException(OrtRuntimeFailure.DYNAMIC_SHAPE_UNSUPPORTED)
    }
    return elementCount(info.shape) * elementWidth
}

fun bindDenoiserInputs(inputs: List<OrtJavaNamedTensorInfo>): DenoiserInputBinding {
    inputs.forEach { input -> expectedByteCount(input.info) }

    val timestepCandidates = inputs.filter { input -> isScalarLikeCandidate(input.info) }
    val timestep = timestepCandidates.singleOrNull()
        ?: throw DenoiserBindingException(OrtRuntimeFailure.INPUT_MAPPING_FAILED)

    val latentCandidates = inputs.filter { input -> input != timestep && isLatentCandidate(input.info) }
    val latent = latentCandidates.singleOrNull()
        ?: throw DenoiserBindingException(OrtRuntimeFailure.INPUT_MAPPING_FAILED)

    val remaining = inputs.filter { input -> input != timestep && input != latent }
    if (remaining.size != 2) {
        throw DenoiserBindingException(OrtRuntimeFailure.INPUT_MAPPING_FAILED)
    }

    val first = remaining[0]
    val second = remaining[1]
    if (first.info.elementType != second.info.elementType || !first.info.shape.contentEquals(second.info.shape)) {
        throw DenoiserBindingException(OrtRuntimeFailure.INPUT_MAPPING_FAILED)
    }

    val cond = remaining.singleOrNull { input -> conditioningRole(input.name) == ConditioningRole.COND }
        ?: throw DenoiserBindingException(OrtRuntimeFailure.INPUT_MAPPING_FAILED)
    val uncond = remaining.singleOrNull { input -> conditioningRole(input.name) == ConditioningRole.UNCOND }
        ?: throw DenoiserBindingException(OrtRuntimeFailure.INPUT_MAPPING_FAILED)

    return DenoiserInputBinding(
        latent = latent,
        timestep = timestep,
        cond = cond,
        uncond = uncond,
    )
}
