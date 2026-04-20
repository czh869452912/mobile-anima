package com.example.animanpu.runtime

import java.io.File

class ArtifactManager(
    private val filesDir: File,
) {
    fun denoiserContextOnnx(): File = File(filesDir, "runtime/denoiser_ctx.onnx")
    fun denoiserContextBin(): File = File(filesDir, "runtime/denoiser_qnn.bin")
    fun profilingCsv(): File = File(filesDir, "runtime/denoiser_profile.csv")
}
