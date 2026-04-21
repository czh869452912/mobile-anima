package com.example.animanpu.runtime

import java.io.File

class ArtifactManager(
    private val filesDir: File,
) {
    private val runtimeDir = File(filesDir, "runtime")
    private val inputDir = File(runtimeDir, "inputs")
    private val outputDir = File(runtimeDir, "outputs")

    fun denoiserContextOnnx(): File = File(filesDir, "runtime/denoiser_ctx.onnx")
    fun denoiserContextBin(): File = File(filesDir, "runtime/denoiser_qnn.bin")
    fun profilingCsv(): File = File(filesDir, "runtime/denoiser_profile.csv")

    fun latentInput(): File = File(inputDir, "latent.raw")

    fun timestepInput(): File = File(inputDir, "timestep.raw")

    fun condInput(): File = File(inputDir, "cond.raw")

    fun uncondInput(): File = File(inputDir, "uncond.raw")

    fun outputTensor(): File = File(outputDir, "denoiser_output.raw")

    fun ensureRuntimeDirectories() {
        inputDir.mkdirs()
        outputDir.mkdirs()
    }
}
