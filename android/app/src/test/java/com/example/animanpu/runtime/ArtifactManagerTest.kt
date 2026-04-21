package com.example.animanpu.runtime

import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.io.path.createTempDirectory

class ArtifactManagerTest {
    @Test
    fun resolves_denoiser_runtime_inputs_and_outputs_under_runtime_directory() {
        val root = createTempDirectory(prefix = "anima-runtime-").toFile()
        val manager = ArtifactManager(root)

        assertEquals(File(root, "runtime/denoiser_ctx.onnx"), manager.denoiserContextOnnx())
        assertEquals(File(root, "runtime/denoiser_qnn.bin"), manager.denoiserContextBin())
        assertEquals(File(root, "runtime/inputs/latent.raw"), manager.latentInput())
        assertEquals(File(root, "runtime/inputs/timestep.raw"), manager.timestepInput())
        assertEquals(File(root, "runtime/inputs/cond.raw"), manager.condInput())
        assertEquals(File(root, "runtime/inputs/uncond.raw"), manager.uncondInput())
        assertEquals(File(root, "runtime/outputs/denoiser_output.raw"), manager.outputTensor())
        assertEquals(File(root, "runtime/denoiser_profile.csv"), manager.profilingCsv())
    }

    @Test
    fun ensures_runtime_parent_directories_exist() {
        val root = createTempDirectory(prefix = "anima-runtime-").toFile()
        val manager = ArtifactManager(root)

        manager.ensureRuntimeDirectories()

        assertTrue(File(root, "runtime/inputs").isDirectory)
        assertTrue(File(root, "runtime/outputs").isDirectory)
    }
}
