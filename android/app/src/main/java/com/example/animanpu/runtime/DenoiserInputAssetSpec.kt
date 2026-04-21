package com.example.animanpu.runtime

data class DenoiserInputAssetSpec(
    val latentFileName: String = "latent.raw",
    val timestepFileName: String = "timestep.raw",
    val condFileName: String = "cond.raw",
    val uncondFileName: String = "uncond.raw",
)
