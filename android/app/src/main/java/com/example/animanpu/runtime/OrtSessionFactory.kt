package com.example.animanpu.runtime

import java.io.File

interface OrtSessionHandle {
    fun run(outputPath: File): Boolean
}

interface OrtSessionFactory {
    fun providerVisible(): Boolean
    fun createDenoiserSession(modelPath: File, config: OrtQnnConfig): OrtSessionHandle
}

class ReflectionOrtSessionFactory : OrtSessionFactory {
    override fun providerVisible(): Boolean {
        return runCatching {
            val environmentClass = Class.forName("ai.onnxruntime.OrtEnvironment")
            val environment = environmentClass.getMethod("getEnvironment").invoke(null)
            val providers = environmentClass.getMethod("getAvailableProviders").invoke(environment) as List<*>
            providers.any { it?.toString() == "QNNExecutionProvider" }
        }.getOrDefault(false)
    }

    override fun createDenoiserSession(modelPath: File, config: OrtQnnConfig): OrtSessionHandle {
        return object : OrtSessionHandle {
            override fun run(outputPath: File): Boolean {
                outputPath.parentFile?.mkdirs()
                return false
            }
        }
    }
}
