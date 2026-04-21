package com.example.animanpu.runtime

import java.io.File

interface OrtJavaSession : AutoCloseable {
    override fun close()
}

interface OrtJavaBridge {
    fun availableProviders(): Set<String>
    fun createSession(modelPath: File, config: OrtQnnConfig): OrtJavaSession
}

class OrtJavaApiUnavailableException(
    cause: Throwable,
) : RuntimeException(cause)

class OrtJavaSessionCreateException(
    cause: Throwable,
) : RuntimeException(cause)

fun normalizeProviderNames(rawProviders: Iterable<*>): Set<String> =
    rawProviders.mapNotNull { provider ->
        if (provider == null) {
            null
        } else {
            runCatching {
                provider.javaClass.getMethod("getName").invoke(provider)?.toString()
            }.getOrNull() ?: provider.toString()
        }
    }.toSet()

class ReflectionOrtJavaBridge : OrtJavaBridge {
    override fun availableProviders(): Set<String> {
        return try {
            val environmentClass = Class.forName("ai.onnxruntime.OrtEnvironment")
            val providers = environmentClass.getMethod("getAvailableProviders").invoke(null) as Iterable<*>
            normalizeProviderNames(providers)
        } catch (error: Throwable) {
            throw OrtJavaApiUnavailableException(error)
        }
    }

    override fun createSession(modelPath: File, config: OrtQnnConfig): OrtJavaSession {
        try {
            val environmentClass = Class.forName("ai.onnxruntime.OrtEnvironment")
            val sessionOptionsClass = Class.forName("ai.onnxruntime.OrtSession\$SessionOptions")
            val environment = environmentClass.getMethod("getEnvironment").invoke(null)
            val sessionOptions = sessionOptionsClass.getConstructor().newInstance()

            val qnnOptions = config.providerOptions().filterKeys { it != "session.disable_cpu_ep_fallback" }
            sessionOptionsClass.getMethod("addQnn", Map::class.java).invoke(sessionOptions, qnnOptions)
            runCatching {
                sessionOptionsClass
                    .getMethod("addConfigEntry", String::class.java, String::class.java)
                    .invoke(
                        sessionOptions,
                        "session.disable_cpu_ep_fallback",
                        if (config.disableCpuFallback) "1" else "0",
                    )
            }
            runCatching {
                sessionOptionsClass
                    .getMethod("enableProfiling", String::class.java)
                    .invoke(sessionOptions, config.profilingPath)
            }

            val session = environmentClass
                .getMethod("createSession", String::class.java, sessionOptionsClass)
                .invoke(environment, modelPath.path, sessionOptions)

            return object : OrtJavaSession {
                private var closed = false

                override fun close() {
                    if (closed) {
                        return
                    }
                    runCatching {
                        session.javaClass.getMethod("close").invoke(session)
                    }
                    runCatching {
                        sessionOptionsClass.getMethod("close").invoke(sessionOptions)
                    }
                    closed = true
                }
            }
        } catch (error: RuntimeException) {
            throw error
        } catch (error: Throwable) {
            throw OrtJavaSessionCreateException(error)
        }
    }
}
