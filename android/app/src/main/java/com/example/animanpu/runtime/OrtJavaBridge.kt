package com.example.animanpu.runtime

import java.io.File
import java.nio.Buffer
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.LongBuffer

enum class OrtJavaElementType {
    FLOAT,
    INT64,
    UNKNOWN,
}

data class OrtJavaTensorInfo(
    val elementType: OrtJavaElementType,
    val shape: LongArray,
)

data class OrtJavaNamedTensorInfo(
    val name: String,
    val info: OrtJavaTensorInfo,
)

interface OrtJavaTensorValue : AutoCloseable {
    override fun close()
}

interface OrtJavaValue : AutoCloseable {
    fun readRawBytes(): ByteArray
    override fun close()
}

interface OrtJavaSession : AutoCloseable {
    fun inputInfos(): List<OrtJavaNamedTensorInfo>
    fun run(inputs: Map<String, OrtJavaTensorValue>): Map<String, OrtJavaValue>
    override fun close()
}

interface OrtJavaBridge {
    fun availableProviders(): Set<String>
    fun createSession(modelPath: File, config: OrtQnnConfig): OrtJavaSession
    fun tensorFromRaw(info: OrtJavaTensorInfo, rawBytes: ByteArray): OrtJavaTensorValue
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

private class ReflectionOrtJavaTensorValue(
    val inner: Any,
) : OrtJavaTensorValue {
    override fun close() {
        runCatching {
            inner.javaClass.getMethod("close").invoke(inner)
        }
    }
}

private class ReflectionOrtJavaValue(
    private val inner: Any,
) : OrtJavaValue {
    override fun readRawBytes(): ByteArray {
        val optional = inner.javaClass.getMethod("getBufferRef").invoke(inner)
        val buffer = optional.javaClass.getMethod("get").invoke(optional) as Buffer
        return when (buffer) {
            is ByteBuffer -> {
                val dup = buffer.duplicate().order(ByteOrder.nativeOrder())
                val out = ByteArray(dup.remaining())
                dup.get(out)
                out
            }
            is FloatBuffer -> {
                val dup = buffer.duplicate()
                val out = ByteBuffer.allocate(dup.remaining() * 4).order(ByteOrder.nativeOrder())
                while (dup.hasRemaining()) {
                    out.putFloat(dup.get())
                }
                out.array()
            }
            is LongBuffer -> {
                val dup = buffer.duplicate()
                val out = ByteBuffer.allocate(dup.remaining() * 8).order(ByteOrder.nativeOrder())
                while (dup.hasRemaining()) {
                    out.putLong(dup.get())
                }
                out.array()
            }
            else -> throw IllegalStateException("Unsupported output buffer: ${buffer.javaClass.name}")
        }
    }

    override fun close() {
        runCatching {
            inner.javaClass.getMethod("close").invoke(inner)
        }
    }
}

private fun toOrtJavaElementType(valueInfo: Any): OrtJavaElementType {
    val typeField = valueInfo.javaClass.getField("type")
    return when (typeField.get(valueInfo).toString()) {
        "FLOAT" -> OrtJavaElementType.FLOAT
        "INT64" -> OrtJavaElementType.INT64
        else -> OrtJavaElementType.UNKNOWN
    }
}

private fun extractNamedTensorInfo(nodeInfo: Any): OrtJavaNamedTensorInfo? {
    val name = nodeInfo.javaClass.getMethod("getName").invoke(nodeInfo)?.toString() ?: return null
    val valueInfo = nodeInfo.javaClass.getMethod("getInfo").invoke(nodeInfo) ?: return null
    if (valueInfo.javaClass.simpleName != "TensorInfo") {
        return null
    }
    val shape = valueInfo.javaClass.getMethod("getShape").invoke(valueInfo) as LongArray
    return OrtJavaNamedTensorInfo(
        name = name,
        info = OrtJavaTensorInfo(
            elementType = toOrtJavaElementType(valueInfo),
            shape = shape,
        ),
    )
}

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

                override fun inputInfos(): List<OrtJavaNamedTensorInfo> {
                    val inputMap = session.javaClass.getMethod("getInputInfo").invoke(session) as Map<*, *>
                    return inputMap.values.mapNotNull { node -> extractNamedTensorInfo(node ?: return@mapNotNull null) }
                }

                override fun run(inputs: Map<String, OrtJavaTensorValue>): Map<String, OrtJavaValue> {
                    val ortInputs = inputs.mapValues { (_, value) -> (value as ReflectionOrtJavaTensorValue).inner }
                    val result = session.javaClass.getMethod("run", Map::class.java).invoke(session, ortInputs)
                    val iterator = result.javaClass.getMethod("iterator").invoke(result) as Iterator<*>
                    val outputs = linkedMapOf<String, OrtJavaValue>()
                    while (iterator.hasNext()) {
                        val entry = iterator.next() ?: continue
                        val key = entry.javaClass.getMethod("getKey").invoke(entry).toString()
                        val value = entry.javaClass.getMethod("getValue").invoke(entry)
                        outputs[key] = ReflectionOrtJavaValue(value)
                    }
                    return outputs
                }

                override fun close() {
                    if (closed) {
                        return
                    }
                    runCatching { session.javaClass.getMethod("close").invoke(session) }
                    runCatching { sessionOptionsClass.getMethod("close").invoke(sessionOptions) }
                    closed = true
                }
            }
        } catch (error: RuntimeException) {
            throw error
        } catch (error: Throwable) {
            throw OrtJavaSessionCreateException(error)
        }
    }

    override fun tensorFromRaw(info: OrtJavaTensorInfo, rawBytes: ByteArray): OrtJavaTensorValue {
        try {
            val environmentClass = Class.forName("ai.onnxruntime.OrtEnvironment")
            val onnxTensorClass = Class.forName("ai.onnxruntime.OnnxTensor")
            val environment = environmentClass.getMethod("getEnvironment").invoke(null)
            val nativeBuffer = ByteBuffer.wrap(rawBytes).order(ByteOrder.nativeOrder())
            val tensor = when (info.elementType) {
                OrtJavaElementType.FLOAT -> {
                    val floatBuffer = nativeBuffer.asFloatBuffer()
                    onnxTensorClass.getMethod(
                        "createTensor",
                        environmentClass,
                        FloatBuffer::class.java,
                        LongArray::class.java,
                    ).invoke(null, environment, floatBuffer, info.shape)
                }
                OrtJavaElementType.INT64 -> {
                    val longBuffer = nativeBuffer.asLongBuffer()
                    onnxTensorClass.getMethod(
                        "createTensor",
                        environmentClass,
                        LongBuffer::class.java,
                        LongArray::class.java,
                    ).invoke(null, environment, longBuffer, info.shape)
                }
                else -> throw IllegalArgumentException("Unsupported tensor type: ${info.elementType}")
            }
            return ReflectionOrtJavaTensorValue(tensor)
        } catch (error: RuntimeException) {
            throw error
        } catch (error: Throwable) {
            throw OrtJavaSessionCreateException(error)
        }
    }
}
