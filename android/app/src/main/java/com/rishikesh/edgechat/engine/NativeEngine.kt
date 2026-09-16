package com.rishikesh.edgechat.engine

/** Raw JNI surface of libedgechat.so. Use [LlamaEngine] instead; this is not thread-safe by itself. */
class NativeEngine {
    interface Listener {
        fun onPromptProcessed(promptTokens: Int, cachedTokens: Int, droppedTurns: Int, seconds: Double)
        fun onToken(piece: String)
    }
    interface Progress { fun onProgress(fraction: Float) }
    interface Logger { fun log(level: Int, message: String) }

    external fun create(): Long
    external fun destroy(handle: Long)
    external fun setLogger(logger: Logger?)
    external fun load(
        handle: Long, modelPath: String, mmprojPath: String?, contextLength: Int, batchSize: Int, gpuLayers: Int, threads: Int,
        flashAttention: Boolean, kvCacheQ8: Boolean, disableThinking: Boolean, contextShift: Boolean, progress: Progress?,
    ): NativeModelInfo
    external fun unload(handle: Long)
    external fun clearCache(handle: Long)
    external fun cancel(handle: Long)
    external fun countTokens(handle: Long, text: String): Int
    external fun promptBudget(handle: Long, maxTokens: Int): Int
    external fun planTruncation(
        handle: Long, systemPrompt: String, roles: IntArray, texts: Array<String>,
        imageTurn: IntArray?, imageIds: Array<String>?, imageW: IntArray?, imageH: IntArray?, imageRgb: Array<ByteArray>?,
        maxTokens: Int, budgetFraction: Double,
    ): Int
    external fun respond(
        handle: Long, systemPrompt: String, roles: IntArray, texts: Array<String>,
        imageTurn: IntArray?, imageIds: Array<String>?, imageW: IntArray?, imageH: IntArray?, imageRgb: Array<ByteArray>?,
        temperature: Float, topK: Int, topP: Float, minP: Float, repeatPenalty: Float, repeatLastN: Int, maxTokens: Int, seed: Int,
        continuing: Boolean, listener: Listener,
    ): NativeStats
    external fun embed(handle: Long, texts: Array<String>, maxTokens: Int): Array<FloatArray>
    external fun saveState(handle: Long, path: String): Long
    external fun loadState(handle: Long, path: String): Boolean

    companion object {
        init { System.loadLibrary("edgechat") }
    }
}

/** Constructed from JNI; keep the constructor signature in sync with jni.cpp. */
class NativeModelInfo(
    val name: String,
    val architectureDescription: String,
    val parameterCount: Long,
    val sizeBytes: Long,
    val trainingContext: Int,
    val contextLength: Int,
    val hasVision: Boolean,
    val supportsThinkingToggle: Boolean,
    val threads: Int,
    val kvBytesPerToken: Int,
)

/** Constructed from JNI; `stopReason`: 0 eos, 1 maxTokens, 2 contextFull, 3 cancelled. */
class NativeStats(
    val promptTokens: Int,
    val cachedTokens: Int,
    val generatedTokens: Int,
    val prefillSeconds: Double,
    val decodeSeconds: Double,
    val stopReason: Int,
    val contextShifts: Int,
    val contextLength: Int,
)

class EngineException(val code: Int, message: String) : Exception(message) {
    companion object {
        const val NOT_LOADED = 0
        const val CONTEXT_FULL = 7
        const val CANCELLED = 12
    }
}
