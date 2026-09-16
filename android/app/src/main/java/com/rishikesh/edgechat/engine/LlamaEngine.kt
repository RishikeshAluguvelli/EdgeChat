package com.rishikesh.edgechat.engine

import com.rishikesh.edgechat.model.EngineConfig
import com.rishikesh.edgechat.model.GenerationStats
import com.rishikesh.edgechat.model.Role
import com.rishikesh.edgechat.model.SamplingConfig
import com.rishikesh.edgechat.model.StopReason
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.ExecutorCoroutineDispatcher
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

/** An RGB888 bitmap ready for the vision encoder. */
class ImageInput(val id: String, val width: Int, val height: Int, val rgb: ByteArray) {
    init { require(rgb.size == width * height * 3) { "rgb must be width*height*3 bytes" } }
}

/** One turn as fed to the engine. `text` already contains any document text the app injected. */
data class ChatTurn(val role: Role, val text: String, val images: List<ImageInput> = emptyList())

data class LoadedModelInfo(
    val modelPath: String,
    val mmprojPath: String?,
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
) {
    fun kvCacheBytes(contextLength: Int): Long = kvBytesPerToken.toLong() * contextLength
}

sealed class GenerationEvent {
    data class PromptProcessed(val promptTokens: Int, val cachedTokens: Int, val droppedTurns: Int, val seconds: Double) : GenerationEvent()
    data class Token(val text: String) : GenerationEvent()
    data class Finished(val stats: GenerationStats) : GenerationEvent()
}

/**
 * Kotlin face of the native engine. All native calls run on one dedicated thread ([dispatcher]); only one
 * generation runs at a time and it is cancelled by cancelling the collecting coroutine.
 */
class LlamaEngine {
    private val native = NativeEngine()
    private val handle = native.create()
    private val executor = Executors.newSingleThreadExecutor { r -> Thread(r, "edgechat-llama").apply { priority = Thread.MAX_PRIORITY } }
    private val dispatcher: ExecutorCoroutineDispatcher = executor.asCoroutineDispatcher()
    private val generating = AtomicBoolean(false)

    @Volatile var info: LoadedModelInfo? = null
        private set
    val isLoaded get() = info != null
    val isGenerating get() = generating.get()

    fun setLogger(logger: NativeEngine.Logger?) = native.setLogger(logger)

    suspend fun load(modelPath: String, mmprojPath: String?, config: EngineConfig, progress: ((Float) -> Unit)? = null): LoadedModelInfo =
        withContext(dispatcher) {
            val cb = progress?.let { p -> object : NativeEngine.Progress { override fun onProgress(fraction: Float) = p(fraction) } }
            val n = native.load(handle, modelPath, mmprojPath, config.contextLength, config.batchSize, config.gpuLayers, config.threads,
                config.flashAttention, config.kvCacheQ8, config.disableThinking, config.contextShift, cb)
            LoadedModelInfo(modelPath, mmprojPath, n.name, n.architectureDescription, n.parameterCount, n.sizeBytes, n.trainingContext,
                n.contextLength, n.hasVision, n.supportsThinkingToggle, n.threads, n.kvBytesPerToken).also { info = it }
        }

    suspend fun unload() = withContext(dispatcher) { native.unload(handle); info = null }
    suspend fun clearCache() = withContext(dispatcher) { native.clearCache(handle) }
    suspend fun countTokens(text: String): Int = withContext(dispatcher) { native.countTokens(handle, text) }
    suspend fun promptBudget(sampling: SamplingConfig): Int = withContext(dispatcher) { native.promptBudget(handle, sampling.maxTokens) }

    /** How many leading turns `respond` would drop to fit the budget (0 = everything fits). */
    suspend fun planTruncation(systemPrompt: String?, turns: List<ChatTurn>, sampling: SamplingConfig, budgetFraction: Double = 1.0): Int =
        withContext(dispatcher) {
            val t = Turns(turns)
            native.planTruncation(handle, systemPrompt ?: "", t.roles, t.texts, t.imageTurn, t.imageIds, t.imageW, t.imageH, t.imageRgb,
                sampling.maxTokens, budgetFraction)
        }

    /** Streams the reply. Cancelling the collector stops generation promptly. */
    fun respond(systemPrompt: String?, turns: List<ChatTurn>, sampling: SamplingConfig, continuing: Boolean = false): Flow<GenerationEvent> =
        callbackFlow {
            if (!generating.compareAndSet(false, true)) { close(EngineException(11, "The engine is busy.")); return@callbackFlow }
            val job = launch(dispatcher) {
                try {
                    val t = Turns(turns)
                    val listener = object : NativeEngine.Listener {
                        override fun onPromptProcessed(promptTokens: Int, cachedTokens: Int, droppedTurns: Int, seconds: Double) {
                            trySend(GenerationEvent.PromptProcessed(promptTokens, cachedTokens, droppedTurns, seconds))
                        }
                        override fun onToken(piece: String) { trySend(GenerationEvent.Token(piece)) }
                    }
                    val st = native.respond(handle, systemPrompt ?: "", t.roles, t.texts, t.imageTurn, t.imageIds, t.imageW, t.imageH, t.imageRgb,
                        sampling.temperature, sampling.topK, sampling.topP, sampling.minP, sampling.repeatPenalty, sampling.repeatLastN,
                        sampling.maxTokens, sampling.seed, continuing, listener)
                    trySend(GenerationEvent.Finished(GenerationStats(
                        promptTokens = st.promptTokens, cachedTokens = st.cachedTokens, generatedTokens = st.generatedTokens,
                        prefillSeconds = st.prefillSeconds, decodeSeconds = st.decodeSeconds,
                        stopReason = StopReason.entries[st.stopReason.coerceIn(0, 3)],
                        contextShifts = st.contextShifts.takeIf { it > 0 }, contextLength = st.contextLength)))
                    close()
                } catch (e: Throwable) {
                    close(e)
                } finally {
                    generating.set(false)
                }
            }
            awaitClose {
                // Collector went away (cancelled or finished): stop the native loop if it is still running.
                if (job.isActive) native.cancel(handle)
            }
        }

    /** Uses the loaded model to fold `turns` into a running summary (the "memory" of dropped history). */
    suspend fun summarize(previousSummary: String?, turns: List<ChatTurn>, maxWords: Int = 220): String {
        val transcript = StringBuilder()
        for (t in turns) {
            val images = if (t.images.isEmpty()) "" else " [${t.images.size} image(s) attached]"
            transcript.append(if (t.role == Role.USER) "User" else "Assistant").append(images).append(": ").append(t.text).append("\n\n")
        }
        var prompt = "You maintain the memory of a conversation between a user and an assistant so it can continue after older messages are removed.\n\n"
        if (!previousSummary.isNullOrEmpty()) prompt += "Existing memory:\n$previousSummary\n\n"
        prompt += "New messages to fold into the memory:\n$transcript"
        prompt += """Write the updated memory in at most $maxWords words, as plain bullets under these headings (skip a heading with nothing to say):
About the user: name, role, preferences, goals they stated.
Topics so far: each topic in order, with the concrete facts, numbers, names and conclusions that were given.
Decisions and recommendations: what was agreed or advised.
Open threads: what the user asked most recently, and anything left unanswered.
Keep exact names, numbers, identifiers and quoted wording. Do not add information that is not in the messages. Output only the memory.
"""
        val sampling = SamplingConfig(temperature = 0.3f, maxTokens = maxWords * 2 + 64)
        val splitter = ThinkTagSplitter()
        val out = StringBuilder()
        respond(null, listOf(ChatTurn(Role.USER, prompt)), sampling).collect { ev ->
            if (ev is GenerationEvent.Token) for (p in splitter.feed(ev.text)) if (p is ThinkTagSplitter.Part.Text) out.append(p.text)
        }
        for (p in splitter.flush()) if (p is ThinkTagSplitter.Part.Text) out.append(p.text)
        val trimmed = out.toString().trim()
        if (trimmed.isEmpty()) throw EngineException(14, "The model produced an empty summary.")
        return trimmed.take(maxWords * 8)
    }

    /** Mean-pooled, L2-normalized embeddings from the loaded model (temporary embedding context). */
    suspend fun embed(texts: List<String>, maxTokens: Int = 384): List<FloatArray> = withContext(dispatcher) {
        native.embed(handle, texts.toTypedArray(), maxTokens).toList()
    }

    suspend fun saveState(file: File): Long = withContext(dispatcher) {
        file.parentFile?.mkdirs()
        native.saveState(handle, file.absolutePath)
    }

    suspend fun loadState(file: File): Boolean = withContext(dispatcher) { native.loadState(handle, file.absolutePath) }

    fun close() { executor.execute { native.unload(handle); native.destroy(handle) }; executor.shutdown() }

    /** Flattened turn arrays for JNI. */
    private class Turns(turns: List<ChatTurn>) {
        val roles = IntArray(turns.size) { when (turns[it].role) { Role.SYSTEM -> 0; Role.USER -> 1; Role.ASSISTANT -> 2 } }
        val texts = Array(turns.size) { turns[it].text }
        private val images = turns.flatMapIndexed { i, t -> t.images.map { i to it } }
        val imageTurn = if (images.isEmpty()) null else IntArray(images.size) { images[it].first }
        val imageIds = if (images.isEmpty()) null else Array(images.size) { images[it].second.id }
        val imageW = if (images.isEmpty()) null else IntArray(images.size) { images[it].second.width }
        val imageH = if (images.isEmpty()) null else IntArray(images.size) { images[it].second.height }
        val imageRgb = if (images.isEmpty()) null else Array(images.size) { images[it].second.rgb }
    }

    companion object {
        /** Cells held back for the reply: the full `maxTokens` up to a quarter of the context (at least 128). */
        fun replyReserve(contextLength: Int, sampling: SamplingConfig): Int =
            minOf(sampling.effectiveMaxTokens, maxOf(128, contextLength / 4))
    }
}

/** Splits a streamed reply into visible text and `<think>…</think>` reasoning, tolerating tags split across tokens. */
class ThinkTagSplitter {
    sealed class Part {
        data class Reasoning(val text: String) : Part()
        data class Text(val text: String) : Part()
    }
    private var buffer = ""
    private var inside = false

    fun feed(chunk: String): List<Part> {
        buffer += chunk
        val parts = mutableListOf<Part>()
        while (true) {
            val tag = if (inside) CLOSE else OPEN
            val idx = buffer.indexOf(tag)
            if (idx >= 0) {
                val before = buffer.substring(0, idx)
                if (before.isNotEmpty()) parts += if (inside) Part.Reasoning(before) else Part.Text(before)
                buffer = buffer.substring(idx + tag.length)
                inside = !inside
                continue
            }
            val hold = longestSuffixThatPrefixes(buffer, tag)
            val emit = buffer.length - hold
            if (emit > 0) {
                val s = buffer.substring(0, emit)
                parts += if (inside) Part.Reasoning(s) else Part.Text(s)
                buffer = buffer.substring(emit)
            }
            break
        }
        return parts
    }

    fun flush(): List<Part> {
        if (buffer.isEmpty()) return emptyList()
        val out = listOf(if (inside) Part.Reasoning(buffer) else Part.Text(buffer))
        buffer = ""
        return out
    }

    private fun longestSuffixThatPrefixes(s: String, tag: String): Int {
        val max = minOf(s.length, tag.length - 1)
        for (len in max downTo 1) if (tag.startsWith(s.takeLast(len))) return len
        return 0
    }

    companion object { private const val OPEN = "<think>"; private const val CLOSE = "</think>" }
}
