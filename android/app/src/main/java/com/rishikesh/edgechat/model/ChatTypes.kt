package com.rishikesh.edgechat.model

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import java.util.UUID

// Persistent chat model. JSON layout matches the iOS app (Documents/Conversations/<id>.json).

@Serializable
enum class Role { @SerialName("system") SYSTEM, @SerialName("user") USER, @SerialName("assistant") ASSISTANT }

@Serializable
enum class AttachmentKind { @SerialName("image") IMAGE, @SerialName("document") DOCUMENT }

@Serializable
data class Attachment(
    val id: String = UUID.randomUUID().toString().uppercase(),
    val kind: AttachmentKind,
    /** Original display name (e.g. "report.pdf", "IMG_0421.jpeg"). */
    val fileName: String,
    /** File stored inside the conversation's attachment folder. */
    val storedFileName: String,
    val byteCount: Long,
    /** Text extracted from a document, or OCR for an image (used for text-only models). */
    val extractedText: String? = null,
    val textTruncated: Boolean = false,
    val pageCount: Int? = null,
    val imageWidth: Int? = null,
    val imageHeight: Int? = null,
    /** How the attachment was fed to the model on send ("vision", "ocr", "text"). */
    val ingestion: String? = null,
)

@Serializable
enum class StopReason { @SerialName("eos") EOS, @SerialName("maxTokens") MAX_TOKENS, @SerialName("contextFull") CONTEXT_FULL, @SerialName("cancelled") CANCELLED }

@Serializable
data class GenerationStats(
    /** Total prompt tokens (including image tokens). */
    val promptTokens: Int,
    /** Prompt cells reused from the KV cache (no recompute). */
    val cachedTokens: Int,
    val generatedTokens: Int,
    val prefillSeconds: Double,
    val decodeSeconds: Double,
    val stopReason: StopReason,
    val modelID: String? = null,
    /** Mid-generation context shifts (oldest tokens discarded to keep going). */
    val contextShifts: Int? = null,
    /** Context window the reply was generated with. */
    val contextLength: Int? = null,
) {
    val tokensPerSecond: Double get() = if (decodeSeconds > 0) generatedTokens / decodeSeconds else 0.0
}

@Serializable
data class Message(
    val id: String = UUID.randomUUID().toString().uppercase(),
    val role: Role,
    val content: String,
    /** Chain-of-thought emitted inside <think>…</think> (Qwen3 etc.), shown collapsed. */
    val reasoning: String? = null,
    val attachments: List<Attachment> = emptyList(),
    val createdAt: String = Iso.now(),
    val stats: GenerationStats? = null,
    val error: String? = null,
    /** Snippets retrieved from earlier in the conversation and shown to the model with this user turn. */
    val recalledContext: String? = null,
)

@Serializable
data class Conversation(
    val id: String = UUID.randomUUID().toString().uppercase(),
    val title: String = "New chat",
    val messages: List<Message> = emptyList(),
    val createdAt: String = Iso.now(),
    val updatedAt: String = Iso.now(),
    /** Per-conversation override; null uses the global default. */
    val systemPrompt: String? = null,
    val modelID: String? = null,
    /** Model-written memory of messages that no longer fit the context window. */
    val summary: String? = null,
    /** Number of leading messages folded into `summary` (they are not sent to the model any more). */
    val summaryCoversMessages: Int = 0,
)

// Engine configuration

@Serializable
data class SamplingConfig(
    val temperature: Float = 0.7f,
    val topK: Int = 40,
    val topP: Float = 0.9f,
    val minP: Float = 0.05f,
    val repeatPenalty: Float = 1.1f,
    val repeatLastN: Int = 64,
    /** Maximum tokens per reply; 0 = no limit (the reply ends when the model stops or the window is full). */
    val maxTokens: Int = 0,
    /** -1 (0xFFFFFFFF) = random seed each call. */
    val seed: Int = -1,
) {
    val effectiveMaxTokens: Int get() = if (maxTokens > 0) maxTokens else Int.MAX_VALUE
}

@Serializable
data class EngineConfig(
    val contextLength: Int = 4096,
    val batchSize: Int = 512,
    /** GPU layers (no GPU backend in this build; kept for parity with iOS). */
    val gpuLayers: Int = 0,
    /** 0 = auto. */
    val threads: Int = 0,
    val flashAttention: Boolean = true,
    /** Quantize the KV cache to Q8_0 (halves cache memory, enables longer contexts). */
    val kvCacheQ8: Boolean = false,
    /** For hybrid reasoning models (Qwen3): suppress <think> blocks for faster replies. */
    val disableThinking: Boolean = true,
    /** When a reply would overflow the context, discard the oldest half of the history and keep generating. */
    val contextShift: Boolean = true,
) {
    companion object {
        /** Sensible context length for a device: 8k on 8 GB+ phones, 4k on 6 GB, 2k below. */
        fun defaultContextLength(physicalMemoryBytes: Long): Int {
            val gb = physicalMemoryBytes / 1_073_741_824.0
            return when { gb >= 7.5 -> 8192; gb >= 5.5 -> 4096; else -> 2048 }
        }
    }
}

/** ISO-8601 UTC timestamps, second precision, same as the iOS JSON. */
object Iso {
    private val fmt = java.time.format.DateTimeFormatter.ISO_INSTANT
    fun now(): String = fmt.format(java.time.Instant.now().truncatedTo(java.time.temporal.ChronoUnit.SECONDS))
    fun parse(s: String): java.time.Instant = runCatching { java.time.Instant.parse(s) }.getOrDefault(java.time.Instant.EPOCH)
}
