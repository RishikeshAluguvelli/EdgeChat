package com.rishikesh.edgechat.model

import android.content.Context
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

/** User-tunable settings, persisted as JSON in SharedPreferences. Unknown/missing keys fall back to defaults. */
@Serializable
data class AppSettings(
    val systemPrompt: String = DEFAULT_SYSTEM_PROMPT,
    val sampling: SamplingConfig = SamplingConfig(),
    val engine: EngineConfig = EngineConfig(),
    /** Catalog id or "file:<name>" for imported models. */
    val activeModelID: String? = null,
    /** Also run OCR so text-only models can read images. */
    val ocrForImages: Boolean = true,
    /** ~576 image tokens on Qwen3-VL; 1024 px costs ~1000. */
    val maxImageDimension: Int = 768,
    val showStats: Boolean = true,
    val autoLoadLastModel: Boolean = true,
    /** Long-conversation memory. */
    val summarizeDroppedTurns: Boolean = true,
    val memoryRetrieval: Boolean = true,
    val kvSnapshots: Boolean = true,
    /** Summarize early (after a reply crosses ~75% of the window) instead of waiting for it to fill. */
    val autoCompact: Boolean = true,
) {
    companion object {
        const val DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant running entirely on the user's phone, offline.
Answer the current message directly. Match the length to the question: short questions get short answers, and reply to acknowledgements like "thanks" or "good to know" in one line. Do not restate or re-answer earlier messages unless asked.
If the message is unclear, ask a brief clarifying question instead of guessing what was meant.
If you are unsure or do not know something (a specific paper, product, person or recent event), say so plainly rather than inventing details.
When documents or images are attached, ground your answers in them. Continue the conversation naturally.
A "Memory of earlier conversation" section or a <recalled_context> block, when present, is background from earlier in this same chat: use it silently for continuity and never mention or repeat it."""

        private const val PREFS = "edgechat"
        private const val KEY = "settings.v1"
        private val json = Json { ignoreUnknownKeys = true; encodeDefaults = true }

        fun defaults(context: Context): AppSettings {
            val am = context.getSystemService(Context.ACTIVITY_SERVICE) as android.app.ActivityManager
            val mi = android.app.ActivityManager.MemoryInfo().also { am.getMemoryInfo(it) }
            return AppSettings(engine = EngineConfig(contextLength = EngineConfig.defaultContextLength(mi.totalMem)))
        }

        fun load(context: Context): AppSettings {
            val raw = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).getString(KEY, null) ?: return defaults(context)
            return runCatching { json.decodeFromString<AppSettings>(raw) }.getOrElse { defaults(context) }
        }
    }

    fun save(context: Context) {
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit().putString(KEY, json.encodeToString(serializer(), this)).apply()
    }
}
