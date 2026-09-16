package com.rishikesh.edgechat.model

enum class ModelCapability { VISION, REASONING }

enum class Tier(val title: String) {
    FLAGSHIP("Best for 8 GB+ phones"),
    BALANCED("Good on 6 GB phones"),
    TINY("Tiny (testing / older phones)"),
}

/** A downloadable GGUF model (plus optional vision projector). Same catalog as the iOS app. */
data class ModelSpec(
    val id: String,
    val name: String,
    val family: String,
    val parameters: String,
    val quantization: String,
    val url: String,
    val sizeBytes: Long,
    val mmprojURL: String?,
    val mmprojSizeBytes: Long,
    val capabilities: Set<ModelCapability>,
    val contextTrain: Int,
    val summary: String,
    val tier: Tier,
) {
    val fileName: String get() = url.substringAfterLast('/')
    val mmprojFileName: String? get() = mmprojURL?.let { "$id-${it.substringAfterLast('/')}" }
    val totalBytes: Long get() = sizeBytes + mmprojSizeBytes
    val hasVision: Boolean get() = ModelCapability.VISION in capabilities

    /** Rough working-set estimate: weights + projector + KV cache/activations headroom. */
    fun estimatedRAMBytes(contextLength: Int): Long =
        (totalBytes * 1.1).toLong() + contextLength.toLong() * 64 * 1024 + 300L * 1024 * 1024

    enum class Fit { GOOD, TIGHT, TOO_LARGE }

    fun fit(physicalMemory: Long, contextLength: Int): Fit {
        val need = estimatedRAMBytes(contextLength).toDouble()
        val ram = physicalMemory.toDouble()
        return when { need < ram * 0.5 -> Fit.GOOD; need < ram * 0.68 -> Fit.TIGHT; else -> Fit.TOO_LARGE }
    }
}

object ModelCatalog {
    private fun hf(repo: String, file: String) = "https://huggingface.co/$repo/resolve/main/$file"
    private const val GB = 1_073_741_824.0

    val models: List<ModelSpec> = listOf(
        ModelSpec("qwen3-vl-4b-instruct", "Qwen3-VL 4B Instruct", "Qwen3-VL", "4B", "Q4_K_M",
            hf("unsloth/Qwen3-VL-4B-Instruct-GGUF", "Qwen3-VL-4B-Instruct-Q4_K_M.gguf"), (2.33 * GB).toLong(),
            hf("unsloth/Qwen3-VL-4B-Instruct-GGUF", "mmproj-F16.gguf"), (0.78 * GB).toLong(),
            setOf(ModelCapability.VISION), 262_144,
            "Recommended. Best all-rounder that fits a phone: strong chat, coding and math plus genuine image understanding (photos, screenshots, charts, documents).",
            Tier.FLAGSHIP),
        ModelSpec("qwen3-4b-instruct-2507", "Qwen3 4B Instruct (2507)", "Qwen3", "4B", "Q4_K_M",
            hf("unsloth/Qwen3-4B-Instruct-2507-GGUF", "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"), (2.33 * GB).toLong(), null, 0,
            emptySet(), 262_144,
            "Strongest text-only 4B model. Images are handled with on-device OCR instead of a vision encoder.",
            Tier.FLAGSHIP),
        ModelSpec("gemma-3-4b-it", "Gemma 3 4B IT", "Gemma 3", "4B", "Q4_K_M",
            hf("unsloth/gemma-3-4b-it-GGUF", "gemma-3-4b-it-Q4_K_M.gguf"), (2.32 * GB).toLong(),
            hf("unsloth/gemma-3-4b-it-GGUF", "mmproj-F16.gguf"), (0.79 * GB).toLong(),
            setOf(ModelCapability.VISION), 131_072,
            "Google's multimodal 4B model. Friendly writing style, good multilingual support, image input.",
            Tier.FLAGSHIP),
        ModelSpec("qwen3-vl-4b-instruct-q3", "Qwen3-VL 4B Instruct (compact)", "Qwen3-VL", "4B", "Q3_K_M",
            hf("unsloth/Qwen3-VL-4B-Instruct-GGUF", "Qwen3-VL-4B-Instruct-Q3_K_M.gguf"), (1.93 * GB).toLong(),
            hf("unsloth/Qwen3-VL-4B-Instruct-GGUF", "mmproj-F16.gguf"), (0.78 * GB).toLong(),
            setOf(ModelCapability.VISION), 262_144,
            "The 4B vision model squeezed to 3-bit for 6 GB phones. Slightly less accurate than Q4 but far more capable than the 2B. Use a 4k context and the 8-bit KV cache.",
            Tier.BALANCED),
        ModelSpec("llama-3.2-3b-instruct", "Llama 3.2 3B Instruct", "Llama 3.2", "3B", "Q4_K_M",
            hf("unsloth/Llama-3.2-3B-Instruct-GGUF", "Llama-3.2-3B-Instruct-Q4_K_M.gguf"), (1.88 * GB).toLong(), null, 0,
            emptySet(), 131_072,
            "Meta's compact instruct model. Fast and reliable for everyday chat and summarization.",
            Tier.BALANCED),
        ModelSpec("qwen3-vl-2b-instruct", "Qwen3-VL 2B Instruct", "Qwen3-VL", "2B", "Q4_K_M",
            hf("unsloth/Qwen3-VL-2B-Instruct-GGUF", "Qwen3-VL-2B-Instruct-Q4_K_M.gguf"), (1.03 * GB).toLong(),
            hf("unsloth/Qwen3-VL-2B-Instruct-GGUF", "mmproj-F16.gguf"), (0.76 * GB).toLong(),
            setOf(ModelCapability.VISION), 262_144,
            "Smallest model with real image understanding. Good choice for 6 GB phones that need vision.",
            Tier.BALANCED),
        ModelSpec("qwen3-1.7b", "Qwen3 1.7B", "Qwen3", "1.7B", "Q4_K_M",
            hf("unsloth/Qwen3-1.7B-GGUF", "Qwen3-1.7B-Q4_K_M.gguf"), (1.03 * GB).toLong(), null, 0,
            setOf(ModelCapability.REASONING), 40_960,
            "Very fast hybrid reasoning model. Thinking can be toggled in Settings.",
            Tier.BALANCED),
        ModelSpec("qwen3-0.6b", "Qwen3 0.6B", "Qwen3", "0.6B", "Q4_K_M",
            hf("unsloth/Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q4_K_M.gguf"), (0.37 * GB).toLong(), null, 0,
            setOf(ModelCapability.REASONING), 40_960,
            "Tiny and instant. Limited knowledge; handy for testing and the emulator.",
            Tier.TINY),
        ModelSpec("smolvlm-256m-instruct", "SmolVLM 256M Instruct", "SmolVLM", "256M", "Q8_0",
            hf("ggml-org/SmolVLM-256M-Instruct-GGUF", "SmolVLM-256M-Instruct-Q8_0.gguf"), (0.16 * GB).toLong(),
            hf("ggml-org/SmolVLM-256M-Instruct-GGUF", "mmproj-SmolVLM-256M-Instruct-Q8_0.gguf"), (0.10 * GB).toLong(),
            setOf(ModelCapability.VISION), 8_192,
            "Tiny vision model for testing image input end to end. Very limited quality.",
            Tier.TINY),
    )

    const val recommendedID = "qwen3-vl-4b-instruct"
    fun spec(id: String): ModelSpec? = models.firstOrNull { it.id == id }
}
