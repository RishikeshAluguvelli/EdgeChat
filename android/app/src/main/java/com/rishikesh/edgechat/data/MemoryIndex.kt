package com.rishikesh.edgechat.data

import android.util.Base64
import com.rishikesh.edgechat.model.Message
import com.rishikesh.edgechat.model.Role
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.text.BreakIterator
import java.util.UUID
import kotlin.math.ln
import kotlin.math.sqrt

/** A retrievable chunk of conversation history (a message, or a slice of a long message/document). */
@Serializable
data class MemoryChunk(
    val id: String = UUID.randomUUID().toString().uppercase(),
    val messageID: String,
    val messageIndex: Int,
    val role: Role,
    /** "message" or the attachment file name. */
    val source: String,
    val text: String,
    /** Little-endian Float32 base64 (same encoding as the iOS app); null when not embedded yet. */
    val vector: String? = null,
    val createdAt: String,
) {
    val embedding: FloatArray get() = vector?.let { decodeVector(it) } ?: FloatArray(0)
    val hasEmbedding: Boolean get() = !vector.isNullOrEmpty()

    companion object {
        fun encodeVector(v: FloatArray): String {
            val bb = ByteBuffer.allocate(v.size * 4).order(ByteOrder.LITTLE_ENDIAN)
            for (f in v) bb.putFloat(f)
            return Base64.encodeToString(bb.array(), Base64.NO_WRAP)
        }
        fun decodeVector(s: String): FloatArray {
            val bytes = runCatching { Base64.decode(s, Base64.NO_WRAP) }.getOrNull() ?: return FloatArray(0)
            val bb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
            return FloatArray(bytes.size / 4) { bb.getFloat() }
        }
    }
}

/** Per-conversation retrieval index: LLM mean-pooled embeddings (via the engine) + BM25. */
@Serializable
data class MemoryIndex(
    val chunks: List<MemoryChunk> = emptyList(),
    val indexedMessageIDs: Set<String> = emptySet(),
    /** Identifies the embedding space (model) the chunk vectors were produced in. */
    val embeddingKey: String? = null,
) {
    data class Hit(val chunk: MemoryChunk, val score: Double)

    /** Adds a message (and any extra texts such as full document extractions). Returns the new index. */
    suspend fun index(
        message: Message, messageIndex: Int, extraTexts: List<Pair<String, String>> = emptyList(),
        embedder: (suspend (List<String>) -> List<FloatArray>)?, embeddingKey: String?,
    ): MemoryIndex {
        if (message.id in indexedMessageIDs) return this
        val sources = mutableListOf<Pair<String, String>>()
        if (message.content.isNotBlank()) sources += "message" to message.content
        sources += extraTexts
        var new = sources.flatMap { (source, text) ->
            chunk(text).map { MemoryChunk(messageID = message.id, messageIndex = messageIndex, role = message.role, source = source, text = it, createdAt = message.createdAt) }
        }
        var chunksOut = chunks
        var key = this.embeddingKey
        if (embedder != null && embeddingKey != null && new.isNotEmpty()) {
            if (key != embeddingKey) { key = embeddingKey; chunksOut = chunksOut.map { it.copy(vector = null) } }
            val vectors = runCatching { embedder(new.map { it.text }) }.getOrNull()
            if (vectors != null && vectors.size == new.size) new = new.mapIndexed { i, c -> c.copy(vector = MemoryChunk.encodeVector(vectors[i])) }
        }
        return copy(chunks = chunksOut + new, indexedMessageIDs = indexedMessageIDs + message.id, embeddingKey = key)
    }

    /** Re-embeds chunks that lack a vector in the current space (bounded, so a model switch never stalls a turn). */
    suspend fun refreshEmbeddings(embedder: suspend (List<String>) -> List<FloatArray>, embeddingKey: String, limit: Int = 24): MemoryIndex {
        var chunksOut = chunks
        if (this.embeddingKey != embeddingKey) chunksOut = chunksOut.map { it.copy(vector = null) }
        val todo = chunksOut.indices.filter { !chunksOut[it].hasEmbedding }.take(limit)
        if (todo.isEmpty()) return copy(chunks = chunksOut, embeddingKey = embeddingKey)
        val vectors = runCatching { embedder(todo.map { chunksOut[it].text }) }.getOrNull()
        if (vectors == null || vectors.size != todo.size) return copy(chunks = chunksOut, embeddingKey = embeddingKey)
        val mutable = chunksOut.toMutableList()
        todo.forEachIndexed { j, i -> mutable[i] = mutable[i].copy(vector = MemoryChunk.encodeVector(vectors[j])) }
        return copy(chunks = mutable, embeddingKey = embeddingKey)
    }

    /** Hybrid ranking: min-max rescaled cosine (0.55) + normalized BM25 (0.45), grounded by a shared term or a clear spread winner. */
    fun search(query: String, queryEmbedding: FloatArray, candidates: List<MemoryChunk>, limit: Int = 4, minScore: Double = 0.35): List<Hit> {
        if (candidates.isEmpty() || query.isBlank()) return emptyList()
        val qTerms = terms(query)
        val docs = candidates.map { terms(it.text) }
        val avgLen = maxOf(1.0, docs.sumOf { it.size }.toDouble() / docs.size)
        val df = HashMap<String, Int>()
        for (d in docs) for (t in d.toSet()) df[t] = (df[t] ?: 0) + 1
        val n = docs.size.toDouble()
        fun bm25(d: List<String>): Double {
            if (d.isEmpty()) return 0.0
            val tf = HashMap<String, Int>()
            for (t in d) tf[t] = (tf[t] ?: 0) + 1
            var score = 0.0
            for (q in qTerms.toSet()) {
                val f = tf[q] ?: continue
                val dfq = (df[q] ?: 0).toDouble()
                val idf = ln(1 + (n - dfq + 0.5) / (dfq + 0.5))
                val lenNorm = 0.25 + 0.75 * d.size / avgLen
                score += idf * (f * 2.2 / (f + 1.2 * lenNorm))
            }
            return score
        }
        val lexical = docs.map(::bm25)
        val maxLex = lexical.maxOrNull() ?: 0.0
        val cosines = candidates.map { c -> if (queryEmbedding.isEmpty() || !c.hasEmbedding) Double.NaN else cosine(queryEmbedding, c.embedding) }
        val valid = cosines.filter { !it.isNaN() }
        val cMin = valid.minOrNull() ?: 0.0
        val cMax = valid.maxOrNull() ?: 0.0
        val hits = mutableListOf<Hit>()
        for ((i, c) in candidates.withIndex()) {
            val cos = cosines[i]
            val semantic = when {
                cos.isNaN() -> 0.0
                cMax > cMin -> (cos - cMin) / (cMax - cMin)
                valid.size == 1 -> 1.0
                else -> 0.0
            }
            val lex = if (maxLex > 0) lexical[i] / maxLex else 0.0
            val score = 0.55 * semantic + 0.45 * lex
            val grounded = lexical[i] > 0 || (semantic >= 0.9 && cMax - cMin >= 0.04)
            if (score >= minScore && grounded) hits += Hit(c, score)
        }
        return hits.sortedByDescending { it.score }.take(limit)
    }

    companion object {
        private val json = Json { ignoreUnknownKeys = true; encodeDefaults = true }

        fun load(file: File): MemoryIndex =
            if (file.exists()) runCatching { json.decodeFromString<MemoryIndex>(file.readText()) }.getOrDefault(MemoryIndex()) else MemoryIndex()

        fun save(index: MemoryIndex, file: File) {
            runCatching {
                val tmp = File(file.parentFile, file.name + ".tmp")
                tmp.writeText(json.encodeToString(serializer(), index))
                if (!tmp.renameTo(file)) { file.delete(); tmp.renameTo(file) }
            }
        }

        /** Renders hits as a block for the prompt, oldest first, within a character budget. */
        fun renderRecall(hits: List<Hit>, maxCharacters: Int): String? {
            if (hits.isEmpty()) return null
            val lines = mutableListOf<String>()
            var used = 0
            for (h in hits.sortedBy { it.chunk.messageIndex }) {
                val who = if (h.chunk.source == "message") (if (h.chunk.role == Role.USER) "User said" else "Assistant said") else "From ${h.chunk.source}"
                val line = "- ($who, message ${h.chunk.messageIndex + 1}) ${h.chunk.text.replace("\n", " ")}"
                if (used + line.length > maxCharacters) break
                lines += line; used += line.length
            }
            if (lines.isEmpty()) return null
            return "<recalled_context note=\"background only: earlier parts of this conversation that may be relevant. Do not repeat or re-answer them; use them only if they help with the message below.\">\n" +
                lines.joinToString("\n") + "\n</recalled_context>"
        }

        /** ~600-character chunks on paragraph/sentence boundaries. */
        fun chunk(text: String, target: Int = 600, maxLength: Int = 900): List<String> {
            val paragraphs = text.split("\n\n").map { it.trim() }.filter { it.isNotEmpty() }
            val out = mutableListOf<String>()
            var current = StringBuilder()
            fun flush() { if (current.isNotEmpty()) { out += current.toString(); current = StringBuilder() } }
            for (p in paragraphs) {
                val units = if (p.length > maxLength) sentences(p) else listOf(p)
                for (u in units) {
                    if (current.length + u.length + 1 > target && current.isNotEmpty()) flush()
                    if (u.length > maxLength) {
                        var rest = u
                        while (rest.isNotEmpty()) { val piece = rest.take(maxLength); rest = rest.drop(maxLength); flush(); out += piece }
                    } else {
                        if (current.isEmpty()) current.append(u) else current.append("\n").append(u)
                    }
                }
            }
            flush()
            return out
        }

        private fun sentences(text: String): List<String> {
            val it = BreakIterator.getSentenceInstance()
            it.setText(text)
            val result = mutableListOf<String>()
            var start = it.first()
            var end = it.next()
            while (end != BreakIterator.DONE) {
                val s = text.substring(start, end).trim()
                if (s.isNotEmpty()) result += s
                start = end; end = it.next()
            }
            return result.ifEmpty { listOf(text) }
        }

        fun cosine(a: FloatArray, b: FloatArray): Double {
            if (a.size != b.size || a.isEmpty()) return 0.0
            var dot = 0f; var na = 0f; var nb = 0f
            for (i in a.indices) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i] }
            if (na <= 0 || nb <= 0) return 0.0
            return (dot / (sqrt(na) * sqrt(nb))).toDouble()
        }

        fun terms(text: String): List<String> =
            text.lowercase().split(Regex("[^\\p{L}\\p{N}]+")).filter { it.length > 1 && it !in stopwords }

        private val stopwords = setOf("the", "a", "an", "and", "or", "of", "to", "in", "on", "is", "it", "that", "this", "for", "with", "as", "was", "are", "be", "at", "by", "i", "you", "we", "he", "she", "they", "my", "your", "me", "do", "did", "what", "how", "why", "can", "could", "would", "should", "about", "from", "so", "if", "not", "no", "yes", "there", "here", "have", "has", "had", "will", "just", "also", "but")
    }
}
