package com.rishikesh.edgechat.engine

import android.content.Context
import com.rishikesh.edgechat.data.AttachmentIngest
import com.rishikesh.edgechat.data.ConversationStore
import com.rishikesh.edgechat.data.DiagnosticsLog
import com.rishikesh.edgechat.data.InstalledModel
import com.rishikesh.edgechat.data.MemoryIndex
import com.rishikesh.edgechat.data.PendingAttachment
import com.rishikesh.edgechat.model.AppSettings
import com.rishikesh.edgechat.model.Conversation
import com.rishikesh.edgechat.model.EngineConfig
import com.rishikesh.edgechat.model.GenerationStats
import com.rishikesh.edgechat.model.Message
import com.rishikesh.edgechat.model.Role
import com.rishikesh.edgechat.model.StopReason
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlinx.coroutines.yield
import java.io.File
import java.security.MessageDigest
import java.util.Locale

/**
 * Owns the LlamaEngine and drives conversation turns: load, send, stop, continue, regenerate, plus the
 * long-conversation machinery (rolling summary, background compaction, retrieval, KV snapshots).
 * Port of the iOS EngineController; all methods are called from the main thread.
 */
class EngineController(
    private val context: Context,
    private val store: ConversationStore,
    private val scope: CoroutineScope,
    private val settingsProvider: () -> AppSettings,
) {
    sealed class State {
        data object Idle : State()
        data class Loading(val progress: Float, val name: String) : State()
        data class Ready(val info: LoadedModelInfo) : State()
        data class Failed(val message: String) : State()
    }
    enum class Activity { IDLE, RETRIEVING, SUMMARIZING, GENERATING, SAVING }

    val engine = LlamaEngine()
    private val _state = MutableStateFlow<State>(State.Idle)
    val state: StateFlow<State> = _state
    private val _activity = MutableStateFlow(Activity.IDLE)
    val activity: StateFlow<Activity> = _activity
    private val _isGenerating = MutableStateFlow(false)
    val isGenerating: StateFlow<Boolean> = _isGenerating
    private val _isCompacting = MutableStateFlow(false)
    val isCompacting: StateFlow<Boolean> = _isCompacting
    private val _generatingConversationID = MutableStateFlow<String?>(null)
    val generatingConversationID: StateFlow<String?> = _generatingConversationID
    private val _lastError = MutableStateFlow<String?>(null)
    val lastError: StateFlow<String?> = _lastError

    var loadedModel: InstalledModel? = null
        private set
    private var loadedConfig: EngineConfig? = null
    private var generationJob: Job? = null
    private var compactionJob: Job? = null
    private var cacheConversationID: String? = null
    private val memoryIndexes = HashMap<String, MemoryIndex>()
    private val snapshotsRoot = File(context.cacheDir, "KVSnapshots")

    val isReady get() = _state.value is State.Ready
    val info get() = (_state.value as? State.Ready)?.info
    val hasVision get() = info?.hasVision ?: false

    fun clearError() { _lastError.value = null }

    private val modelKey: String? get() {
        val m = loadedModel ?: return null; val c = loadedConfig ?: return null
        val raw = "${m.modelFile.name}|${m.mmprojFile?.name ?: ""}|${c.contextLength}|${c.kvCacheQ8}"
        return MessageDigest.getInstance("SHA-256").digest(raw.toByteArray()).take(8).joinToString("") { String.format(Locale.US, "%02x", it) }
    }
    private val embeddingKey: String? get() = loadedModel?.modelFile?.name

    // MARK: Model lifecycle

    suspend fun load(model: InstalledModel) {
        stop()
        val config = settingsProvider().engine
        _state.value = State.Loading(0f, model.name)
        try {
            val info = engine.load(model.modelFile.absolutePath, model.mmprojFile?.absolutePath, config) { p ->
                scope.launch { (_state.value as? State.Loading)?.let { _state.value = it.copy(progress = p) } }
            }
            loadedModel = model
            loadedConfig = config
            cacheConversationID = null
            _state.value = State.Ready(info)
            modelKey?.let { key -> snapshotsRoot.listFiles()?.filter { it.name != key }?.forEach { it.deleteRecursively() } }
        } catch (e: Throwable) {
            DiagnosticsLog.append("[load-error] ${model.name}: ${e.message}")
            _state.value = State.Failed(e.message ?: "Could not load the model.")
            loadedModel = null; loadedConfig = null
        }
    }

    suspend fun unload() {
        stop()
        engine.unload()
        _state.value = State.Idle
        loadedModel = null; loadedConfig = null; cacheConversationID = null
    }

    suspend fun applyEngineSettingsIfNeeded() {
        val model = loadedModel ?: return
        if (loadedConfig == settingsProvider().engine) return
        load(model)
    }

    suspend fun clearCache() { engine.clearCache(); cacheConversationID = null }

    fun stop() { generationJob?.cancel(); compactionJob?.cancel() }

    private fun snapshotFile(conversationID: String): File? = modelKey?.let { File(File(snapshotsRoot, it), "$conversationID.kv") }

    /** Writes the KV cache of the chat currently in the engine to disk (on chat switch / background). */
    suspend fun saveCurrentSnapshot() {
        val settings = settingsProvider()
        if (!settings.kvSnapshots || _isGenerating.value || _isCompacting.value) return
        val id = cacheConversationID ?: return
        val file = snapshotFile(id) ?: return
        _activity.value = Activity.SAVING
        runCatching { engine.saveState(file) }
        file.parentFile?.listFiles { f -> f.extension == "kv" }?.sortedByDescending { it.lastModified() }?.drop(3)?.forEach { f ->
            f.delete(); File(f.path + ".units").delete()
        }
        _activity.value = Activity.IDLE
    }

    fun forgetConversation(id: String) {
        snapshotsRoot.listFiles()?.forEach { dir -> File(dir, "$id.kv").delete(); File(dir, "$id.kv.units").delete() }
        memoryIndexes.remove(id)
        if (cacheConversationID == id) cacheConversationID = null
    }

    // MARK: Turns

    suspend fun send(conversationID: String, text: String, pending: List<PendingAttachment>) {
        if (!isReady || _isGenerating.value) return
        if (pending.isEmpty() && isContinueRequest(text) && canContinue(conversationID)) { continueReply(conversationID); return }
        val settings = settingsProvider()
        val dir = store.attachmentsDirectory(conversationID)
        val result = AttachmentIngest.ingest(context, pending, dir, settings, hasVision)
        if (result.errors.isNotEmpty()) _lastError.value = result.errors.joinToString("\n")
        if (text.isEmpty() && result.attachments.isEmpty()) return
        val user = Message(role = Role.USER, content = text, attachments = result.attachments)
        store.modify(conversationID) { c ->
            val cleaned = c.messages.filterNot { it.role == Role.ASSISTANT && it.content.isEmpty() && it.error == null }
            val messages = cleaned + user
            val title = if (messages.count { it.role == Role.USER } == 1) {
                (text.ifEmpty { result.attachments.firstOrNull()?.fileName ?: "New chat" }).take(48).trim()
            } else c.title
            c.copy(messages = messages, title = title, modelID = loadedModel?.id)
        }
        generate(conversationID)
    }

    suspend fun regenerate(conversationID: String) {
        if (!isReady || _isGenerating.value) return
        var c = store.conversation(conversationID) ?: return
        if (c.messages.lastOrNull()?.role == Role.ASSISTANT) c = c.copy(messages = c.messages.dropLast(1))
        if (c.messages.lastOrNull()?.role != Role.USER) return
        store.update(c, touch = false)
        generate(conversationID)
    }

    fun canContinue(conversationID: String): Boolean {
        val last = store.conversation(conversationID)?.messages?.lastOrNull() ?: return false
        if (last.role != Role.ASSISTANT || last.content.isEmpty() || last.error != null) return false
        val stop = last.stats?.stopReason ?: return false
        return stop == StopReason.MAX_TOKENS || stop == StopReason.CONTEXT_FULL
    }

    suspend fun continueReply(conversationID: String) {
        if (!isReady || _isGenerating.value || !canContinue(conversationID)) return
        generate(conversationID, Mode.CONTINUATION)
    }

    // MARK: Prompt assembly

    private fun systemPrompt(c: Conversation, settings: AppSettings): String {
        var s = c.systemPrompt ?: settings.systemPrompt
        c.summary?.takeIf { it.isNotEmpty() }?.let { s += "\n\n## Memory of earlier conversation (older messages were removed to save space)\n$it" }
        return s
    }

    private fun makeTurns(c: Conversation, from: Int, conversationID: String, settings: AppSettings): Pair<List<ChatTurn>, List<Int>> {
        val vision = hasVision
        val dir = store.attachmentsDirectory(conversationID)
        val turns = mutableListOf<ChatTurn>(); val indices = mutableListOf<Int>()
        for ((i, m) in c.messages.withIndex()) {
            if (i < from) continue
            when (m.role) {
                Role.SYSTEM -> continue
                Role.USER -> turns += ChatTurn(Role.USER, AttachmentIngest.engineText(m, vision), if (vision) AttachmentIngest.imageInputs(m, dir, settings) else emptyList())
                Role.ASSISTANT -> { if (m.content.isEmpty()) continue; turns += ChatTurn(Role.ASSISTANT, m.content) }
            }
            indices += i
        }
        return turns to indices
    }

    // MARK: Long-conversation memory

    private suspend fun updateMemoryIndex(c: Conversation, conversationID: String, upTo: Int): MemoryIndex {
        val file = store.memoryIndexFile(conversationID)
        var index = memoryIndexes[conversationID] ?: withContext(Dispatchers.IO) { MemoryIndex.load(file) }
        val embedder: suspend (List<String>) -> List<FloatArray> = { engine.embed(it) }
        var changed = false
        for ((i, m) in c.messages.withIndex()) {
            if (i >= upTo || m.id in index.indexedMessageIDs) continue
            if (m.role == Role.SYSTEM || (m.role == Role.ASSISTANT && m.content.isEmpty())) continue
            val docs = m.attachments.mapNotNull { a -> a.extractedText?.takeIf { it.isNotEmpty() }?.let { a.fileName to it } }
            index = index.index(m, i, docs, embedder, embeddingKey)
            changed = true
        }
        embeddingKey?.let { key ->
            if (index.embeddingKey != key || index.chunks.any { !it.hasEmbedding }) { index = index.refreshEmbeddings(embedder, key); changed = true }
        }
        memoryIndexes[conversationID] = index
        if (changed) { val snapshot = index; withContext(Dispatchers.IO) { MemoryIndex.save(snapshot, file) } }
        return index
    }

    /** Folds turns that no longer fit into the conversation's summary. `budgetFraction` < 1 compacts further (proactive). */
    private suspend fun summarizeIfNeeded(conversation: Conversation, conversationID: String, settings: AppSettings, reserve: Int, budgetFraction: Double = 1.0): Conversation {
        if (!settings.summarizeDroppedTurns) return conversation
        var c = conversation
        val sampling = settings.sampling.copy(maxTokens = settings.sampling.maxTokens + reserve)
        repeat(3) {
            if (!scope.isActive) return c
            val (turns, indices) = makeTurns(c, c.summaryCoversMessages, conversationID, settings)
            if (turns.size <= 1) return c
            val system = systemPrompt(c, settings)
            var planned = runCatching { engine.planTruncation(system, turns, sampling, budgetFraction) }.getOrNull()
            if (planned == null && budgetFraction < 1) planned = runCatching { engine.planTruncation(system, turns, sampling) }.getOrNull()
            val drop = planned ?: return c
            if (drop <= 0 || drop >= turns.size) return c
            _activity.value = Activity.SUMMARIZING
            val t0 = System.currentTimeMillis()
            val maxWords = minOf(400, maxOf(160, (info?.contextLength ?: 4096) / 16))
            val summary = runCatching { engine.summarize(c.summary, turns.subList(0, drop), maxWords) }.getOrNull() ?: return c
            val covers = indices[drop - 1] + 1
            c = c.copy(summary = summary, summaryCoversMessages = covers)
            store.modify(conversationID, touch = false) { live -> live.copy(summary = summary, summaryCoversMessages = covers) }
            DiagnosticsLog.append("[summary] turns=$drop covers=$covers words=${summary.split(' ').size} fraction=$budgetFraction seconds=${(System.currentTimeMillis() - t0) / 1000.0}")
        }
        return c
    }

    private fun windowUsage(stats: GenerationStats, settings: AppSettings): Double? {
        val nCtx = stats.contextLength ?: return null
        if (nCtx <= 0) return null
        val budget = nCtx - LlamaEngine.replyReserve(nCtx, settings.sampling) - 8
        if (budget <= 0) return null
        return (stats.promptTokens + stats.generatedTokens).toDouble() / budget
    }

    private fun scheduleCompactionIfNeeded(conversationID: String, stats: GenerationStats, settings: AppSettings) {
        if (!settings.autoCompact || !settings.summarizeDroppedTurns || compactionJob?.isActive == true) return
        val usage = windowUsage(stats, settings) ?: return
        val shifted = (stats.contextShifts ?: 0) > 0 || stats.stopReason == StopReason.CONTEXT_FULL
        if (usage < COMPACT_TRIGGER && !shifted) return
        DiagnosticsLog.append("[compact] start usage=${String.format(Locale.US, "%.2f", usage)} shifted=$shifted")
        compactionJob = scope.launch {
            val c = store.conversation(conversationID) ?: return@launch
            _isCompacting.value = true
            try {
                val recallReserve = if (settings.memoryRetrieval) ((info?.contextLength ?: 4096) * 0.12).toInt() else 0
                val after = summarizeIfNeeded(c, conversationID, settings, recallReserve, COMPACT_TARGET)
                DiagnosticsLog.append("[compact] done covers=${after.summaryCoversMessages}")
            } finally {
                _isCompacting.value = false
                _activity.value = Activity.IDLE
            }
        }
    }

    /** RAG over messages outside the live window and over full document text, injected into the latest user turn. */
    private suspend fun recallIfUseful(conversation: Conversation, conversationID: String, settings: AppSettings): Conversation {
        val info = info ?: return conversation
        if (!settings.memoryRetrieval) return conversation
        val lastIndex = conversation.messages.lastIndex
        if (lastIndex < 0 || conversation.messages[lastIndex].role != Role.USER) return conversation
        val last = conversation.messages[lastIndex]
        _activity.value = Activity.RETRIEVING
        val index = updateMemoryIndex(conversation, conversationID, lastIndex)
        val windowStart = conversation.summaryCoversMessages
        val truncatedDocMessages = conversation.messages.filter { m -> m.attachments.any { it.textTruncated } }.map { it.id }.toSet()
        val candidates = index.chunks.filter { ch ->
            ch.messageID != last.id && (ch.messageIndex < windowStart || (ch.source != "message" && ch.messageID in truncatedDocMessages))
        }
        if (candidates.isEmpty()) return conversation
        val query = last.content.ifEmpty { last.attachments.joinToString(" ") { it.fileName } }
        if (isSmallTalk(query)) return conversation
        val qEmb = runCatching { engine.embed(listOf(query)).firstOrNull() }.getOrNull() ?: FloatArray(0)
        val hits = index.search(query, qEmb, candidates, limit = 4, minScore = 0.35)
        val budgetChars = maxOf(600, (info.contextLength * 0.12 * 3.5).toInt())
        val recall = MemoryIndex.renderRecall(hits, budgetChars)
        val updated = conversation.copy(messages = conversation.messages.toMutableList().also { it[lastIndex] = last.copy(recalledContext = recall) })
        store.modify(conversationID, touch = false) { live ->
            live.copy(messages = live.messages.map { if (it.id == last.id) it.copy(recalledContext = recall) else it })
        }
        return updated
    }

    // MARK: Generation

    enum class Mode { REPLY, CONTINUATION }

    private suspend fun generate(conversationID: String, mode: Mode = Mode.REPLY, autoContinued: Int = 0) {
        var conversation = store.conversation(conversationID) ?: return
        val settings = settingsProvider()
        _isGenerating.value = true
        _generatingConversationID.value = conversationID
        try {
            // 1. Park the previous chat's KV cache on disk, then restore this conversation's if we have one.
            if (cacheConversationID != conversationID) {
                saveCurrentSnapshot()
                val file = snapshotFile(conversationID)
                if (!(settings.kvSnapshots && file != null && engine.loadState(file))) engine.clearCache()
                cacheConversationID = conversationID
            }
            // 2. Let a running background compaction finish, then summarize what still would not fit, then recall.
            compactionJob?.let { if (it.isActive) { _activity.value = Activity.SUMMARIZING; it.join(); store.conversation(conversationID)?.let { f -> conversation = f } } }
            val recallReserve = if (settings.memoryRetrieval) ((info?.contextLength ?: 4096) * 0.12).toInt() else 0
            conversation = summarizeIfNeeded(conversation, conversationID, settings, recallReserve)
            if (mode == Mode.REPLY) conversation = recallIfUseful(conversation, conversationID, settings)

            val (turns, _) = makeTurns(conversation, conversation.summaryCoversMessages, conversationID, settings)
            val system = systemPrompt(conversation, settings)

            val target: Message
            val previousStats: GenerationStats?
            when (mode) {
                Mode.REPLY -> {
                    target = Message(role = Role.ASSISTANT, content = "")
                    previousStats = null
                    store.modify(conversationID, touch = false) { it.copy(messages = it.messages + target) }
                }
                Mode.CONTINUATION -> {
                    val last = conversation.messages.lastOrNull()
                    if (last == null || last.role != Role.ASSISTANT || turns.lastOrNull()?.role != Role.ASSISTANT) return
                    target = last; previousStats = last.stats
                }
            }
            _activity.value = Activity.GENERATING

            val splitter = ThinkTagSplitter()
            var content = if (mode == Mode.CONTINUATION) target.content else ""
            var reasoning = if (mode == Mode.CONTINUATION) (target.reasoning ?: "") else ""
            var lastFlush = 0L
            var finalStats: GenerationStats? = null
            var failure: String? = null
            fun flush(force: Boolean) {
                val now = System.currentTimeMillis()
                if (!force && now - lastFlush < 50) return
                lastFlush = now
                val c = content; val r = reasoning
                store.modify(conversationID, touch = false, persist = force) { conv ->
                    conv.copy(messages = conv.messages.map { if (it.id == target.id) it.copy(content = c, reasoning = r.ifEmpty { null }) else it })
                }
            }

            val job = scope.launch {
                try {
                    engine.respond(system, turns, settings.sampling, continuing = mode == Mode.CONTINUATION).collect { ev ->
                        when (ev) {
                            is GenerationEvent.PromptProcessed -> {}
                            is GenerationEvent.Token -> {
                                for (p in splitter.feed(ev.text)) when (p) {
                                    is ThinkTagSplitter.Part.Reasoning -> reasoning += p.text
                                    is ThinkTagSplitter.Part.Text -> content += p.text
                                }
                                flush(false)
                            }
                            is GenerationEvent.Finished -> finalStats = ev.stats
                        }
                    }
                } catch (e: kotlinx.coroutines.CancellationException) {
                    // user stopped
                } catch (e: Throwable) {
                    failure = e.message ?: "Generation failed."
                }
            }
            generationJob = job
            job.join()
            yield()
            for (p in splitter.flush()) when (p) {
                is ThinkTagSplitter.Part.Reasoning -> reasoning += p.text
                is ThinkTagSplitter.Part.Text -> content += p.text
            }
            content = content.trim(); reasoning = reasoning.trim()
            flush(true)
            var stats = finalStats ?: if (failure == null) GenerationStats(0, 0, 0, 0.0, 0.0, StopReason.CANCELLED) else null
            stats = stats?.copy(modelID = loadedModel?.id)
            stats?.let { st ->
                DiagnosticsLog.append("[turn${if (mode == Mode.CONTINUATION) "+" else ""}] model=${loadedModel?.id ?: "-"} ctx=${st.contextLength ?: 0} prompt=${st.promptTokens} cached=${st.cachedTokens} gen=${st.generatedTokens} stop=${st.stopReason} shifts=${st.contextShifts ?: 0} tps=${String.format(Locale.US, "%.1f", st.tokensPerSecond)}")
            }
            failure?.let { DiagnosticsLog.append("[turn-error] $it") }
            if (stats != null && previousStats != null) {
                val merged = stats.copy(generatedTokens = stats.generatedTokens + previousStats.generatedTokens,
                    decodeSeconds = stats.decodeSeconds + previousStats.decodeSeconds,
                    prefillSeconds = stats.prefillSeconds + previousStats.prefillSeconds,
                    contextShifts = ((stats.contextShifts ?: 0) + (previousStats.contextShifts ?: 0)).takeIf { it > 0 })
                stats = if (merged.stopReason == StopReason.CANCELLED && merged.generatedTokens == previousStats.generatedTokens) previousStats else merged
            }
            val s = stats; val f = failure
            store.modify(conversationID) { conv -> conv.copy(messages = conv.messages.map { if (it.id == target.id) it.copy(stats = s, error = f) else it }) }
            _isGenerating.value = false
            _generatingConversationID.value = null

            // 3. Index the new reply for later recall; 4. compact early if the window is nearly full.
            if (settings.memoryRetrieval) store.conversation(conversationID)?.let { c -> updateMemoryIndex(c, conversationID, c.messages.size) }
            if (s != null && f == null && !job.isCancelled && s.stopReason != StopReason.CONTEXT_FULL) scheduleCompactionIfNeeded(conversationID, s, settings)

            // 5. The reply ran into the window edge: summarize the oldest turns away and keep writing the same reply.
            val lastNow = store.conversation(conversationID)?.messages?.lastOrNull()
            if (!job.isCancelled && autoContinued < MAX_AUTO_CONTINUATIONS && lastNow != null && lastNow.id == target.id &&
                lastNow.error == null && lastNow.stats?.stopReason == StopReason.CONTEXT_FULL) {
                DiagnosticsLog.append("[auto-continue] ${autoContinued + 1} after context-full stop")
                generate(conversationID, Mode.CONTINUATION, autoContinued + 1)
            }
        } finally {
            _isGenerating.value = false
            _generatingConversationID.value = null
            _activity.value = Activity.IDLE
        }
    }

    companion object {
        const val COMPACT_TRIGGER = 0.75
        const val COMPACT_TARGET = 0.5
        const val MAX_AUTO_CONTINUATIONS = 3

        private val continuePattern = Regex("^\\s*(please\\s+)?(continue|go on|keep going|carry on|finish( the answer| it| that)?|more|resume|and then\\??)(\\s+please)?[\\s.!?]*$", RegexOption.IGNORE_CASE)
        fun isContinueRequest(text: String): Boolean { val t = text.trim(); return t.isNotEmpty() && t.length <= 40 && continuePattern.matches(t) }

        private val smallTalkWords = setOf("ok", "okay", "k", "thanks", "thank", "you", "thx", "ty", "great", "good", "to", "know", "cool", "nice", "awesome",
            "perfect", "got", "it", "sure", "yes", "yeah", "yep", "no", "nope", "ha", "haha", "lol", "hmm", "hi", "hello", "hey", "sounds", "makes", "sense",
            "understood", "noted", "fine", "alright", "right", "wow", "oh", "interesting", "super", "brilliant", "cheers", "please", "continue", "go", "on",
            "more", "next", "a", "lot", "very", "much", "that", "this", "is", "was", "helpful", "really", "so", "and", "the", "i", "see", "well", "done", "bye", "goodbye", "ah", "use")
        fun isSmallTalk(text: String): Boolean {
            val words = text.lowercase().split(Regex("[^\\p{L}\\p{N}]+")).filter { it.isNotEmpty() }
            if (words.isEmpty() || words.size < 3) return true
            return words.all { it in smallTalkWords }
        }
    }
}
