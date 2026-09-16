import Foundation
import CryptoKit
import UIKit
import EdgeChatCore

/// Owns the LlamaEngine and drives conversation turns: load, send, stop, regenerate, plus the
/// long-conversation machinery (rolling summary, retrieval of older context, KV snapshots).
@MainActor @Observable
final class EngineController {
    enum State: Equatable {
        case idle
        case loading(progress: Double, name: String)
        case ready(LoadedModelInfo)
        case failed(String)
    }

    enum Activity: Equatable { case idle, retrieving, summarizing, generating, saving }

    weak var app: AppModel?
    private(set) var state: State = .idle
    private(set) var loadedModel: InstalledModel?
    private(set) var loadedConfig: EngineConfig?
    private(set) var isGenerating = false
    /// A background compaction (summary of the oldest turns) is running; the next turn waits for it.
    private(set) var isCompacting = false
    private(set) var activity: Activity = .idle
    private(set) var generatingConversationID: UUID?
    var lastError: String?

    private let engine = LlamaEngine()
    private var generationTask: Task<Void, Never>?
    private var compactionTask: Task<Void, Never>?
    /// Retrieval indexes by conversation, so a turn does not re-read the JSON from disk.
    private var memoryIndexes: [UUID: MemoryIndex] = [:]

    /// Auto-compaction: once a reply leaves the window more than `compactTriggerFraction` full (of the prompt budget),
    /// the oldest turns are summarized down to `compactTargetFraction` while the user reads the reply.
    static let compactTriggerFraction = 0.75
    static let compactTargetFraction = 0.5
    /// Conversation whose history currently occupies the engine's KV cache.
    private var cacheConversationID: UUID?
    private let snapshots = SnapshotStore()

    var isReady: Bool { if case .ready = state { return true } else { return false } }
    var info: LoadedModelInfo? { if case .ready(let i) = state { return i } else { return nil } }
    var hasVision: Bool { info?.hasVision ?? false }

    /// Identifies the loaded weights + cache layout (snapshots and embeddings are only valid within one key).
    private var modelKey: String? {
        guard let m = loadedModel, let c = loadedConfig else { return nil }
        let raw = "\(m.modelURL.lastPathComponent)|\(m.mmprojURL?.lastPathComponent ?? "")|\(c.contextLength)|\(c.kvCacheQ8)"
        return SHA256.hash(data: Data(raw.utf8)).prefix(8).map { String(format: "%02x", $0) }.joined()
    }
    private var embeddingKey: String? { loadedModel.map { $0.modelURL.lastPathComponent } }

    // MARK: Model lifecycle

    func load(_ model: InstalledModel) async {
        guard let app else { return }
        stop()
        let config = app.settings.engine
        state = .loading(progress: 0, name: model.name)
        do {
            let info = try await engine.load(modelPath: model.modelURL.path, mmprojPath: model.mmprojURL?.path, config: config) { p in
                Task { @MainActor [weak self] in
                    if case .loading(_, let name) = self?.state { self?.state = .loading(progress: p, name: name) }
                }
            }
            state = .ready(info)
            loadedModel = model
            loadedConfig = config
            cacheConversationID = nil
            app.settings.activeModelID = model.id
            if let key = modelKey { snapshots.pruneOtherModels(keeping: key) }
        } catch {
            DiagnosticsLog.append("[load-error] \(model.name): \(error.localizedDescription)")
            state = .failed(error.localizedDescription)
            loadedModel = nil
            loadedConfig = nil
        }
    }

    func unload() async {
        stop()
        await engine.unload()
        state = .idle
        loadedModel = nil
        loadedConfig = nil
        cacheConversationID = nil
    }

    /// Reloads when engine settings (context length, GPU, KV cache…) differ from what is loaded.
    func applyEngineSettingsIfNeeded() async {
        guard let model = loadedModel, let app, loadedConfig != app.settings.engine else { return }
        await load(model)
    }

    func clearCache() async {
        await engine.clearCache()
        cacheConversationID = nil
    }

    /// Writes the KV cache of the chat currently in the engine to disk (called when switching chats or backgrounding).
    func saveCurrentSnapshot() async {
        guard let app, app.settings.kvSnapshots, !isGenerating, !isCompacting, let id = cacheConversationID, let key = modelKey,
              let url = snapshots.url(conversationID: id, modelKey: key) else { return }
        activity = .saving
        try? await engine.saveState(to: url)
        snapshots.prune(keep: 3, modelKey: key)
        activity = .idle
    }

    func forgetConversation(_ id: UUID) {
        snapshots.delete(conversationID: id)
        memoryIndexes[id] = nil
        if cacheConversationID == id { cacheConversationID = nil }
    }

    // MARK: Turns

    func send(conversationID: UUID, text: String, attachments pending: [PendingAttachment]) async {
        guard let app, isReady, !isGenerating else { return }
        // "Continue" typed after a reply that was cut short resumes that reply instead of starting a new turn.
        if pending.isEmpty, Self.isContinueRequest(text), canContinue(conversationID: conversationID) {
            await continueReply(conversationID: conversationID)
            return
        }
        let dir = app.store.attachmentsDirectory(conversationID)
        let (attachments, errors) = await AttachmentIngest.ingest(pending, attachmentsDirectory: dir, settings: app.settings)
        if !errors.isEmpty { lastError = errors.joined(separator: "\n") }
        guard !text.isEmpty || !attachments.isEmpty else { return }

        let user = Message(role: .user, content: text, attachments: attachments)
        app.store.modify(conversationID) { c in
            // An empty bubble left by an interrupted reply would confuse the model (two user turns in a row).
            c.messages.removeAll { $0.role == .assistant && $0.content.isEmpty && $0.error == nil }
            c.messages.append(user)
            if c.messages.filter({ $0.role == .user }).count == 1 {
                let base = text.isEmpty ? (attachments.first?.fileName ?? "New chat") : text
                c.title = String(base.prefix(48)).trimmingCharacters(in: .whitespacesAndNewlines)
            }
            c.modelID = self.loadedModel?.id
        }
        await generate(conversationID: conversationID)
    }

    func regenerate(conversationID: UUID) async {
        guard let app, isReady, !isGenerating, var c = app.store.conversation(conversationID) else { return }
        if c.messages.last?.role == .assistant { c.messages.removeLast() }
        guard c.messages.last?.role == .user else { return }
        app.store.update(c, touch: false)
        await generate(conversationID: conversationID)
    }

    /// The last reply can be resumed: it is an assistant message that stopped at its length limit or the window edge.
    func canContinue(conversationID: UUID) -> Bool {
        guard let app, let last = app.store.conversation(conversationID)?.messages.last, last.role == .assistant,
              !last.content.isEmpty, last.error == nil, let stop = last.stats?.stopReason else { return false }
        return stop == .maxTokens || stop == .contextFull
    }

    /// Resumes the last reply where it stopped (same bubble), like "Continue generating" in hosted chat apps.
    func continueReply(conversationID: UUID) async {
        guard isReady, !isGenerating, canContinue(conversationID: conversationID) else { return }
        await generate(conversationID: conversationID, mode: .continuation)
    }

    private static let continuePattern = try! NSRegularExpression(
        pattern: #"^\s*(please\s+)?(continue|go on|keep going|carry on|finish( the answer| it| that)?|more|resume|and then\??)(\s+please)?[\s.!?]*$"#,
        options: [.caseInsensitive])

    static func isContinueRequest(_ text: String) -> Bool {
        let t = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !t.isEmpty, t.count <= 40 else { return false }
        return continuePattern.firstMatch(in: t, range: NSRange(t.startIndex..., in: t)) != nil
    }

    /// Acknowledgements and other messages that carry no retrievable intent ("thanks", "great, good to know").
    private static let smallTalkWords: Set<String> = [
        "ok", "okay", "k", "thanks", "thank", "you", "thx", "ty", "great", "good", "to", "know", "cool", "nice", "awesome",
        "perfect", "got", "it", "sure", "yes", "yeah", "yep", "no", "nope", "ha", "haha", "lol", "hmm", "hi", "hello", "hey",
        "sounds", "makes", "sense", "understood", "noted", "fine", "alright", "right", "wow", "oh", "interesting", "super",
        "brilliant", "cheers", "please", "continue", "go", "on", "more", "next", "a", "lot", "very", "much", "that", "this",
        "is", "was", "helpful", "really", "so", "and", "the", "i", "see", "well", "done", "bye", "goodbye", "ah", "use",
    ]

    static func isSmallTalk(_ text: String) -> Bool {
        let words = text.lowercased().split { !$0.isLetter && !$0.isNumber }.map(String.init)
        guard !words.isEmpty else { return true }
        if words.count < 3 { return true }
        return words.allSatisfy { smallTalkWords.contains($0) }
    }

    func stop() {
        generationTask?.cancel()
        compactionTask?.cancel()
    }

    // MARK: Prompt assembly

    private func systemPrompt(for c: Conversation, settings: AppSettings) -> String {
        var s = c.systemPrompt ?? settings.systemPrompt
        if let summary = c.summary, !summary.isEmpty {
            s += "\n\n## Memory of earlier conversation (older messages were removed to save space)\n" + summary
        }
        return s
    }

    /// Turns for messages from `start` on, plus the message index each turn came from.
    private func makeTurns(_ c: Conversation, from start: Int, conversationID: UUID, settings: AppSettings) -> (turns: [ChatTurn], indices: [Int]) {
        guard let app else { return ([], []) }
        let vision = hasVision
        let dir = app.store.attachmentsDirectory(conversationID)
        var turns: [ChatTurn] = []
        var indices: [Int] = []
        for (i, m) in c.messages.enumerated() where i >= start {
            switch m.role {
            case .system: continue
            case .user:
                turns.append(ChatTurn(role: .user,
                                      text: AttachmentIngest.engineText(for: m, modelHasVision: vision),
                                      images: vision ? AttachmentIngest.imageInputs(for: m, attachmentsDirectory: dir, settings: settings) : []))
            case .assistant:
                guard !m.content.isEmpty else { continue }
                turns.append(ChatTurn(role: .assistant, text: m.content))
            }
            indices.append(i)
        }
        return (turns, indices)
    }

    // MARK: Long-conversation memory

    /// Keeps the per-conversation retrieval index in sync with the transcript (embeds with the loaded model).
    private func updateMemoryIndex(_ c: Conversation, conversationID: UUID, upTo end: Int) async -> MemoryIndex {
        guard let app else { return MemoryIndex() }
        let url = app.store.memoryIndexURL(conversationID)
        var index = memoryIndexes[conversationID] ?? MemoryIndex.load(from: url)
        defer { memoryIndexes[conversationID] = index }
        let embedder: MemoryIndex.Embedder = { [engine] texts in try await engine.embed(texts) }
        var changed = false
        for (i, m) in c.messages.enumerated() where i < end && !index.indexedMessageIDs.contains(m.id) {
            guard m.role != .system, !(m.role == .assistant && m.content.isEmpty) else { continue }
            let docs = m.attachments.compactMap { a -> (String, String)? in
                guard let t = a.extractedText, !t.isEmpty else { return nil }
                return (a.fileName, t)
            }
            await index.index(message: m, at: i, extraTexts: docs, embedder: embedder, embeddingKey: embeddingKey)
            changed = true
        }
        if let key = embeddingKey, index.embeddingKey != key || index.chunks.contains(where: { $0.embedding.isEmpty }) {
            await index.refreshEmbeddings(embedder: embedder, embeddingKey: key)
            changed = true
        }
        if changed { try? index.save(to: url) }
        return index
    }

    /// Folds turns that no longer fit into the conversation's summary. Returns the updated conversation.
    /// `budgetFraction` < 1 compacts further than strictly necessary (used by proactive compaction).
    private func summarizeIfNeeded(_ conversation: Conversation, conversationID: UUID, settings: AppSettings, reserve: Int,
                                   budgetFraction: Double = 1) async -> Conversation {
        guard let app, settings.summarizeDroppedTurns else { return conversation }
        var c = conversation
        var sampling = settings.sampling
        sampling.maxTokens += reserve
        for _ in 0..<3 {
            if Task.isCancelled { break }
            let (turns, indices) = makeTurns(c, from: c.summaryCoversMessages, conversationID: conversationID, settings: settings)
            guard turns.count > 1 else { break }
            let system = systemPrompt(for: c, settings: settings)
            var planned = try? await engine.planTruncation(systemPrompt: system, turns: turns, sampling: sampling, budgetFraction: budgetFraction)
            if planned == nil, budgetFraction < 1 {
                // The tighter target is unreachable (e.g. tiny context); fall back to dropping only what must go.
                planned = try? await engine.planTruncation(systemPrompt: system, turns: turns, sampling: sampling)
            }
            guard let drop = planned, drop > 0, drop < turns.count else { break }
            activity = .summarizing
            let t0 = Date()
            // Memory size scales with the window: ~1/16 of it in words (256 words at 4k, 400 at 8k).
            let maxWords = min(400, max(160, Int(info?.contextLength ?? 4096) / 16))
            guard let summary = try? await engine.summarize(previousSummary: c.summary, turns: Array(turns[0..<drop]), maxWords: maxWords) else { break }
            if Task.isCancelled { break }
            c.summary = summary
            c.summaryCoversMessages = indices[drop - 1] + 1
            // Write only the summary fields: messages may have been appended (a new user turn) while the model was busy.
            let covers = c.summaryCoversMessages
            app.store.modify(conversationID, touch: false) { live in
                live.summary = summary
                live.summaryCoversMessages = covers
            }
            DiagnosticsLog.append("[summary] turns=\(drop) covers=\(c.summaryCoversMessages) words=\(summary.split(separator: " ").count) fraction=\(budgetFraction) seconds=\(String(format: "%.1f", Date().timeIntervalSince(t0)))")
        }
        return c
    }

    /// Fraction of the prompt budget a finished reply left occupied (prompt + generated cells), or nil if unknown.
    private func windowUsage(after stats: GenerationStats, settings: AppSettings) -> Double? {
        guard let nCtx = stats.contextLength, nCtx > 0 else { return nil }
        let budget = nCtx - LlamaEngine.replyReserve(contextLength: nCtx, sampling: settings.sampling) - 8
        guard budget > 0 else { return nil }
        return Double(stats.promptTokens + stats.generatedTokens) / Double(budget)
    }

    /// Starts a background compaction when the last reply left the window nearly full (or had to shift mid-reply),
    /// so the summary is written while the user reads instead of delaying their next message.
    private func scheduleCompactionIfNeeded(conversationID: UUID, stats: GenerationStats, settings: AppSettings) {
        guard settings.autoCompact, settings.summarizeDroppedTurns, compactionTask == nil,
              let usage = windowUsage(after: stats, settings: settings) else { return }
        let shifted = (stats.contextShifts ?? 0) > 0 || stats.stopReason == .contextFull
        guard usage >= Self.compactTriggerFraction || shifted else { return }
        DiagnosticsLog.append("[compact] start usage=\(String(format: "%.2f", usage)) shifted=\(shifted)")
        compactionTask = Task { @MainActor [weak self] in
            guard let self, let app, let c = app.store.conversation(conversationID) else { return }
            self.isCompacting = true
            // Keep running for a while if the phone is locked meanwhile (otherwise iOS suspends us mid-summary).
            var bgTask: UIBackgroundTaskIdentifier = .invalid
            bgTask = UIApplication.shared.beginBackgroundTask(withName: "EdgeChat.compact") {
                UIApplication.shared.endBackgroundTask(bgTask); bgTask = .invalid
            }
            defer {
                self.isCompacting = false
                self.activity = .idle
                self.compactionTask = nil
                if bgTask != .invalid { UIApplication.shared.endBackgroundTask(bgTask) }
            }
            let recallReserve = settings.memoryRetrieval ? Int(Double(self.info?.contextLength ?? 4096) * 0.12) : 0
            let after = await self.summarizeIfNeeded(c, conversationID: conversationID, settings: settings, reserve: recallReserve,
                                                     budgetFraction: Self.compactTargetFraction)
            DiagnosticsLog.append("[compact] done covers=\(after.summaryCoversMessages) cancelled=\(Task.isCancelled)")
        }
    }

    /// RAG over messages outside the live window and over full document text, injected into the latest user turn.
    private func recallIfUseful(_ conversation: Conversation, conversationID: UUID, settings: AppSettings) async -> Conversation {
        guard let app, settings.memoryRetrieval, let info, let lastIndex = conversation.messages.indices.last,
              conversation.messages[lastIndex].role == .user else { return conversation }
        var c = conversation
        let last = c.messages[lastIndex]
        activity = .retrieving
        let index = await updateMemoryIndex(c, conversationID: conversationID, upTo: lastIndex)
        let windowStart = c.summaryCoversMessages
        let truncatedDocMessages = Set(c.messages.enumerated().filter { $0.element.attachments.contains { $0.textTruncated } }.map { $0.element.id })
        let candidates = index.chunks.filter { ch in
            ch.messageID != last.id && (ch.messageIndex < windowStart || (ch.source != "message" && truncatedDocMessages.contains(ch.messageID)))
        }
        guard !candidates.isEmpty else { return c }
        let query = last.content.isEmpty ? last.attachments.map(\.fileName).joined(separator: " ") : last.content
        // Nothing to look up for "thanks" / "great, good to know": injecting old snippets there makes the model re-answer them.
        guard !Self.isSmallTalk(query) else { return c }
        let qEmb = (try? await engine.embed([query]))?.first ?? []
        let hits = index.search(query: query, queryEmbedding: qEmb, candidates: candidates, limit: 4, minScore: 0.35)
        let budgetChars = max(600, Int(Double(info.contextLength) * 0.12 * 3.5))
        let recall = MemoryIndex.renderRecall(hits, maxCharacters: budgetChars)
        c.messages[lastIndex].recalledContext = recall
        app.store.modify(conversationID, touch: false) { live in
            if let i = live.messages.firstIndex(where: { $0.id == last.id }) { live.messages[i].recalledContext = recall }
        }
        return c
    }

    // MARK: Generation

    enum GenerateMode { case reply, continuation }
    /// Automatic continuations after a reply stops at the window edge (each one re-plans the prompt so the oldest
    /// turns are summarized away and the reply keeps going).
    static let maxAutoContinuations = 3

    private func generate(conversationID: UUID, mode: GenerateMode = .reply, autoContinued: Int = 0) async {
        guard let app, var conversation = app.store.conversation(conversationID) else { return }
        let settings = app.settings
        isGenerating = true
        generatingConversationID = conversationID
        defer { activity = .idle }

        // 1. Park the previous chat's KV cache on disk, then restore this conversation's if we have one.
        if cacheConversationID != conversationID {
            await saveCurrentSnapshot()
            if settings.kvSnapshots, let key = modelKey, let url = snapshots.url(conversationID: conversationID, modelKey: key),
               await engine.loadState(from: url) {
                // cache restored
            } else {
                await engine.clearCache()
            }
            cacheConversationID = conversationID
        }

        // 2. Let a running background compaction finish, then summarize anything that still would not fit
        //    and recall relevant older context for this turn.
        if let pending = compactionTask {
            activity = .summarizing
            await pending.value
            if let fresh = app.store.conversation(conversationID) { conversation = fresh }
        }
        let recallReserve = settings.memoryRetrieval ? Int(Double(info?.contextLength ?? 4096) * 0.12) : 0
        conversation = await summarizeIfNeeded(conversation, conversationID: conversationID, settings: settings, reserve: recallReserve)
        if mode == .reply, Task.isCancelled == false {
            conversation = await recallIfUseful(conversation, conversationID: conversationID, settings: settings)
        }

        let (turns, _) = makeTurns(conversation, from: conversation.summaryCoversMessages, conversationID: conversationID, settings: settings)
        let system = systemPrompt(for: conversation, settings: settings)

        // The bubble being written: a new one, or the cut-short reply we are resuming.
        let target: Message
        let previousStats: GenerationStats?
        switch mode {
        case .reply:
            target = Message(role: .assistant, content: "")
            previousStats = nil
            app.store.modify(conversationID, touch: false) { $0.messages.append(target) }
        case .continuation:
            guard let last = conversation.messages.last, last.role == .assistant, turns.last?.role == .assistant else {
                isGenerating = false; generatingConversationID = nil; return
            }
            target = last
            previousStats = last.stats
        }
        activity = .generating

        let task = Task { @MainActor [weak self] in
            guard let self else { return }
            var splitter = ThinkTagSplitter()
            var content = mode == .continuation ? target.content : ""
            var reasoning = mode == .continuation ? (target.reasoning ?? "") : ""
            var lastFlush = Date.distantPast
            var finalStats: GenerationStats?
            var failure: String?

            let flush: (Bool) -> Void = { force in
                guard force || Date().timeIntervalSince(lastFlush) > 0.05 else { return }
                lastFlush = Date()
                let c = content, r = reasoning
                app.store.modify(conversationID, touch: false, persist: force) { conv in
                    guard let i = conv.messages.firstIndex(where: { $0.id == target.id }) else { return }
                    conv.messages[i].content = c
                    conv.messages[i].reasoning = r.isEmpty ? nil : r
                }
            }

            do {
                for try await event in self.engine.respond(systemPrompt: system, turns: turns, sampling: settings.sampling,
                                                           continuing: mode == .continuation) {
                    if Task.isCancelled { break }
                    switch event {
                    case .promptProcessed:
                        break
                    case .token(let s):
                        for part in splitter.feed(s) {
                            switch part {
                            case .reasoning(let r): reasoning += r
                            case .text(let t): content += t
                            }
                        }
                        flush(false)
                    case .finished(let stats):
                        finalStats = stats
                    }
                }
            } catch is CancellationError {
                // consumer cancelled
            } catch {
                failure = error.localizedDescription
            }
            for part in splitter.flush() {
                switch part {
                case .reasoning(let r): reasoning += r
                case .text(let t): content += t
                }
            }
            content = content.trimmingCharacters(in: .whitespacesAndNewlines)
            reasoning = reasoning.trimmingCharacters(in: .whitespacesAndNewlines)
            flush(true)
            var stats = finalStats
            if stats == nil, failure == nil {
                stats = GenerationStats(promptTokens: 0, cachedTokens: 0, generatedTokens: 0, prefillSeconds: 0, decodeSeconds: 0, stopReason: .cancelled)
            }
            stats?.modelID = self.loadedModel?.id
            if let st = stats {
                DiagnosticsLog.append("[turn\(mode == .continuation ? "+" : "")] model=\(self.loadedModel?.id ?? "-") ctx=\(st.contextLength ?? 0) prompt=\(st.promptTokens) cached=\(st.cachedTokens) gen=\(st.generatedTokens) stop=\(st.stopReason.rawValue) shifts=\(st.contextShifts ?? 0) tps=\(String(format: "%.1f", st.tokensPerSecond))")
            }
            if let failure { DiagnosticsLog.append("[turn-error] \(failure)") }
            // A resumed reply keeps one stats line: totals across the pieces, the latest stop reason.
            if var merged = stats, let prev = previousStats {
                merged.generatedTokens += prev.generatedTokens
                merged.decodeSeconds += prev.decodeSeconds
                merged.prefillSeconds += prev.prefillSeconds
                merged.contextShifts = ((merged.contextShifts ?? 0) + (prev.contextShifts ?? 0)).nonZero
                if merged.stopReason == .cancelled, merged.generatedTokens == prev.generatedTokens { merged = prev }
                stats = merged
            }
            let s = stats, f = failure
            app.store.modify(conversationID) { conv in
                guard let i = conv.messages.firstIndex(where: { $0.id == target.id }) else { return }
                conv.messages[i].stats = s
                conv.messages[i].error = f
            }
            self.isGenerating = false
            self.generatingConversationID = nil

            // 3. Index the new reply for later recall (the KV snapshot is written when the user leaves this chat).
            if settings.memoryRetrieval, let c = app.store.conversation(conversationID) {
                _ = await self.updateMemoryIndex(c, conversationID: conversationID, upTo: c.messages.count)
            }
            // 4. If the window is nearly full, fold the oldest turns into the summary now rather than before the next turn.
            if let st = stats, f == nil, !Task.isCancelled, st.stopReason != .contextFull {
                self.scheduleCompactionIfNeeded(conversationID: conversationID, stats: st, settings: settings)
            }
        }
        generationTask = task
        await task.value

        // 5. The reply ran into the window edge: summarize the oldest turns away and keep writing the same reply.
        if !task.isCancelled, autoContinued < Self.maxAutoContinuations,
           let last = app.store.conversation(conversationID)?.messages.last, last.id == target.id,
           last.error == nil, last.stats?.stopReason == .contextFull {
            DiagnosticsLog.append("[auto-continue] \(autoContinued + 1) after context-full stop")
            await generate(conversationID: conversationID, mode: .continuation, autoContinued: autoContinued + 1)
        }
    }
}

private extension Int {
    var nonZero: Int? { self == 0 ? nil : self }
}

/// KV-cache snapshots live in Caches/KVSnapshots/<modelKey>/<conversationID>.kv (+ .units sidecar).
struct SnapshotStore {
    let root: URL = {
        let caches = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
        return caches.appendingPathComponent("KVSnapshots", isDirectory: true)
    }()

    func url(conversationID: UUID, modelKey: String) -> URL? {
        root.appendingPathComponent(modelKey, isDirectory: true).appendingPathComponent("\(conversationID.uuidString).kv")
    }

    func delete(conversationID: UUID) {
        let fm = FileManager.default
        for dir in (try? fm.contentsOfDirectory(at: root, includingPropertiesForKeys: nil)) ?? [] {
            let base = dir.appendingPathComponent("\(conversationID.uuidString).kv")
            try? fm.removeItem(at: base)
            try? fm.removeItem(at: base.appendingPathExtension("units"))
        }
    }

    func pruneOtherModels(keeping key: String) {
        let fm = FileManager.default
        for dir in (try? fm.contentsOfDirectory(at: root, includingPropertiesForKeys: nil)) ?? [] where dir.lastPathComponent != key {
            try? fm.removeItem(at: dir)
        }
    }

    /// Keeps the `keep` most recently written snapshots for the model.
    func prune(keep: Int, modelKey: String) {
        let fm = FileManager.default
        let dir = root.appendingPathComponent(modelKey, isDirectory: true)
        let files = ((try? fm.contentsOfDirectory(at: dir, includingPropertiesForKeys: [.contentModificationDateKey])) ?? [])
            .filter { $0.pathExtension == "kv" }
            .sorted { ((try? $0.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast) >
                      ((try? $1.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast) }
        for f in files.dropFirst(keep) {
            try? fm.removeItem(at: f)
            try? fm.removeItem(at: f.appendingPathExtension("units"))
        }
    }
}
