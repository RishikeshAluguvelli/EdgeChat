import Foundation
import llama

/// Information about the currently loaded model.
public struct LoadedModelInfo: Sendable, Hashable {
    public let modelPath: String
    public let mmprojPath: String?
    public let name: String
    public let architectureDescription: String
    public let parameterCount: UInt64
    public let sizeBytes: UInt64
    public let trainingContext: Int
    public let contextLength: Int
    public let hasVision: Bool
    public let supportsThinkingToggle: Bool
    public let gpuOffloaded: Bool
    public let threads: Int
    /// Approximate KV-cache footprint per context token (depends on architecture and cache type).
    public let kvBytesPerToken: Int

    public func kvCacheBytes(contextLength: Int) -> Int { kvBytesPerToken * contextLength }
}

public enum LlamaLogLevel: Int, Sendable { case debug = 1, info = 2, warn = 3, error = 4 }

/// A single-sequence llama.cpp engine with KV-cache prefix reuse across turns.
///
/// All llama.cpp calls run on one serial queue; the public API is async. Only one
/// generation runs at a time and it can be cancelled by cancelling the consuming task.
public final class LlamaEngine: @unchecked Sendable {

    // MARK: Logging

    nonisolated(unsafe) private static var _logHandler: (@Sendable (LlamaLogLevel, String) -> Void)?
    private static let logLock = NSLock()
    public static var logHandler: (@Sendable (LlamaLogLevel, String) -> Void)? {
        get { logLock.withLock { _logHandler } }
        set { logLock.withLock { _logHandler = newValue } }
    }

    private static let backendInit: Void = {
        llama_backend_init()
        let cb: ggml_log_callback = { level, text, _ in
            guard let text else { return }
            let lvl = LlamaLogLevel(rawValue: Int(level.rawValue)) ?? .info
            let s = String(cString: text)
            if let h = LlamaEngine.logHandler { h(lvl, s) }
            else if lvl.rawValue >= LlamaLogLevel.warn.rawValue { print("[llama] \(s)", terminator: "") }
        }
        llama_log_set(cb, nil)
        mtmd_helper_log_set(cb, nil)
    }()

    // MARK: State (queue-confined unless noted)

    private let queue = DispatchQueue(label: "com.edgechat.llama", qos: .userInitiated)
    private var model: OpaquePointer?
    private var ctx: OpaquePointer?
    private var vocab: OpaquePointer?
    private var mctx: OpaquePointer?          // mtmd_context*
    private var batch = llama_batch()
    private var hasBatch = false
    private var nBatch: Int32 = 512
    private var config = EngineConfig()
    private var chatTemplate: String?
    /// Exactly what sequence 0 of the KV cache currently holds.
    private var cachedUnits: [PromptUnit] = []

    private let cancelFlag = CancelFlag()
    private let stateLock = NSLock()
    private var _info: LoadedModelInfo?
    private var _isGenerating = false

    public var info: LoadedModelInfo? { stateLock.withLock { _info } }
    public var isLoaded: Bool { info != nil }
    public var isGenerating: Bool { stateLock.withLock { _isGenerating } }

    /// Backend (and Metal shader library) initialization is deferred to the engine queue: on iPhone it can take
    /// several seconds and must never run on the main thread at launch.
    public init() {}

    deinit { unloadSync() }

    // MARK: - Loading

    public func load(modelPath: String, mmprojPath: String? = nil, config: EngineConfig,
                     progress: (@Sendable (Double) -> Void)? = nil) async throws -> LoadedModelInfo {
        try await run { try self.loadSync(modelPath: modelPath, mmprojPath: mmprojPath, config: config, progress: progress) }
    }

    public func unload() async {
        try? await run { self.unloadSync() }
    }

    /// Drops the KV cache (next turn re-evaluates the whole prompt).
    public func clearCache() async {
        try? await run {
            if let ctx = self.ctx { llama_memory_clear(llama_get_memory(ctx), true) }
            self.cachedUnits = []
        }
    }

    public func countTokens(_ text: String) async throws -> Int {
        try await run {
            guard self.vocab != nil else { throw EngineError.notLoaded }
            return try self.tokenize(text, addSpecial: false, parseSpecial: true).count
        }
    }

    private final class ProgressBox { let cb: (Double) -> Void; init(_ cb: @escaping (Double) -> Void) { self.cb = cb } }

    private func loadSync(modelPath: String, mmprojPath: String?, config: EngineConfig,
                          progress: (@Sendable (Double) -> Void)?) throws -> LoadedModelInfo {
        _ = Self.backendInit
        unloadSync()
        self.config = config

        #if targetEnvironment(simulator)
        let gpuLayers: Int32 = 0
        #else
        let gpuLayers: Int32 = llama_supports_gpu_offload() ? config.gpuLayers : 0
        #endif
        let threads = config.resolvedThreads

        var mparams = llama_model_default_params()
        mparams.n_gpu_layers = gpuLayers
        mparams.use_extra_bufts = true
        var box: Unmanaged<ProgressBox>?
        if let progress {
            let b = Unmanaged.passRetained(ProgressBox(progress))
            box = b
            mparams.progress_callback = { p, ud in
                guard let ud else { return true }
                Unmanaged<ProgressBox>.fromOpaque(ud).takeUnretainedValue().cb(Double(p))
                return true
            }
            mparams.progress_callback_user_data = b.toOpaque()
        }
        defer { box?.release() }

        guard let model = modelPath.withCString({ llama_model_load_from_file($0, mparams) }) else {
            throw EngineError.modelLoadFailed(modelPath)
        }
        self.model = model
        self.vocab = llama_model_get_vocab(model)

        let nCtxTrain = Int(llama_model_n_ctx_train(model))
        var cparams = llama_context_default_params()
        cparams.n_ctx = min(config.contextLength, UInt32(max(nCtxTrain, 2048)))
        cparams.n_batch = min(config.batchSize, cparams.n_ctx)
        cparams.n_ubatch = min(cparams.n_batch, 512)
        cparams.n_seq_max = 1
        cparams.n_threads = threads
        cparams.n_threads_batch = threads
        cparams.offload_kqv = gpuLayers > 0
        cparams.flash_attn_type = config.flashAttention ? LLAMA_FLASH_ATTN_TYPE_AUTO : LLAMA_FLASH_ATTN_TYPE_DISABLED
        if config.kvCacheQ8 {
            cparams.type_k = GGML_TYPE_Q8_0
            cparams.type_v = GGML_TYPE_Q8_0
            cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED
        }
        guard let ctx = llama_init_from_model(model, cparams) else {
            unloadSync()
            throw EngineError.contextCreateFailed
        }
        self.ctx = ctx
        self.nBatch = Int32(cparams.n_batch)
        self.batch = llama_batch_init(nBatch, 0, 1)
        self.hasBatch = true

        if let tmpl = llama_model_chat_template(model, nil) {
            chatTemplate = String(cString: tmpl)
        } else {
            chatTemplate = nil
        }

        var hasVision = false
        if let mmprojPath {
            var mp = mtmd_context_params_default()
            mp.use_gpu = gpuLayers > 0
            mp.n_threads = threads
            mp.print_timings = false
            mp.warmup = false
            mp.flash_attn_type = cparams.flash_attn_type
            guard let m = mmprojPath.withCString({ mtmd_init_from_file($0, model, mp) }) else {
                unloadSync()
                throw EngineError.mmprojLoadFailed(mmprojPath)
            }
            self.mctx = m
            hasVision = mtmd_support_vision(m)
        }

        var descBuf = [CChar](repeating: 0, count: 256)
        _ = llama_model_desc(model, &descBuf, descBuf.count)
        var nameBuf = [CChar](repeating: 0, count: 256)
        let n = llama_model_meta_val_str(model, "general.name", &nameBuf, nameBuf.count)
        let name = n > 0 ? String(cString: nameBuf) : URL(fileURLWithPath: modelPath).deletingPathExtension().lastPathComponent

        let nHead = max(1, Int(llama_model_n_head(model)))
        let headDim = Int(llama_model_n_embd(model)) / nHead
        let kvPerToken = Int(llama_model_n_layer(model)) * 2 * Int(llama_model_n_head_kv(model)) * headDim * (config.kvCacheQ8 ? 1 : 2)
        let info = LoadedModelInfo(
            modelPath: modelPath, mmprojPath: mmprojPath, name: name,
            architectureDescription: String(cString: descBuf),
            parameterCount: llama_model_n_params(model), sizeBytes: llama_model_size(model),
            trainingContext: nCtxTrain, contextLength: Int(llama_n_ctx(ctx)), hasVision: hasVision,
            supportsThinkingToggle: chatTemplate?.contains("enable_thinking") == true,
            gpuOffloaded: gpuLayers > 0, threads: Int(threads), kvBytesPerToken: kvPerToken)
        stateLock.withLock { _info = info }
        return info
    }

    private func unloadSync() {
        _ = Self.backendInit
        if hasBatch { llama_batch_free(batch); hasBatch = false }
        if let m = mctx { mtmd_free(m); mctx = nil }
        if let c = ctx { llama_free(c); ctx = nil }
        if let m = model { llama_model_free(m); model = nil }
        vocab = nil
        chatTemplate = nil
        cachedUnits = []
        stateLock.withLock { _info = nil }
    }

    // MARK: - Planning, summaries, snapshots

    /// KV cells available to the prompt (context minus the reply reserve) — what `respond` fits the transcript into.
    public func promptBudget(sampling: SamplingConfig) async throws -> Int {
        try await run {
            guard let ctx = self.ctx else { throw EngineError.notLoaded }
            let nCtx = Int(llama_n_ctx(ctx))
            return nCtx - Self.replyReserve(contextLength: nCtx, sampling: sampling) - 8
        }
    }

    /// How many leading turns `respond` would drop to fit the context budget (0 = everything fits).
    /// `budgetFraction` < 1 plans a tighter fit, e.g. 0.5 to compact a chat down to half the window ahead of time.
    public func planTruncation(systemPrompt: String?, turns: [ChatTurn], sampling: SamplingConfig, budgetFraction: Double = 1) async throws -> Int {
        try await run {
            guard let ctx = self.ctx else { throw EngineError.notLoaded }
            let nCtx = Int(llama_n_ctx(ctx))
            let full = nCtx - Self.replyReserve(contextLength: nCtx, sampling: sampling) - 8
            guard full >= 128 else { throw EngineError.contextTooSmall(nCtx) }
            let budget = max(128, Int(Double(full) * min(1, max(0.1, budgetFraction))))
            return try self.truncatedHistory(systemPrompt: systemPrompt, turns: turns, budget: budget).dropped
        }
    }

    /// Uses the loaded model to fold `turns` into a running summary (the "memory" of dropped history).
    public func summarize(previousSummary: String?, turns: [ChatTurn], maxWords: Int = 220) async throws -> String {
        var transcript = ""
        for t in turns {
            let images = t.images.isEmpty ? "" : " [\(t.images.count) image(s) attached]"
            transcript += "\(t.role == .user ? "User" : "Assistant")\(images): \(t.text)\n\n"
        }
        var prompt = "You maintain the memory of a conversation between a user and an assistant so it can continue after older messages are removed.\n\n"
        if let previousSummary, !previousSummary.isEmpty {
            prompt += "Existing memory:\n\(previousSummary)\n\n"
        }
        prompt += "New messages to fold into the memory:\n\(transcript)"
        prompt += """
Write the updated memory in at most \(maxWords) words, as plain bullets under these headings (skip a heading with nothing to say):
About the user: name, role, preferences, goals they stated.
Topics so far: each topic in order, with the concrete facts, numbers, names and conclusions that were given.
Decisions and recommendations: what was agreed or advised.
Open threads: what the user asked most recently, and anything left unanswered.
Keep exact names, numbers, identifiers and quoted wording. Do not add information that is not in the messages. Output only the memory.
"""
        var sampling = SamplingConfig()
        sampling.temperature = 0.3
        sampling.maxTokens = maxWords * 2 + 64
        var splitter = ThinkTagSplitter()
        var out = ""
        for try await event in respond(systemPrompt: nil, turns: [ChatTurn(role: .user, text: prompt)], sampling: sampling) {
            if case .token(let s) = event {
                for part in splitter.feed(s) { if case .text(let t) = part { out += t } }
            }
        }
        for part in splitter.flush() { if case .text(let t) = part { out += t } }
        let trimmed = out.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { throw EngineError.summaryFailed }
        return String(trimmed.prefix(maxWords * 8))
    }

    /// Snapshot container: our own header around llama's sequence-state bytes. We deliberately avoid
    /// `llama_state_seq_save_file/load_file`: the file loader hard-asserts (abort, no error) when the stored byte
    /// count differs from what the current context reads back, which killed the app on every send in a chat whose
    /// snapshot no longer matched. The buffer API returns 0 on any mismatch instead.
    private static let snapshotMagic: UInt32 = 0x4543_4B56   // "ECKV"
    private static let snapshotVersion: UInt32 = 2
    private static let snapshotHeaderSize = 4 + 4 + 4 + 4 + 8  // magic, version, n_ctx, cells, payload bytes

    /// Saves the KV cache (sequence 0) plus its unit map so a conversation can resume without re-prefill.
    @discardableResult
    public func saveState(to url: URL) async throws -> Int {
        try await run {
            guard let ctx = self.ctx else { throw EngineError.notLoaded }
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            let size = llama_state_seq_get_size(ctx, 0)
            guard size > 0 else { throw EngineError.stateFailed }
            var payload = [UInt8](repeating: 0, count: size)
            let written = payload.withUnsafeMutableBufferPointer { llama_state_seq_get_data(ctx, $0.baseAddress, size, 0) }
            guard written > 0, written <= size else { throw EngineError.stateFailed }
            let cells = self.cachedUnits.reduce(Int32(0)) { $0 + $1.cells }
            var data = Data(capacity: Self.snapshotHeaderSize + written)
            func put<T: FixedWidthInteger>(_ v: T) { var le = v.littleEndian; withUnsafeBytes(of: &le) { data.append(contentsOf: $0) } }
            put(Self.snapshotMagic); put(Self.snapshotVersion); put(llama_n_ctx(ctx)); put(UInt32(max(0, cells))); put(UInt64(written))
            data.append(contentsOf: payload[0..<written])
            let units = self.cachedUnits.map { u -> String in
                switch u { case .token(let t): return "t:\(t)"; case .image(let id, let p, let c): return "i:\(p):\(c):\(id)" }
            }
            // Units first, then the state: a kill between the two leaves a state file without a matching sidecar,
            // which the loader rejects. Both writes are atomic (temp file + rename).
            try JSONEncoder().encode(units).write(to: url.appendingPathExtension("units"), options: .atomic)
            try data.write(to: url, options: .atomic)
            return written
        }
    }

    /// Restores a snapshot made by `saveState` for the same model and context settings. Returns false (and
    /// leaves the cache empty) if the file is missing, damaged or incompatible; such files are deleted.
    public func loadState(from url: URL) async -> Bool {
        let ok = (try? await run { () -> Bool in
            guard let ctx = self.ctx else { return false }
            let mem = llama_get_memory(ctx)
            llama_memory_clear(mem, true)
            self.cachedUnits = []
            guard let data = try? Data(contentsOf: url), data.count > Self.snapshotHeaderSize,
                  let sidecar = try? Data(contentsOf: url.appendingPathExtension("units")),
                  let names = try? JSONDecoder().decode([String].self, from: sidecar) else { return false }
            var units: [PromptUnit] = []
            for n in names {
                if n.hasPrefix("t:"), let t = Int32(n.dropFirst(2)) { units.append(.token(t)) }
                else if n.hasPrefix("i:") {
                    let parts = n.dropFirst(2).split(separator: ":", maxSplits: 2)
                    guard parts.count == 3, let p = Int32(parts[0]), let c = Int32(parts[1]) else { return false }
                    units.append(.image(id: String(parts[2]), positions: p, cells: c))
                } else { return false }
            }
            // Header checks: ours, current version, same context size, unit map matches the stored cell count.
            var offset = 0
            func get<T: FixedWidthInteger>(_: T.Type) -> T {
                let v = data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: offset, as: T.self) }
                offset += MemoryLayout<T>.size
                return T(littleEndian: v)
            }
            let magic = get(UInt32.self), version = get(UInt32.self), nCtx = get(UInt32.self), cells = get(UInt32.self), length = get(UInt64.self)
            guard magic == Self.snapshotMagic, version == Self.snapshotVersion, nCtx == llama_n_ctx(ctx),
                  Int(length) == data.count - Self.snapshotHeaderSize,
                  Int32(cells) == units.reduce(Int32(0), { $0 + $1.cells }), cells > 0, cells <= nCtx else { return false }
            let read = data.withUnsafeBytes { raw -> Int in
                llama_state_seq_set_data(ctx, raw.baseAddress!.advanced(by: Self.snapshotHeaderSize).assumingMemoryBound(to: UInt8.self), Int(length), 0)
            }
            guard read == Int(length) else { llama_memory_clear(mem, true); return false }
            let positions = units.reduce(Int32(0)) { $0 + $1.positions }
            guard llama_memory_seq_pos_max(mem, 0) == positions - 1 else { llama_memory_clear(mem, true); return false }
            self.cachedUnits = units
            return true
        }) ?? false
        if !ok {
            try? FileManager.default.removeItem(at: url)
            try? FileManager.default.removeItem(at: url.appendingPathExtension("units"))
        }
        return ok
    }

    // MARK: - Embeddings (for conversation memory retrieval)

    /// Mean-pooled, L2-normalized embeddings from the loaded model. A temporary embedding context is created on
    /// the shared weights and released afterwards, so the cost is a short prefill per text.
    public func embed(_ texts: [String], maxTokens: Int = 384) async throws -> [[Float]] {
        try await run {
            guard let model = self.model, let vocab = self.vocab else { throw EngineError.notLoaded }
            var cp = llama_context_default_params()
            cp.n_ctx = UInt32(maxTokens + 8)
            cp.n_batch = UInt32(maxTokens + 8)
            cp.n_ubatch = UInt32(maxTokens + 8)
            cp.n_seq_max = 1
            cp.n_threads = self.config.resolvedThreads
            cp.n_threads_batch = self.config.resolvedThreads
            cp.embeddings = true
            cp.pooling_type = LLAMA_POOLING_TYPE_MEAN
            cp.offload_kqv = self.info?.gpuOffloaded ?? false
            cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO
            guard let ectx = llama_init_from_model(model, cp) else { throw EngineError.contextCreateFailed }
            defer { llama_free(ectx) }
            let nEmbd = Int(llama_model_n_embd(model))
            var batch = llama_batch_init(Int32(maxTokens + 8), 0, 1)
            defer { llama_batch_free(batch) }
            var out: [[Float]] = []
            for text in texts {
                var tokens = try self.tokenize(text, addSpecial: true, parseSpecial: false)
                if tokens.count > maxTokens { tokens = Array(tokens.prefix(maxTokens)) }
                guard !tokens.isEmpty else { out.append([]); continue }
                llama_memory_clear(llama_get_memory(ectx), true)
                batch.n_tokens = Int32(tokens.count)
                for (i, t) in tokens.enumerated() {
                    batch.token[i] = t; batch.pos[i] = Int32(i); batch.n_seq_id[i] = 1; batch.seq_id[i]![0] = 0; batch.logits[i] = 1
                }
                guard llama_decode(ectx, batch) == 0, let e = llama_get_embeddings_seq(ectx, 0) else { out.append([]); continue }
                var v = Array(UnsafeBufferPointer(start: e, count: nEmbd))
                let norm = v.reduce(Float(0)) { $0 + $1 * $1 }.squareRoot()
                if norm > 0 { for i in 0..<v.count { v[i] /= norm } }
                out.append(v)
            }
            _ = vocab
            return out
        }
    }

    // MARK: - Generation

    /// Generates the assistant reply for `turns` (oldest first, last must be a user turn).
    /// Older turns are dropped automatically when the prompt does not fit the context budget.
    /// With `continuing`, the last turn is a partial assistant reply and generation resumes right after its text
    /// (used for "Continue" after a reply hit its length limit or the window edge).
    public func respond(systemPrompt: String?, turns: [ChatTurn], sampling: SamplingConfig, continuing: Bool = false) -> AsyncThrowingStream<GenerationEvent, Error> {
        AsyncThrowingStream { continuation in
            let started = self.stateLock.withLock { () -> Bool in
                if self._isGenerating { return false }
                self._isGenerating = true
                return true
            }
            guard started else { continuation.finish(throwing: EngineError.busy); return }
            self.cancelFlag.reset()
            continuation.onTermination = { @Sendable _ in self.cancelFlag.cancel() }
            self.queue.async {
                defer { self.stateLock.withLock { self._isGenerating = false } }
                do {
                    try self.respondSync(systemPrompt: systemPrompt, turns: turns, sampling: sampling, continuing: continuing) { continuation.yield($0) }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    private func respondSync(systemPrompt: String?, turns: [ChatTurn], sampling: SamplingConfig, continuing: Bool = false,
                             emit: (GenerationEvent) -> Void) throws {
        guard let ctx, let vocab else { throw EngineError.notLoaded }
        let nCtx = Int(llama_n_ctx(ctx))
        let budget = nCtx - Self.replyReserve(contextLength: nCtx, sampling: sampling) - 8
        guard budget >= 128 else { throw EngineError.contextTooSmall(nCtx) }
        if continuing { guard turns.last?.role == .assistant else { throw EngineError.tokenizeFailed } }

        // 1. Build the prompt, dropping the oldest turns until it fits the budget.
        let (prepared, _, dropped) = try truncatedHistory(systemPrompt: systemPrompt, turns: turns, budget: budget, continuing: continuing)

        // 2. Reuse the longest common prefix already in the KV cache.
        let units = prepared.units
        guard !units.isEmpty else { throw EngineError.tokenizeFailed }
        var matched = 0
        while matched < units.count && matched < cachedUnits.count && units[matched] == cachedUnits[matched] { matched += 1 }
        if matched == units.count { matched -= 1 } // always re-decode the last token to get logits
        let nPastMatched = units[0..<matched].reduce(Int32(0)) { $0 + $1.positions }
        let cachedCells = Int(units[0..<matched].reduce(Int32(0)) { $0 + $1.cells })
        cachedUnits.removeLast(cachedUnits.count - matched)
        let mem = llama_get_memory(ctx)
        _ = llama_memory_seq_rm(mem, 0, nPastMatched, -1)

        // 3. Evaluate the remaining prompt.
        let tPrefill = DispatchTime.now()
        var nPast = nPastMatched
        try evaluate(prepared, matched: matched, nPast: &nPast)
        var nCells = Int(cachedUnits.reduce(Int32(0)) { $0 + $1.cells })
        let prefillSeconds = seconds(since: tPrefill)
        emit(.promptProcessed(promptTokens: prepared.tokenCount, cachedTokens: cachedCells, droppedTurns: dropped, seconds: prefillSeconds))

        // 4. Sample tokens. "Unlimited" still has a ceiling (two windows, at least 4k) so a looping model cannot
        //    shift the context forever; the app offers Continue when it is reached.
        let replyCap = sampling.maxTokens > 0 ? sampling.maxTokens : max(4096, nCtx * 2)
        let sampler = makeSampler(sampling, vocab: vocab)
        defer { llama_sampler_free(sampler) }
        var acc = UTF8Accumulator()
        var generated = 0
        var shifts = 0
        var reason: StopReason = .eos
        let tDecode = DispatchTime.now()
        while true {
            if cancelFlag.isCancelled { reason = .cancelled; break }
            let tok = llama_sampler_sample(sampler, ctx, -1)
            if llama_vocab_is_eog(vocab, tok) { reason = .eos; break }
            if let s = acc.append(pieceBytes(tok, vocab: vocab)) { emit(.token(s)) }
            generated += 1
            if generated >= replyCap { reason = .maxTokens; break }
            if nCells + 1 >= nCtx {
                if config.contextShift, let shifted = shiftContext(nPast: nPast, keep: prepared.systemTokenCount) {
                    nPast = shifted
                    nCells = Int(cachedUnits.reduce(Int32(0)) { $0 + $1.cells })
                    shifts += 1
                } else {
                    reason = .contextFull
                    break
                }
            }
            do {
                try decodeTokens([tok], startPos: nPast, logitsLast: true)
            } catch EngineError.cancelled {
                reason = .cancelled; break
            } catch EngineError.contextFull {
                // The KV cache ran out of room unexpectedly: shift and retry once, otherwise end the reply cleanly.
                if config.contextShift, let shifted = shiftContext(nPast: nPast, keep: prepared.systemTokenCount),
                   (try? decodeTokens([tok], startPos: shifted, logitsLast: true)) != nil {
                    nPast = shifted
                    nCells = Int(cachedUnits.reduce(Int32(0)) { $0 + $1.cells })
                    shifts += 1
                } else {
                    reason = .contextFull
                    break
                }
            }
            nPast += 1
            nCells += 1
        }
        if let s = acc.flush() { emit(.token(s)) }
        var stats = GenerationStats(promptTokens: prepared.tokenCount, cachedTokens: cachedCells,
                                    generatedTokens: generated, prefillSeconds: prefillSeconds,
                                    decodeSeconds: seconds(since: tDecode), stopReason: reason,
                                    contextShifts: shifts > 0 ? shifts : nil)
        stats.contextLength = nCtx
        emit(.finished(stats))
    }

    /// Positions held back for the reply: the full `maxTokens` up to a quarter of the context (at least 128).
    /// Replies that run past this rely on context shifting (or stop with `.contextFull` when shifting is off).
    public static func replyReserve(contextLength: Int, sampling: SamplingConfig) -> Int {
        min(sampling.effectiveMaxTokens, max(128, contextLength / 4))
    }

    /// Drops the oldest turns (pairs first) until the formatted prompt fits `budget` positions.
    private func truncatedHistory(systemPrompt: String?, turns: [ChatTurn], budget: Int, continuing: Bool = false) throws -> (prepared: PreparedPrompt, history: [ChatTurn], dropped: Int) {
        var history = turns
        var dropped = 0
        while true {
            let prepared = try buildPrompt(systemPrompt: systemPrompt, turns: history, continuing: continuing)
            if prepared.cellCount <= budget { return (prepared, history, dropped) }
            if history.count > 1 {
                let n = min(2, history.count - 1)
                history.removeFirst(n)
                dropped += n
                continue
            }
            // A single over-long turn (big document, or a very long partial reply being continued): shrink its text.
            let text = history[0].text
            guard text.count > 512 else { throw EngineError.promptTooLong }
            let ratio = Double(budget) / Double(prepared.cellCount) * 0.9
            let keep = Int(Double(text.count) * ratio)
            if continuing {
                // Keep the tail of the partial reply so the model resumes mid-thought; the head is dropped.
                history[0].text = String(text.suffix(keep))
            } else {
                history[0].text = TextUtils.truncateMiddle(text, maxCharacters: keep).text
            }
        }
    }

    /// Frees room mid-generation by discarding the oldest half of the non-system tokens (llama.cpp K-shift).
    /// Only for pure-text caches; returns the new `nPast`, or nil if shifting is not possible.
    private func shiftContext(nPast: Int32, keep: Int32) -> Int32? {
        guard let ctx else { return nil }
        guard cachedUnits.count == Int(nPast),
              !cachedUnits.contains(where: { if case .image = $0 { return true } else { return false } }) else { return nil }
        let nKeep = min(max(0, keep), nPast / 2)
        let nDiscard = (nPast - nKeep) / 2
        guard nDiscard > 0 else { return nil }
        let mem = llama_get_memory(ctx)
        guard llama_memory_seq_rm(mem, 0, nKeep, nKeep + nDiscard) else { return nil }
        llama_memory_seq_add(mem, 0, nKeep + nDiscard, nPast, -nDiscard)
        cachedUnits.removeSubrange(Int(nKeep)..<Int(nKeep + nDiscard))
        return nPast - nDiscard
    }

    // MARK: Prompt construction

    enum PromptUnit: Equatable {
        case token(llama_token)
        /// `positions` = RoPE positions consumed (small for M-RoPE images), `cells` = KV-cache cells (= image tokens).
        case image(id: String, positions: Int32, cells: Int32)
        var positions: Int32 {
            switch self { case .token: return 1; case .image(_, let p, _): return p }
        }
        var cells: Int32 {
            switch self { case .token: return 1; case .image(_, _, let c): return c }
        }
    }

    /// Tokenized prompt plus the mtmd resources that must outlive evaluation.
    final class PreparedPrompt {
        var units: [PromptUnit] = []
        var tokens: [llama_token] = []          // text-only path
        var chunks: OpaquePointer?              // mtmd_input_chunks* (multimodal path)
        var bitmaps: [OpaquePointer?] = []
        var tokenCount = 0
        var positionCount = 0
        /// KV-cache cells the prompt occupies (image tokens count fully, unlike positions under M-RoPE).
        var cellCount: Int { tokenCount }
        /// Tokens at the start of the prompt that belong to the system message (kept across context shifts).
        var systemTokenCount: Int32 = 0
        deinit {
            if let chunks { mtmd_input_chunks_free(chunks) }
            for b in bitmaps { if let b { mtmd_bitmap_free(b) } }
        }
    }

    /// Marker appended to a partial assistant reply so the formatted prompt can be cut right after the reply text
    /// (whatever the template would add to close the turn is dropped).
    private static let continueSentinel = "\u{1}\u{2}EDGECHAT_CONTINUE\u{2}\u{1}"

    private func buildPrompt(systemPrompt: String?, turns: [ChatTurn], continuing: Bool = false) throws -> PreparedPrompt {
        let prepared = PreparedPrompt()
        var messages: [(role: String, content: String)] = []
        if let systemPrompt, !systemPrompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            messages.append(("system", systemPrompt))
        }
        let marker = mctx != nil ? String(cString: mtmd_default_marker()) : nil
        for turn in turns {
            var content = turn.text
            if let marker, !turn.images.isEmpty {
                content = turn.images.map { _ in marker }.joined(separator: "\n") + "\n" + turn.text
                for image in turn.images {
                    let bitmap = image.rgb.withUnsafeBytes { raw -> OpaquePointer? in
                        mtmd_bitmap_init(UInt32(image.width), UInt32(image.height),
                                         raw.baseAddress!.assumingMemoryBound(to: UInt8.self))
                    }
                    if let bitmap { mtmd_bitmap_set_id(bitmap, image.id) }
                    prepared.bitmaps.append(bitmap)
                }
            }
            messages.append((turn.role.rawValue, content))
        }
        let thinkOff = config.disableThinking && chatTemplate?.contains("enable_thinking") == true
        var text: String
        if continuing, let last = messages.last, last.role == Role.assistant.rawValue {
            // Resume a partial reply: format it as the final assistant message and cut at the sentinel so the
            // model continues the same turn instead of starting a new one.
            var partial = last.content
            if thinkOff { partial = "<think>\n\n</think>\n\n" + partial }
            messages[messages.count - 1].content = partial + Self.continueSentinel
            let formatted = try applyChatTemplate(messages, addAssistant: false)
            guard let cut = formatted.range(of: Self.continueSentinel) else { throw EngineError.templateFailed }
            text = String(formatted[..<cut.lowerBound])
        } else {
            text = try applyChatTemplate(messages, addAssistant: true)
            if thinkOff { text += "<think>\n\n</think>\n\n" }
        }

        if let mctx {
            guard let chunks = mtmd_input_chunks_init() else { throw EngineError.tokenizeFailed }
            prepared.chunks = chunks
            let rc: Int32 = text.withCString { cstr in
                var input = mtmd_input_text(text: cstr, text_len: strlen(cstr), add_special: true, parse_special: true)
                return prepared.bitmaps.withUnsafeBufferPointer { bp in
                    mtmd_tokenize(mctx, chunks, &input, bp.baseAddress, bp.count)
                }
            }
            guard rc == 0 else { throw EngineError.tokenizeFailed }
            let n = mtmd_input_chunks_size(chunks)
            for i in 0..<n {
                guard let chunk = mtmd_input_chunks_get(chunks, i) else { continue }
                switch mtmd_input_chunk_get_type(chunk) {
                case MTMD_INPUT_CHUNK_TYPE_TEXT:
                    var count = 0
                    if let toks = mtmd_input_chunk_get_tokens_text(chunk, &count) {
                        for j in 0..<count { prepared.units.append(.token(toks[j])) }
                    }
                case MTMD_INPUT_CHUNK_TYPE_IMAGE:
                    let id = mtmd_input_chunk_get_id(chunk).map { String(cString: $0) } ?? "img-\(i)"
                    prepared.units.append(.image(id: id, positions: mtmd_input_chunk_get_n_pos(chunk),
                                                 cells: Int32(mtmd_input_chunk_get_n_tokens(chunk))))
                default:
                    throw EngineError.tokenizeFailed
                }
            }
            prepared.tokenCount = Int(mtmd_helper_get_n_tokens(chunks))
            prepared.positionCount = Int(mtmd_helper_get_n_pos(chunks))
        } else {
            let tokens = try tokenize(text, addSpecial: true, parseSpecial: true)
            prepared.tokens = tokens
            prepared.units = tokens.map { .token($0) }
            prepared.tokenCount = tokens.count
            prepared.positionCount = tokens.count
            if let first = messages.first, first.role == "system",
               let sysText = try? applyChatTemplate([first], addAssistant: false),
               let sysTokens = try? tokenize(sysText, addSpecial: true, parseSpecial: true),
               sysTokens.count < tokens.count, Array(tokens[0..<sysTokens.count]) == sysTokens {
                prepared.systemTokenCount = Int32(sysTokens.count)
            }
        }
        return prepared
    }

    /// Decodes everything after the matched prefix, keeping `cachedUnits` in sync with the KV cache.
    private func evaluate(_ prepared: PreparedPrompt, matched: Int, nPast: inout Int32) throws {
        guard let chunks = prepared.chunks else {
            let tail = Array(prepared.tokens[matched...])
            try decodeTokens(tail, startPos: nPast, logitsLast: true)
            nPast += Int32(tail.count)
            return
        }
        guard let ctx, let mctx else { throw EngineError.notLoaded }
        let n = mtmd_input_chunks_size(chunks)
        var unitIndex = 0
        for i in 0..<n {
            guard let chunk = mtmd_input_chunks_get(chunks, i) else { continue }
            let isLast = i == n - 1
            switch mtmd_input_chunk_get_type(chunk) {
            case MTMD_INPUT_CHUNK_TYPE_TEXT:
                var count = 0
                let toks = mtmd_input_chunk_get_tokens_text(chunk, &count)
                if unitIndex + count <= matched { unitIndex += count; continue }
                let skip = max(matched - unitIndex, 0)
                var tail: [llama_token] = []
                if let toks { tail = Array(UnsafeBufferPointer(start: toks + skip, count: count - skip)) }
                try decodeTokens(tail, startPos: nPast, logitsLast: isLast)
                nPast += Int32(tail.count)
                unitIndex += count
            case MTMD_INPUT_CHUNK_TYPE_IMAGE:
                if unitIndex < matched { unitIndex += 1; continue }
                if cancelFlag.isCancelled { throw EngineError.cancelled }
                var newPast: llama_pos = nPast
                let rc = mtmd_helper_eval_chunk_single(mctx, ctx, chunk, nPast, 0, nBatch, isLast, &newPast)
                guard rc == 0 else { throw EngineError.imageEncodeFailed(rc) }
                let id = mtmd_input_chunk_get_id(chunk).map { String(cString: $0) } ?? "img-\(i)"
                cachedUnits.append(.image(id: id, positions: mtmd_input_chunk_get_n_pos(chunk),
                                          cells: Int32(mtmd_input_chunk_get_n_tokens(chunk))))
                nPast = newPast
                unitIndex += 1
            default:
                throw EngineError.tokenizeFailed
            }
        }
    }

    private func decodeTokens(_ tokens: [llama_token], startPos: Int32, logitsLast: Bool) throws {
        guard let ctx else { throw EngineError.notLoaded }
        var pos = startPos
        var i = 0
        while i < tokens.count {
            if cancelFlag.isCancelled { throw EngineError.cancelled }
            let n = min(Int(nBatch), tokens.count - i)
            batch.n_tokens = Int32(n)
            for j in 0..<n {
                batch.token[j] = tokens[i + j]
                batch.pos[j] = pos + Int32(j)
                batch.n_seq_id[j] = 1
                batch.seq_id[j]![0] = 0
                batch.logits[j] = 0
            }
            if logitsLast && i + n == tokens.count { batch.logits[n - 1] = 1 }
            let rc = llama_decode(ctx, batch)
            guard rc == 0 else {
                Self.logHandler?(.warn, "llama_decode returned \(rc) at pos \(pos) (n=\(n), cached units \(cachedUnits.count), n_ctx \(llama_n_ctx(ctx)))\n")
                throw rc == 1 ? EngineError.contextFull : EngineError.decodeFailed(rc)
            }
            for j in 0..<n { cachedUnits.append(.token(tokens[i + j])) }
            pos += Int32(n)
            i += n
        }
    }

    // MARK: Sampling

    private func makeSampler(_ s: SamplingConfig, vocab: OpaquePointer) -> UnsafeMutablePointer<llama_sampler> {
        var p = llama_sampler_chain_default_params()
        p.no_perf = true
        let chain = llama_sampler_chain_init(p)!
        if s.repeatPenalty != 1.0 && s.repeatLastN != 0 {
            llama_sampler_chain_add(chain, llama_sampler_init_penalties(llama_vocab_n_tokens(vocab), s.repeatLastN, s.repeatPenalty, 0, 0))
        }
        if s.temperature <= 0 {
            llama_sampler_chain_add(chain, llama_sampler_init_greedy())
        } else {
            if s.topK > 0 { llama_sampler_chain_add(chain, llama_sampler_init_top_k(s.topK)) }
            if s.topP < 1 { llama_sampler_chain_add(chain, llama_sampler_init_top_p(s.topP, 1)) }
            if s.minP > 0 { llama_sampler_chain_add(chain, llama_sampler_init_min_p(s.minP, 1)) }
            llama_sampler_chain_add(chain, llama_sampler_init_temp(s.temperature))
            llama_sampler_chain_add(chain, llama_sampler_init_dist(s.seed))
        }
        return chain
    }

    // MARK: Tokenizer / template helpers

    private func tokenize(_ text: String, addSpecial: Bool, parseSpecial: Bool) throws -> [llama_token] {
        guard let vocab else { throw EngineError.notLoaded }
        let utf8Count = Int32(text.utf8.count)
        var tokens = [llama_token](repeating: 0, count: Int(utf8Count) + 16)
        var n = text.withCString { llama_tokenize(vocab, $0, utf8Count, &tokens, Int32(tokens.count), addSpecial, parseSpecial) }
        if n < 0 {
            tokens = [llama_token](repeating: 0, count: Int(-n))
            n = text.withCString { llama_tokenize(vocab, $0, utf8Count, &tokens, Int32(tokens.count), addSpecial, parseSpecial) }
        }
        guard n >= 0 else { throw EngineError.tokenizeFailed }
        return Array(tokens[0..<Int(n)])
    }

    private func pieceBytes(_ token: llama_token, vocab: OpaquePointer) -> [UInt8] {
        var buf = [CChar](repeating: 0, count: 128)
        var n = llama_token_to_piece(vocab, token, &buf, Int32(buf.count), 0, true)
        if n < 0 {
            buf = [CChar](repeating: 0, count: Int(-n))
            n = llama_token_to_piece(vocab, token, &buf, Int32(buf.count), 0, true)
        }
        guard n > 0 else { return [] }
        return buf[0..<Int(n)].map { UInt8(bitPattern: $0) }
    }

    private func applyChatTemplate(_ messages: [(role: String, content: String)], addAssistant: Bool) throws -> String {
        var cStrings: [UnsafeMutablePointer<CChar>] = []
        defer { cStrings.forEach { free($0) } }
        var cMessages: [llama_chat_message] = []
        for m in messages {
            let r = strdup(m.role)!, c = strdup(m.content)!
            cStrings.append(r); cStrings.append(c)
            cMessages.append(llama_chat_message(role: UnsafePointer(r), content: UnsafePointer(c)))
        }
        func apply(_ template: String, size: Int) -> (Int32, [CChar]) {
            var buf = [CChar](repeating: 0, count: size)
            let n = template.withCString { t in
                cMessages.withUnsafeBufferPointer { mp in
                    llama_chat_apply_template(t, mp.baseAddress, mp.count, addAssistant, &buf, Int32(size))
                }
            }
            return (n, buf)
        }
        var size = messages.reduce(0) { $0 + $1.content.utf8.count + $1.role.utf8.count } * 2 + 2048
        var template = chatTemplate ?? "chatml"
        var (n, buf) = apply(template, size: size)
        if n < 0 {
            // Unknown template → fall back to ChatML (works for most instruct models).
            template = "chatml"
            (n, buf) = apply(template, size: size)
        }
        if n > Int32(size) {
            size = Int(n) + 1
            (n, buf) = apply(template, size: size)
        }
        guard n >= 0 else { throw EngineError.templateFailed }
        return String(decoding: buf[0..<Int(n)].map { UInt8(bitPattern: $0) }, as: UTF8.self)
    }

    // MARK: Utilities

    private func run<T>(_ body: @escaping () throws -> T) async throws -> T {
        try await withCheckedThrowingContinuation { cont in
            queue.async {
                do { cont.resume(returning: try body()) } catch { cont.resume(throwing: error) }
            }
        }
    }

    private func seconds(since t: DispatchTime) -> Double {
        Double(DispatchTime.now().uptimeNanoseconds - t.uptimeNanoseconds) / 1e9
    }
}

final class CancelFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var flag = false
    var isCancelled: Bool { lock.withLock { flag } }
    func cancel() { lock.withLock { flag = true } }
    func reset() { lock.withLock { flag = false } }
}
