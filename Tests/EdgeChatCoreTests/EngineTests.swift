import XCTest
import CoreGraphics
@testable import EdgeChatCore

/// Runs only when EDGECHAT_TEST_MODEL (and optionally EDGECHAT_TEST_MMPROJ + EDGECHAT_TEST_VLM) point at GGUF files.
final class EngineTests: XCTestCase {
    static let env = ProcessInfo.processInfo.environment
    static var modelPath: String? { env["EDGECHAT_TEST_MODEL"] }

    private func collect(_ engine: LlamaEngine, system: String?, turns: [ChatTurn], maxTokens: Int = 64, continuing: Bool = false) async throws -> (String, GenerationStats, Int) {
        var sampling = SamplingConfig()
        sampling.maxTokens = maxTokens
        sampling.temperature = 0 // greedy for determinism
        var text = ""
        var stats: GenerationStats?
        var cached = -1
        for try await ev in engine.respond(systemPrompt: system, turns: turns, sampling: sampling, continuing: continuing) {
            switch ev {
            case .promptProcessed(_, let c, _, _): cached = c
            case .token(let t): text += t
            case .finished(let s): stats = s
            }
        }
        return (text, try XCTUnwrap(stats), cached)
    }

    func testMultiTurnReusesCache() async throws {
        guard let path = Self.modelPath else { throw XCTSkip("EDGECHAT_TEST_MODEL not set") }
        let engine = LlamaEngine()
        var config = EngineConfig()
        config.contextLength = 2048
        let info = try await engine.load(modelPath: path, config: config)
        XCTAssertGreaterThan(info.parameterCount, 0)

        var turns = [ChatTurn(role: .user, text: "Reply with exactly the word: pineapple")]
        let (r1, s1, cached1) = try await collect(engine, system: "You are terse.", turns: turns)
        XCTAssertFalse(r1.isEmpty)
        XCTAssertEqual(cached1, 0)
        XCTAssertEqual(s1.stopReason, .eos)

        turns.append(ChatTurn(role: .assistant, text: r1))
        turns.append(ChatTurn(role: .user, text: "Now reply with exactly the word: mango"))
        let (r2, s2, cached2) = try await collect(engine, system: "You are terse.", turns: turns)
        XCTAssertFalse(r2.isEmpty)
        // The whole first exchange should have been served from the KV cache.
        XCTAssertGreaterThan(cached2, s1.promptTokens / 2, "expected prefix reuse, got \(cached2) cached of \(s2.promptTokens)")
        await engine.unload()
    }

    func testLongHistoryIsTruncatedNotFailed() async throws {
        guard let path = Self.modelPath else { throw XCTSkip("EDGECHAT_TEST_MODEL not set") }
        let engine = LlamaEngine()
        var config = EngineConfig()
        config.contextLength = 1024
        _ = try await engine.load(modelPath: path, config: config)
        var turns: [ChatTurn] = []
        for i in 0..<30 {
            turns.append(ChatTurn(role: .user, text: "Message number \(i): " + String(repeating: "lorem ipsum ", count: 20)))
            turns.append(ChatTurn(role: .assistant, text: "Acknowledged \(i)."))
        }
        turns.append(ChatTurn(role: .user, text: "Say OK."))
        var dropped = -1
        var sampling = SamplingConfig(); sampling.maxTokens = 16
        for try await ev in engine.respond(systemPrompt: nil, turns: turns, sampling: sampling) {
            if case .promptProcessed(let n, _, let d, _) = ev { dropped = d; XCTAssertLessThanOrEqual(n, 1024 - 16) }
        }
        XCTAssertGreaterThan(dropped, 0)
        await engine.unload()
    }

    func testSingleHugeMessageIsShrunk() async throws {
        guard let path = Self.modelPath else { throw XCTSkip("EDGECHAT_TEST_MODEL not set") }
        let engine = LlamaEngine()
        var config = EngineConfig(); config.contextLength = 1024
        _ = try await engine.load(modelPath: path, config: config)
        let huge = String(repeating: "The quick brown fox jumps over the lazy dog. ", count: 800)
        var sampling = SamplingConfig(); sampling.maxTokens = 16
        var ok = false
        for try await ev in engine.respond(systemPrompt: nil, turns: [ChatTurn(role: .user, text: huge + "\nSummarize in 3 words.")], sampling: sampling) {
            if case .finished = ev { ok = true }
        }
        XCTAssertTrue(ok)
        await engine.unload()
    }

    /// "Continue": a reply cut at its length limit resumes mid-thought, reusing the cached prefix, without restarting.
    func testContinueResumesPartialReply() async throws {
        guard let path = Self.modelPath else { throw XCTSkip("EDGECHAT_TEST_MODEL not set") }
        let engine = LlamaEngine()
        var config = EngineConfig(); config.contextLength = 1024
        _ = try await engine.load(modelPath: path, config: config)
        let user = ChatTurn(role: .user, text: "Count from one to thirty in words, separated by commas.")
        let (part1, stats1, _) = try await collect(engine, system: nil, turns: [user], maxTokens: 12)
        XCTAssertEqual(stats1.stopReason, .maxTokens)
        let (part2, stats2, cached2) = try await collect(engine, system: nil, turns: [user, ChatTurn(role: .assistant, text: part1)], maxTokens: 40, continuing: true)
        print("part1: \(part1.debugDescription)\npart2: \(part2.debugDescription) cached=\(cached2)/\(stats2.promptTokens)")
        XCTAssertFalse(part2.isEmpty)
        // The prefix (prompt + the partial reply) came from the cache: only a handful of tokens were re-evaluated.
        XCTAssertGreaterThan(cached2, stats2.promptTokens - 8)
        let joined = (part1 + part2).lowercased()
        XCTAssertTrue(joined.contains("one") && joined.contains("ten"), "expected the count to progress: \(joined)")
        // A continuation does not restart the answer from the beginning.
        XCTAssertFalse(part2.lowercased().hasPrefix("one,"))
        // An unlimited reply (maxTokens 0) runs until the model stops.
        let (full, stats3, _) = try await collect(engine, system: nil, turns: [user], maxTokens: 0)
        XCTAssertEqual(stats3.stopReason, .eos)
        XCTAssertTrue(full.lowercased().contains("thirty"))
        await engine.unload()
    }

    func testCancellationStopsGeneration() async throws {
        guard let path = Self.modelPath else { throw XCTSkip("EDGECHAT_TEST_MODEL not set") }
        let engine = LlamaEngine()
        _ = try await engine.load(modelPath: path, config: EngineConfig())
        var sampling = SamplingConfig(); sampling.maxTokens = 400
        let task = Task { () -> Int in
            var n = 0
            for try await ev in engine.respond(systemPrompt: nil, turns: [ChatTurn(role: .user, text: "Write a long story about the sea.")], sampling: sampling) {
                if case .token = ev { n += 1; if n == 5 { break } }
            }
            return n
        }
        let t0 = Date()
        let n = try await task.value
        XCTAssertEqual(n, 5)
        let tConsumer = Date().timeIntervalSince(t0)
        // Engine must be free again shortly after the consumer stops.
        var waited = 0.0
        while engine.isGenerating && waited < 5 { try await Task.sleep(nanoseconds: 50_000_000); waited += 0.05 }
        XCTAssertFalse(engine.isGenerating, "generation kept running after the consumer stopped")
        print("cancel timing: consumer \(tConsumer)s, engine idle after \(waited)s")
        XCTAssertLessThan(waited, 3.0)
        await engine.unload()
        print("cancel timing: unload done at \(Date().timeIntervalSince(t0))s")
    }

    func testVisionDescribesSyntheticImage() async throws {
        guard let path = Self.env["EDGECHAT_TEST_VLM"], let mmproj = Self.env["EDGECHAT_TEST_MMPROJ"] else {
            throw XCTSkip("EDGECHAT_TEST_VLM / EDGECHAT_TEST_MMPROJ not set")
        }
        let engine = LlamaEngine()
        var config = EngineConfig(); config.contextLength = 4096
        let info = try await engine.load(modelPath: path, mmprojPath: mmproj, config: config)
        XCTAssertTrue(info.hasVision)

        // Solid red square on white.
        let size = 256
        let ctx = CGContext(data: nil, width: size, height: size, bitsPerComponent: 8, bytesPerRow: 0,
                            space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
        ctx.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1)); ctx.fill(CGRect(x: 0, y: 0, width: size, height: size))
        ctx.setFillColor(CGColor(red: 1, green: 0, blue: 0, alpha: 1)); ctx.fill(CGRect(x: 48, y: 48, width: 160, height: 160))
        let jpeg = try XCTUnwrap(AttachmentProcessor.jpegData(ctx.makeImage()!))
        let (input, _) = try AttachmentProcessor.makeImageInput(from: jpeg, id: "red-square", maxDimension: 512)

        var turns = [ChatTurn(role: .user, text: "What color is the shape in this image? Answer with one word.", images: [input])]
        let (answer, stats, _) = try await collect(engine, system: nil, turns: turns, maxTokens: 32)
        print("VLM answer: \(answer)")
        XCTAssertFalse(answer.isEmpty)
        XCTAssertGreaterThan(stats.promptTokens, 50, "image tokens should be part of the prompt")

        // Second turn must reuse the image from the cache.
        turns.append(ChatTurn(role: .assistant, text: answer))
        turns.append(ChatTurn(role: .user, text: "Is the background dark or light? One word."))
        let (answer2, _, cached2) = try await collect(engine, system: nil, turns: turns, maxTokens: 32)
        print("VLM answer 2: \(answer2) (cached \(cached2))")
        XCTAssertGreaterThan(cached2, 50)
        await engine.unload()
    }
}

final class LongConversationTests: XCTestCase {
    static var modelPath: String? { ProcessInfo.processInfo.environment["EDGECHAT_TEST_MODEL"] }

    func testContextShiftKeepsGenerating() async throws {
        guard let path = Self.modelPath else { throw XCTSkip("EDGECHAT_TEST_MODEL not set") }
        let engine = LlamaEngine()
        var config = EngineConfig(); config.contextLength = 384; config.contextShift = true
        let info = try await engine.load(modelPath: path, config: config)
        let nCtx = info.contextLength // llama.cpp pads the context (384 → 512)
        var sampling = SamplingConfig(); sampling.maxTokens = 300; sampling.temperature = 0; sampling.repeatPenalty = 1.0
        let reserve = LlamaEngine.replyReserve(contextLength: nCtx, sampling: sampling)
        // Fill the prompt budget, then ask for a reply longer than the reserve.
        let filler = String(repeating: "The lighthouse keeper watched the grey sea. ", count: 40)
        let turns = [ChatTurn(role: .user, text: filler + "\nNow count from 1 to 400 separated by commas. Do not stop early.")]
        var stats: GenerationStats?
        for try await ev in engine.respond(systemPrompt: "You are a counting machine.", turns: turns, sampling: sampling) {
            if case .finished(let s) = ev { stats = s }
        }
        let s = try XCTUnwrap(stats)
        print("shift test: ctx \(nCtx), prompt \(s.promptTokens), generated \(s.generatedTokens), shifts \(s.contextShifts ?? 0), stop \(s.stopReason)")
        XCTAssertNotEqual(s.stopReason, .contextFull)
        XCTAssertLessThanOrEqual(s.promptTokens, nCtx - reserve - 8)
        let overflowed = s.promptTokens + s.generatedTokens >= nCtx
        if overflowed { XCTAssertGreaterThan(s.contextShifts ?? 0, 0) }

        // With shifting off the same request must stop cleanly at the context edge.
        config.contextShift = false
        _ = try await engine.load(modelPath: path, config: config)
        var stats2: GenerationStats?
        for try await ev in engine.respond(systemPrompt: "You are a counting machine.", turns: turns, sampling: sampling) {
            if case .finished(let s) = ev { stats2 = s }
        }
        let s2 = try XCTUnwrap(stats2)
        XCTAssertLessThanOrEqual(s2.promptTokens + s2.generatedTokens, nCtx)
        if overflowed { XCTAssertEqual(s2.stopReason, .contextFull) }
        await engine.unload()
    }

    func testSnapshotRoundTripRestoresCache() async throws {
        guard let path = Self.modelPath else { throw XCTSkip("EDGECHAT_TEST_MODEL not set") }
        let engine = LlamaEngine()
        var config = EngineConfig(); config.contextLength = 2048
        _ = try await engine.load(modelPath: path, config: config)
        var sampling = SamplingConfig(); sampling.maxTokens = 40; sampling.temperature = 0
        var turns = [ChatTurn(role: .user, text: "Remember the code word ORANGE-7. Reply with OK.")]
        var reply = ""
        var promptTokens = 0
        for try await ev in engine.respond(systemPrompt: "Be brief.", turns: turns, sampling: sampling) {
            if case .token(let t) = ev { reply += t }
            if case .finished(let s) = ev { promptTokens = s.promptTokens }
        }
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("edgechat-test-\(UUID().uuidString).kv")
        defer { try? FileManager.default.removeItem(at: url); try? FileManager.default.removeItem(at: url.appendingPathExtension("units")) }
        let bytes = try await engine.saveState(to: url)
        XCTAssertGreaterThan(bytes, 0)
        await engine.clearCache()
        let restored = await engine.loadState(from: url)
        XCTAssertTrue(restored)
        turns.append(ChatTurn(role: .assistant, text: reply))
        turns.append(ChatTurn(role: .user, text: "What was the code word?"))
        var cached = 0
        var answer = ""
        for try await ev in engine.respond(systemPrompt: "Be brief.", turns: turns, sampling: sampling) {
            if case .promptProcessed(_, let c, _, _) = ev { cached = c }
            if case .token(let t) = ev { answer += t }
        }
        print("snapshot test: cached \(cached) of prompt \(promptTokens), answer: \(answer)")
        XCTAssertGreaterThan(cached, promptTokens / 2)
        // A bogus file must fail cleanly.
        let bogus = await engine.loadState(from: url.appendingPathExtension("missing"))
        XCTAssertFalse(bogus)
        // Damaged snapshots (extra bytes, truncated, wrong header) must be rejected and deleted, never abort.
        let good = try Data(contentsOf: url)
        let unitsData = try Data(contentsOf: url.appendingPathExtension("units"))
        for (name, bytes) in [("longer", good + Data(repeating: 7, count: 64)),
                              ("shorter", good.prefix(good.count - 64)),
                              ("garbage", Data((0..<good.count).map { UInt8($0 & 0xff) }))] {
            let bad = url.deletingLastPathComponent().appendingPathComponent("edgechat-bad-\(name).kv")
            try bytes.write(to: bad)
            try unitsData.write(to: bad.appendingPathExtension("units"))
            let loaded = await engine.loadState(from: bad)
            XCTAssertFalse(loaded, name)
            XCTAssertFalse(FileManager.default.fileExists(atPath: bad.path), "\(name) should be deleted")
        }
        // The engine still works afterwards.
        var after = ""
        for try await ev in engine.respond(systemPrompt: "Be brief.", turns: [ChatTurn(role: .user, text: "Say hi.")], sampling: sampling) {
            if case .token(let t) = ev { after += t }
        }
        XCTAssertFalse(after.isEmpty)
        await engine.unload()
    }

    func testPlanTruncationAndSummary() async throws {
        guard let path = Self.modelPath else { throw XCTSkip("EDGECHAT_TEST_MODEL not set") }
        let engine = LlamaEngine()
        var config = EngineConfig(); config.contextLength = 1024
        _ = try await engine.load(modelPath: path, config: config)
        var turns: [ChatTurn] = []
        for i in 0..<12 {
            turns.append(ChatTurn(role: .user, text: "Fact \(i): my favourite fruit number \(i) is " + ["mango", "kiwi", "plum", "fig"][i % 4] + ". " + String(repeating: "Some filler text about nothing in particular. ", count: 12)))
            turns.append(ChatTurn(role: .assistant, text: "Noted, fact \(i)."))
        }
        turns.append(ChatTurn(role: .user, text: "List my fruits."))
        var sampling = SamplingConfig(); sampling.maxTokens = 128
        let drop = try await engine.planTruncation(systemPrompt: nil, turns: turns, sampling: sampling)
        XCTAssertGreaterThan(drop, 0)
        XCTAssertLessThan(drop, turns.count)
        // Proactive compaction plans a tighter fit (half the budget) so it drops at least as much, and everything
        // that fits the full budget needs no drop at all.
        let tighter = try await engine.planTruncation(systemPrompt: nil, turns: turns, sampling: sampling, budgetFraction: 0.5)
        XCTAssertGreaterThan(tighter, drop)
        XCTAssertLessThan(tighter, turns.count)
        let budget = try await engine.promptBudget(sampling: sampling)
        XCTAssertGreaterThan(budget, 128)
        let none = try await engine.planTruncation(systemPrompt: nil, turns: Array(turns.suffix(3)), sampling: sampling, budgetFraction: 0.5)
        XCTAssertEqual(none, 0)
        let summary = try await engine.summarize(previousSummary: nil, turns: Array(turns[0..<drop]))
        print("summary (\(drop) turns): \(summary)")
        XCTAssertFalse(summary.isEmpty)
        await engine.unload()
    }
}


/// Qwen3-VL-style models: image tokens occupy far more KV cells than RoPE positions. The budget must use cells.
final class MRopeVisionTests: XCTestCase {
    func testImageHeavyHistoryIsTruncatedByCells() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let path = env["EDGECHAT_TEST_MROPE_VLM"], let mmproj = env["EDGECHAT_TEST_MROPE_MMPROJ"] else {
            throw XCTSkip("EDGECHAT_TEST_MROPE_VLM / EDGECHAT_TEST_MROPE_MMPROJ not set")
        }
        let engine = LlamaEngine()
        var config = EngineConfig(); config.contextLength = 1024
        let info = try await engine.load(modelPath: path, mmprojPath: mmproj, config: config)
        XCTAssertTrue(info.hasVision)
        func image(_ seed: Int) throws -> ImageInput {
            let size = 512
            let ctx = CGContext(data: nil, width: size, height: size, bitsPerComponent: 8, bytesPerRow: 0,
                                space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
            ctx.setFillColor(CGColor(red: CGFloat(seed % 2), green: 0.5, blue: 1 - CGFloat(seed % 2), alpha: 1))
            ctx.fill(CGRect(x: 0, y: 0, width: size, height: size))
            let jpeg = try XCTUnwrap(AttachmentProcessor.jpegData(ctx.makeImage()!))
            return try AttachmentProcessor.makeImageInput(from: jpeg, id: "img-\(seed)", maxDimension: 512).input
        }
        var sampling = SamplingConfig(); sampling.maxTokens = 48; sampling.temperature = 0
        var turns: [ChatTurn] = []
        var lastStats: GenerationStats?
        // Each 512px image is ~256 cells on Qwen3-VL; four of them exceed a 1024-cell context.
        for i in 0..<4 {
            turns.append(ChatTurn(role: .user, text: "Image \(i): what colour is it? One word.", images: [try image(i)]))
            var reply = ""
            for try await ev in engine.respond(systemPrompt: nil, turns: turns, sampling: sampling) {
                if case .token(let t) = ev { reply += t }
                if case .finished(let st) = ev { lastStats = st }
            }
            turns.append(ChatTurn(role: .assistant, text: reply))
            let st = try XCTUnwrap(lastStats)
            print("mrope turn \(i): prompt cells \(st.promptTokens), cached \(st.cachedTokens), stop \(st.stopReason)")
            XCTAssertNotEqual(st.stopReason, .contextFull)
            XCTAssertLessThanOrEqual(st.promptTokens, info.contextLength - LlamaEngine.replyReserve(contextLength: info.contextLength, sampling: sampling) - 8)
        }
        await engine.unload()
    }

    /// Snapshots of a cache holding image cells (M-RoPE positions) must round-trip and be reused on the next turn.
    func testSnapshotWithImageCellsRoundTrips() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let path = env["EDGECHAT_TEST_MROPE_VLM"], let mmproj = env["EDGECHAT_TEST_MROPE_MMPROJ"] else {
            throw XCTSkip("EDGECHAT_TEST_MROPE_VLM / EDGECHAT_TEST_MROPE_MMPROJ not set")
        }
        let engine = LlamaEngine()
        var config = EngineConfig(); config.contextLength = 2048
        _ = try await engine.load(modelPath: path, mmprojPath: mmproj, config: config)
        let size = 512
        let cg = CGContext(data: nil, width: size, height: size, bitsPerComponent: 8, bytesPerRow: 0,
                           space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
        cg.setFillColor(CGColor(red: 1, green: 0, blue: 0, alpha: 1)); cg.fill(CGRect(x: 0, y: 0, width: size, height: size))
        let image = try AttachmentProcessor.makeImageInput(from: try XCTUnwrap(AttachmentProcessor.jpegData(cg.makeImage()!)), id: "red", maxDimension: 512).input
        var sampling = SamplingConfig(); sampling.maxTokens = 12; sampling.temperature = 0
        var turns = [ChatTurn(role: .user, text: "What colour is this image? One word.", images: [image])]
        var reply = ""; var prompt = 0
        for try await ev in engine.respond(systemPrompt: nil, turns: turns, sampling: sampling) {
            if case .token(let t) = ev { reply += t }
            if case .finished(let s) = ev { prompt = s.promptTokens }
        }
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("edgechat-mrope-\(UUID().uuidString).kv")
        defer { try? FileManager.default.removeItem(at: url); try? FileManager.default.removeItem(at: url.appendingPathExtension("units")) }
        _ = try await engine.saveState(to: url)
        await engine.clearCache()
        let restored = await engine.loadState(from: url)
        XCTAssertTrue(restored)
        turns.append(ChatTurn(role: .assistant, text: reply))
        turns.append(ChatTurn(role: .user, text: "Say the same colour again, one word."))
        var cached = 0; var answer = ""
        for try await ev in engine.respond(systemPrompt: nil, turns: turns, sampling: sampling) {
            if case .promptProcessed(_, let c, _, _) = ev { cached = c }
            if case .token(let t) = ev { answer += t }
        }
        print("mrope snapshot: cached \(cached) of \(prompt) cells; first \(reply.debugDescription) then \(answer.debugDescription)")
        XCTAssertGreaterThan(cached, prompt / 2)
        XCTAssertFalse(answer.isEmpty)
        await engine.unload()
    }
}
