import XCTest
@testable import EdgeChatCore

final class TextUtilsTests: XCTestCase {
    func testThinkSplitterHandlesSplitTags() {
        var s = ThinkTagSplitter()
        var parts: [ThinkTagSplitter.Part] = []
        for chunk in ["<th", "ink>plan", " here</th", "ink>Answer", " text"] { parts += s.feed(chunk) }
        parts += s.flush()
        let reasoning = parts.compactMap { if case .reasoning(let r) = $0 { return r } else { return nil } }.joined()
        let text = parts.compactMap { if case .text(let t) = $0 { return t } else { return nil } }.joined()
        XCTAssertEqual(reasoning, "plan here")
        XCTAssertEqual(text, "Answer text")
    }

    func testThinkSplitterPassesPlainTextThrough() {
        var s = ThinkTagSplitter()
        var parts = s.feed("Hello <b>world</b>")
        parts += s.flush()
        XCTAssertEqual(parts, [.text("Hello <b>world</b>")])
    }

    func testUTF8AccumulatorHoldsPartialSequences() {
        var acc = UTF8Accumulator()
        let emoji = Array("😀".utf8) // 4 bytes
        XCTAssertNil(acc.append([emoji[0], emoji[1]]))
        XCTAssertEqual(acc.append([emoji[2], emoji[3]]), "😀")
        XCTAssertEqual(acc.append(Array("ok".utf8)), "ok")
        XCTAssertNil(acc.flush())
    }

    func testTruncateMiddleKeepsHeadAndTail() {
        let text = String(repeating: "a", count: 500) + String(repeating: "z", count: 500)
        let (t, truncated) = TextUtils.truncateMiddle(text, maxCharacters: 200)
        XCTAssertTrue(truncated)
        XCTAssertLessThanOrEqual(t.count, 200)
        XCTAssertTrue(t.hasPrefix("aaaa"))
        XCTAssertTrue(t.hasSuffix("zzzz"))
        XCTAssertTrue(t.contains("omitted"))
    }
}

final class MemoryIndexTests: XCTestCase {
    static let msgs = [
        Message(role: .user, content: "My dog is called Biscuit and he is a golden retriever who loves swimming."),
        Message(role: .assistant, content: "Biscuit sounds lovely! Golden retrievers are great swimmers."),
        Message(role: .user, content: "I am planning a trip to Lisbon in October for a conference on robotics."),
        Message(role: .assistant, content: "October is a nice time to visit Lisbon; the weather is mild."),
        Message(role: .user, content: "Can you help me write a SQL query that joins orders and customers?"),
    ]

    func testLexicalRetrievalWithoutEmbeddings() async {
        var index = MemoryIndex()
        for (i, m) in Self.msgs.enumerated() { await index.index(message: m, at: i) }
        XCTAssertEqual(index.chunks.count, 5)
        let hits = index.search(query: "when is the Lisbon conference?", queryEmbedding: [], candidates: index.chunks, limit: 2)
        XCTAssertTrue(hits.first?.chunk.text.contains("Lisbon") == true, "hits: \(hits.map { $0.chunk.text })")
        XCTAssertNotNil(MemoryIndex.renderRecall(hits, maxCharacters: 2000))
    }

    func testSemanticRetrievalWithModelEmbeddings() async throws {
        guard let path = ProcessInfo.processInfo.environment["EDGECHAT_TEST_MODEL"] else { throw XCTSkip("EDGECHAT_TEST_MODEL not set") }
        let engine = LlamaEngine()
        _ = try await engine.load(modelPath: path, config: EngineConfig())
        let embedder: MemoryIndex.Embedder = { try await engine.embed($0) }
        var index = MemoryIndex()
        for (i, m) in Self.msgs.enumerated() { await index.index(message: m, at: i, embedder: embedder, embeddingKey: "test") }
        XCTAssertTrue(index.chunks.allSatisfy { !$0.embedding.isEmpty })
        // Vectors round-trip through the compact base64 encoding exactly.
        let data = try JSONEncoder().encode(index)
        let back = try JSONDecoder().decode(MemoryIndex.self, from: data)
        XCTAssertEqual(back.chunks.map(\.embedding), index.chunks.map(\.embedding))
        XCTAssertLessThan(data.count, index.chunks.count * (index.chunks[0].embedding.count * 6 + 400))
        let q = "What breed is my pet?"   // no lexical overlap with the answer
        let qv = try await engine.embed([q])[0]
        let hits = index.search(query: q, queryEmbedding: qv, candidates: index.chunks, limit: 3, minScore: 0)
        print("semantic hits:", hits.map { (String(format: "%.2f", $0.score), String($0.chunk.text.prefix(40))) })
        XCTAssertTrue(hits.first?.chunk.text.contains("Biscuit") == true)
        await engine.unload()
    }

    func testChunkingSplitsLongText() {
        let text = (0..<40).map { "Sentence number \($0) talks about topic \($0 % 5) in some detail." }.joined(separator: " ")
        let chunks = MemoryIndex.chunk(text)
        XCTAssertGreaterThan(chunks.count, 1)
        XCTAssertTrue(chunks.allSatisfy { $0.count <= 900 })
        let squash: (String) -> String = { $0.filter { !$0.isWhitespace } }
        XCTAssertEqual(squash(chunks.joined()), squash(text))
    }
}
