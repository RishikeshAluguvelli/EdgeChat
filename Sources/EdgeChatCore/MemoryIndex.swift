import Foundation
import NaturalLanguage

/// A retrievable chunk of conversation history (a message, or a slice of a long message/document).
public struct MemoryChunk: Codable, Hashable, Sendable, Identifiable {
    public var id: UUID
    public var messageID: UUID
    public var messageIndex: Int
    public var role: Role
    public var source: String          // "message" or the attachment file name
    public var text: String
    public var embedding: [Float]      // empty when the embedder had nothing to say
    public var createdAt: Date

    public init(id: UUID, messageID: UUID, messageIndex: Int, role: Role, source: String, text: String, embedding: [Float], createdAt: Date) {
        self.id = id; self.messageID = messageID; self.messageIndex = messageIndex; self.role = role
        self.source = source; self.text = text; self.embedding = embedding; self.createdAt = createdAt
    }

    // Vectors are stored as little-endian Float32 base64 (~4x smaller and much faster to parse than a JSON number array).
    private enum CodingKeys: String, CodingKey { case id, messageID, messageIndex, role, source, text, embedding, vector, createdAt }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decode(UUID.self, forKey: .id)
        messageID = try c.decode(UUID.self, forKey: .messageID)
        messageIndex = try c.decode(Int.self, forKey: .messageIndex)
        role = try c.decode(Role.self, forKey: .role)
        source = try c.decode(String.self, forKey: .source)
        text = try c.decode(String.self, forKey: .text)
        createdAt = try c.decode(Date.self, forKey: .createdAt)
        if let data = try c.decodeIfPresent(Data.self, forKey: .vector) {
            embedding = data.withUnsafeBytes { raw in
                raw.bindMemory(to: UInt32.self).map { Float(bitPattern: UInt32(littleEndian: $0)) }
            }
        } else {
            embedding = try c.decodeIfPresent([Float].self, forKey: .embedding) ?? []
        }
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(id, forKey: .id)
        try c.encode(messageID, forKey: .messageID)
        try c.encode(messageIndex, forKey: .messageIndex)
        try c.encode(role, forKey: .role)
        try c.encode(source, forKey: .source)
        try c.encode(text, forKey: .text)
        try c.encode(createdAt, forKey: .createdAt)
        if !embedding.isEmpty {
            var data = Data(capacity: embedding.count * 4)
            for f in embedding { var v = f.bitPattern.littleEndian; withUnsafeBytes(of: &v) { data.append(contentsOf: $0) } }
            try c.encode(data, forKey: .vector)
        }
    }
}

/// Per-conversation retrieval index: on-device sentence embeddings (NaturalLanguage) + BM25, no model download.
public struct MemoryIndex: Codable, Sendable {
    public var chunks: [MemoryChunk] = []
    public var indexedMessageIDs: Set<UUID> = []
    /// Identifies the embedding space (model) the chunk vectors were produced in.
    public var embeddingKey: String?

    public init() {}

    public typealias Embedder = ([String]) async throws -> [[Float]]

    public struct Hit: Sendable {
        public let chunk: MemoryChunk
        public let score: Double
    }

    // MARK: Persistence

    public static func load(from url: URL) -> MemoryIndex {
        guard let data = try? Data(contentsOf: url), let idx = try? JSONDecoder().decode(MemoryIndex.self, from: data) else { return MemoryIndex() }
        return idx
    }

    public func save(to url: URL) throws {
        let data = try JSONEncoder().encode(self)
        try data.write(to: url, options: .atomic)
    }

    // MARK: Indexing

    /// Adds a message (and any extra texts such as full document extractions) to the index.
    /// Pass `embedder` + `embeddingKey` to compute vectors; without them chunks are lexical-only until re-embedded.
    public mutating func index(message: Message, at messageIndex: Int, extraTexts: [(source: String, text: String)] = [],
                               embedder: Embedder? = nil, embeddingKey: String? = nil) async {
        guard !indexedMessageIDs.contains(message.id) else { return }
        indexedMessageIDs.insert(message.id)
        var sources: [(String, String)] = []
        if !message.content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { sources.append(("message", message.content)) }
        sources += extraTexts.map { ($0.source, $0.text) }
        var new: [MemoryChunk] = []
        for (source, text) in sources {
            for piece in Self.chunk(text) {
                new.append(MemoryChunk(id: UUID(), messageID: message.id, messageIndex: messageIndex, role: message.role,
                                       source: source, text: piece, embedding: [], createdAt: message.createdAt))
            }
        }
        if let embedder, let embeddingKey, !new.isEmpty {
            if self.embeddingKey != embeddingKey { self.embeddingKey = embeddingKey; for i in chunks.indices { chunks[i].embedding = [] } }
            if let vectors = try? await embedder(new.map(\.text)), vectors.count == new.count {
                for i in new.indices { new[i].embedding = vectors[i] }
            }
        }
        chunks += new
    }

    /// Re-embeds chunks that lack a vector in the current space (bounded, so a model switch never stalls a turn).
    public mutating func refreshEmbeddings(embedder: Embedder, embeddingKey: String, limit: Int = 24, preferring ids: [UUID] = []) async {
        if self.embeddingKey != embeddingKey { self.embeddingKey = embeddingKey; for i in chunks.indices { chunks[i].embedding = [] } }
        var todo = chunks.indices.filter { chunks[$0].embedding.isEmpty }
        let preferred = Set(ids)
        todo.sort { (preferred.contains(chunks[$0].id) ? 0 : 1) < (preferred.contains(chunks[$1].id) ? 0 : 1) }
        let batch = Array(todo.prefix(limit))
        guard !batch.isEmpty, let vectors = try? await embedder(batch.map { chunks[$0].text }), vectors.count == batch.count else { return }
        for (j, i) in batch.enumerated() { chunks[i].embedding = vectors[j] }
    }

    public mutating func remove(messageID: UUID) {
        chunks.removeAll { $0.messageID == messageID }
        indexedMessageIDs.remove(messageID)
    }

    // MARK: Retrieval

    /// Hybrid search: 0.55 × cosine(model embedding) + 0.45 × normalized BM25, over `candidates`.
    /// `queryEmbedding` comes from the same embedder used for indexing (may be empty → lexical only).
    public func search(query: String, queryEmbedding qEmb: [Float], candidates: [MemoryChunk], limit: Int = 6, minScore: Double = 0.3) -> [Hit] {
        guard !candidates.isEmpty, !query.trimmingCharacters(in: .whitespaces).isEmpty else { return [] }
        let qTerms = Self.terms(query)

        // BM25 statistics over the candidate set.
        let docs = candidates.map { Self.terms($0.text) }
        let avgLen = max(1.0, Double(docs.reduce(0) { $0 + $1.count }) / Double(docs.count))
        var df: [String: Int] = [:]
        for d in docs { for t in Set(d) { df[t, default: 0] += 1 } }
        let n = Double(docs.count)
        func bm25(_ d: [String]) -> Double {
            guard !d.isEmpty else { return 0 }
            var tf: [String: Int] = [:]
            for t in d { tf[t, default: 0] += 1 }
            var score = 0.0
            for q in Set(qTerms) {
                guard let f = tf[q] else { continue }
                let dfq: Double = Double(df[q] ?? 0)
                let idf: Double = log(1 + (n - dfq + 0.5) / (dfq + 0.5))
                let tf: Double = Double(f)
                let lenNorm: Double = 0.25 + 0.75 * Double(d.count) / avgLen
                let tfn: Double = tf * 2.2 / (tf + 1.2 * lenNorm)
                score += idf * tfn
            }
            return score
        }
        let lexical = docs.map(bm25)
        let maxLex = lexical.max() ?? 0
        // Cosines from mean-pooled LLM states cluster high; rescale against the candidate spread.
        let cosines = candidates.map { (qEmb.isEmpty || $0.embedding.isEmpty) ? Double.nan : Self.cosine(qEmb, $0.embedding) }
        let valid = cosines.filter { !$0.isNaN }
        let cMin = valid.min() ?? 0, cMax = valid.max() ?? 0
        var hits: [Hit] = []
        for (i, c) in candidates.enumerated() {
            var semantic: Double = 0
            let cos: Double = cosines[i]
            if !cos.isNaN {
                if cMax > cMin {
                    semantic = (cos - cMin) / (cMax - cMin)
                } else if valid.count == 1 {
                    semantic = 1
                }
            }
            var lex: Double = 0
            if maxLex > 0 { lex = lexical[i] / maxLex }
            let score: Double = 0.55 * semantic + 0.45 * lex
            // Min-max rescaling always gives *some* chunk semantic = 1, so demand real evidence too: a shared query
            // term, or a cosine clearly at the top of a meaningful spread.
            let spread: Double = cMax - cMin
            let grounded: Bool = lexical[i] > 0 || (semantic >= 0.9 && spread >= 0.04)
            if score >= minScore, grounded { hits.append(Hit(chunk: c, score: score)) }
        }
        return Array(hits.sorted { $0.score > $1.score }.prefix(limit))
    }

    /// Renders hits as a block for the prompt, oldest first, within a character budget.
    public static func renderRecall(_ hits: [Hit], maxCharacters: Int) -> String? {
        guard !hits.isEmpty else { return nil }
        var lines: [String] = []
        var used = 0
        for h in hits.sorted(by: { $0.chunk.messageIndex < $1.chunk.messageIndex }) {
            let who = h.chunk.source == "message" ? (h.chunk.role == .user ? "User said" : "Assistant said") : "From \(h.chunk.source)"
            let line = "- (\(who), message \(h.chunk.messageIndex + 1)) \(h.chunk.text.replacingOccurrences(of: "\n", with: " "))"
            if used + line.count > maxCharacters { break }
            lines.append(line)
            used += line.count
        }
        guard !lines.isEmpty else { return nil }
        return "<recalled_context note=\"background only: earlier parts of this conversation that may be relevant. Do not repeat or re-answer them; use them only if they help with the message below.\">\n" + lines.joined(separator: "\n") + "\n</recalled_context>"
    }

    // MARK: Helpers

    /// ~600-character chunks on paragraph/sentence boundaries.
    public static func chunk(_ text: String, target: Int = 600, maxLength: Int = 900) -> [String] {
        let paragraphs = text.components(separatedBy: "\n\n").map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }.filter { !$0.isEmpty }
        var out: [String] = []
        var current = ""
        func flush() { if !current.isEmpty { out.append(current); current = "" } }
        for p in paragraphs {
            let units = p.count > maxLength ? sentences(p) : [p]
            for u in units {
                if current.count + u.count + 1 > target && !current.isEmpty { flush() }
                if u.count > maxLength {
                    // Hard split very long runs (code, tables).
                    var rest = Substring(u)
                    while !rest.isEmpty {
                        let piece = String(rest.prefix(maxLength)); rest = rest.dropFirst(maxLength)
                        flush(); out.append(piece)
                    }
                } else {
                    current += current.isEmpty ? u : "\n" + u
                }
            }
        }
        flush()
        return out
    }

    private static func sentences(_ text: String) -> [String] {
        var result: [String] = []
        let tokenizer = NLTokenizer(unit: .sentence)
        tokenizer.string = text
        tokenizer.enumerateTokens(in: text.startIndex..<text.endIndex) { range, _ in
            let s = text[range].trimmingCharacters(in: .whitespacesAndNewlines)
            if !s.isEmpty { result.append(s) }
            return true
        }
        return result.isEmpty ? [text] : result
    }

    static func cosine(_ a: [Float], _ b: [Float]) -> Double {
        guard a.count == b.count, !a.isEmpty else { return 0 }
        var dot: Float = 0, na: Float = 0, nb: Float = 0
        for i in 0..<a.count { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i] }
        guard na > 0, nb > 0 else { return 0 }
        return Double(dot / (na.squareRoot() * nb.squareRoot()))
    }

    static func terms(_ text: String) -> [String] {
        text.lowercased()
            .components(separatedBy: CharacterSet.alphanumerics.inverted)
            .filter { $0.count > 1 && !stopwords.contains($0) }
    }

    private static let stopwords: Set<String> = ["the", "a", "an", "and", "or", "of", "to", "in", "on", "is", "it", "that", "this", "for", "with", "as", "was", "are", "be", "at", "by", "i", "you", "we", "he", "she", "they", "my", "your", "me", "do", "did", "what", "how", "why", "can", "could", "would", "should", "about", "from", "so", "if", "not", "no", "yes", "there", "here", "have", "has", "had", "will", "just", "also", "but"]
}
