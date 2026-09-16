import Foundation

/// Accumulates raw token bytes and emits only complete UTF-8 sequences, so multi-byte
/// characters split across tokens (emoji, CJK) never produce replacement glyphs mid-stream.
struct UTF8Accumulator {
    private var buffer: [UInt8] = []

    mutating func append(_ bytes: [UInt8]) -> String? {
        buffer.append(contentsOf: bytes)
        var cut = buffer.count
        var i = buffer.count - 1
        var trailing = 0
        while i >= 0 && trailing < 4 {
            let b = buffer[i]
            if b & 0xC0 == 0x80 { i -= 1; trailing += 1; continue } // continuation byte
            let need: Int
            if b < 0x80 { need = 1 }
            else if b & 0xE0 == 0xC0 { need = 2 }
            else if b & 0xF0 == 0xE0 { need = 3 }
            else if b & 0xF8 == 0xF0 { need = 4 }
            else { need = 1 }
            if trailing + 1 < need { cut = i } // sequence incomplete: hold it back
            break
        }
        guard cut > 0 else { return nil }
        let out = String(decoding: buffer[0..<cut], as: UTF8.self)
        buffer.removeFirst(cut)
        return out.isEmpty ? nil : out
    }

    mutating func flush() -> String? {
        guard !buffer.isEmpty else { return nil }
        let out = String(decoding: buffer, as: UTF8.self)
        buffer.removeAll()
        return out.isEmpty ? nil : out
    }
}

/// Splits a streamed reply into visible text and `<think>…</think>` reasoning, tolerating
/// tags that arrive split across several tokens.
public struct ThinkTagSplitter: Sendable {
    public enum Part: Equatable, Sendable {
        case reasoning(String)
        case text(String)
    }

    private static let open = "<think>"
    private static let close = "</think>"
    private var buffer = ""
    private var inside = false

    public init() {}

    public var isInsideReasoning: Bool { inside }

    public mutating func feed(_ chunk: String) -> [Part] {
        buffer += chunk
        var parts: [Part] = []
        while true {
            let tag = inside ? Self.close : Self.open
            if let range = buffer.range(of: tag) {
                let before = String(buffer[buffer.startIndex..<range.lowerBound])
                if !before.isEmpty { parts.append(inside ? .reasoning(before) : .text(before)) }
                buffer = String(buffer[range.upperBound...])
                inside.toggle()
                continue
            }
            // Hold back a suffix that could be the start of a tag.
            let hold = Self.longestSuffixThatPrefixes(buffer, tag)
            let emitCount = buffer.count - hold
            if emitCount > 0 {
                let emit = String(buffer.prefix(emitCount))
                parts.append(inside ? .reasoning(emit) : .text(emit))
                buffer = String(buffer.suffix(hold))
            }
            break
        }
        return parts
    }

    public mutating func flush() -> [Part] {
        defer { buffer = "" }
        guard !buffer.isEmpty else { return [] }
        return [inside ? .reasoning(buffer) : .text(buffer)]
    }

    private static func longestSuffixThatPrefixes(_ s: String, _ tag: String) -> Int {
        let maxLen = min(s.count, tag.count - 1)
        guard maxLen > 0 else { return 0 }
        for len in stride(from: maxLen, through: 1, by: -1) {
            if tag.hasPrefix(String(s.suffix(len))) { return len }
        }
        return 0
    }
}

public enum TextUtils {
    /// Keeps the head and tail of an over-long text, dropping the middle.
    public static func truncateMiddle(_ text: String, maxCharacters: Int, headFraction: Double = 0.7) -> (text: String, truncated: Bool) {
        guard text.count > maxCharacters, maxCharacters > 64 else { return (text, false) }
        let marker = "\n\n[… \(text.count - maxCharacters) characters omitted …]\n\n"
        let budget = max(0, maxCharacters - marker.count)
        let head = Int(Double(budget) * headFraction)
        let tail = budget - head
        return (String(text.prefix(head)) + marker + String(text.suffix(tail)), true)
    }

    /// "512 tokens", "4k", "32k".
    public static func formatContext(_ tokens: Int) -> String {
        tokens >= 1024 && tokens % 1024 == 0 ? "\(tokens / 1024)k" : (tokens >= 1024 ? String(format: "%.1fk", Double(tokens) / 1024) : "\(tokens) tokens")
    }

    public static func formatBytes(_ bytes: Int64) -> String {
        let f = ByteCountFormatter()
        f.countStyle = .file
        f.allowedUnits = [.useMB, .useGB]
        return f.string(fromByteCount: bytes)
    }
}
