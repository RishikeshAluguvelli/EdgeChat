import SwiftUI

/// Lightweight block-level Markdown renderer (paragraphs, headings, lists, quotes, fenced code, rules).
/// Inline formatting is handled by Foundation's AttributedString Markdown parser.
struct MarkdownText: View {
    let text: String
    init(_ text: String) { self.text = text }

    enum Block: Identifiable {
        case paragraph(String)
        case heading(Int, String)
        case bullets([(level: Int, text: String)])
        case numbered([(label: String, text: String)])
        case quote(String)
        case code(String?, String)
        case table(header: [String], rows: [[String]])
        case rule
        var id: String {
            switch self {
            case .paragraph(let s): return "p" + s
            case .heading(let l, let s): return "h\(l)" + s
            case .bullets(let l): return "b" + l.map { "\($0.level)" + $0.text }.joined(separator: "\u{1}")
            case .numbered(let l): return "n" + l.map { $0.label + $0.text }.joined(separator: "\u{1}")
            case .quote(let s): return "q" + s
            case .code(let l, let s): return "c" + (l ?? "") + s
            case .table(let h, let r): return "t" + h.joined(separator: "|") + r.map { $0.joined(separator: "|") }.joined(separator: "\u{1}")
            case .rule: return "rule"
            }
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            ForEach(Array(Self.parse(text).enumerated()), id: \.offset) { _, block in
                render(block)
            }
        }
        .textSelection(.enabled)
    }

    @ViewBuilder
    private func render(_ block: Block) -> some View {
        switch block {
        case .paragraph(let s):
            inline(s).fixedSize(horizontal: false, vertical: true)
        case .heading(let level, let s):
            inline(s).font(level == 1 ? .title2.bold() : level == 2 ? .title3.bold() : .headline).fixedSize(horizontal: false, vertical: true)
        case .bullets(let items):
            VStack(alignment: .leading, spacing: 4) {
                ForEach(Array(items.enumerated()), id: \.offset) { _, item in
                    // fixedSize: without it SwiftUI measured these Texts at one width and laid them out at another,
                    // truncating long bullets to two lines with an ellipsis.
                    HStack(alignment: .firstTextBaseline, spacing: 8) {
                        Text(item.level == 0 ? "•" : "◦")
                        inline(item.text).fixedSize(horizontal: false, vertical: true).frame(maxWidth: .infinity, alignment: .leading)
                    }
                    .padding(.leading, CGFloat(item.level) * 16)
                }
            }
        case .numbered(let items):
            VStack(alignment: .leading, spacing: 4) {
                ForEach(Array(items.enumerated()), id: \.offset) { _, item in
                    HStack(alignment: .firstTextBaseline, spacing: 8) {
                        Text(item.label).monospacedDigit()
                        inline(item.text).fixedSize(horizontal: false, vertical: true).frame(maxWidth: .infinity, alignment: .leading)
                    }
                }
            }
        case .table(let header, let rows):
            MarkdownTable(header: header, rows: rows, inline: inline)
        case .quote(let s):
            HStack(alignment: .top, spacing: 8) {
                RoundedRectangle(cornerRadius: 2).fill(.secondary).frame(width: 3)
                inline(s).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            }
            .fixedSize(horizontal: false, vertical: true)
        case .code(let lang, let code):
            CodeBlock(language: lang, code: code)
        case .rule:
            Divider()
        }
    }

    private func inline(_ s: String) -> Text {
        let safe = Self.escapeAngleBrackets(s)
        if let a = try? AttributedString(markdown: safe, options: .init(interpretedSyntax: .inlineOnlyPreservingWhitespace)) {
            return Text(a)
        }
        return Text(s)
    }

    /// The inline Markdown parser drops anything that looks like an HTML tag (`<eos>`, `<think>`, `<br>`), which
    /// cut words out of sentences. Escape `<` outside code spans so it renders literally.
    static func escapeAngleBrackets(_ s: String) -> String {
        guard s.contains("<") else { return s }
        var out = ""
        var inCode = false
        for ch in s {
            if ch == "`" { inCode.toggle() }
            if ch == "<", !inCode { out.append("\\<") } else { out.append(ch) }
        }
        return out
    }

    private static let cache = NSCache<NSString, BlockBox>()
    final class BlockBox { let blocks: [Block]; init(_ b: [Block]) { blocks = b } }
    private static let numberedItem = try! NSRegularExpression(pattern: #"^(\d+)[.)]\s+"#)

    /// Parses (memoized: finished messages are re-rendered often but never change).
    static func parse(_ text: String) -> [Block] {
        if let hit = cache.object(forKey: text as NSString) { return hit.blocks }
        let blocks = parseUncached(text)
        if text.count > 64 { cache.setObject(BlockBox(blocks), forKey: text as NSString) }
        return blocks
    }

    static func parseUncached(_ text: String) -> [Block] {
        var blocks: [Block] = []
        var paragraph: [String] = []
        var bullets: [(level: Int, text: String)] = []
        var tableLines: [String] = []
        var numbered: [(label: String, text: String)] = []
        var quote: [String] = []
        var codeLang: String?
        var code: [String] = []
        var inCode = false

        func flushTable() {
            guard !tableLines.isEmpty else { return }
            let rows = tableLines.map(splitTableRow).filter { !$0.isEmpty }
            let isSeparator: ([String]) -> Bool = { $0.allSatisfy { $0.allSatisfy { $0 == "-" || $0 == ":" || $0 == " " } && $0.contains("-") } }
            let body = rows.filter { !isSeparator($0) }
            if let header = body.first, body.count >= 1, rows.contains(where: isSeparator) {
                blocks.append(.table(header: header, rows: Array(body.dropFirst())))
            } else {
                blocks.append(.paragraph(tableLines.joined(separator: " ")))
            }
            tableLines = []
        }
        func flush() {
            flushTable()
            if !paragraph.isEmpty { blocks.append(.paragraph(paragraph.joined(separator: " "))); paragraph = [] }
            if !bullets.isEmpty { blocks.append(.bullets(bullets)); bullets = [] }
            if !numbered.isEmpty { blocks.append(.numbered(numbered)); numbered = [] }
            if !quote.isEmpty { blocks.append(.quote(quote.joined(separator: " "))); quote = [] }
        }

        for rawLine in text.components(separatedBy: "\n") {
            let line = rawLine
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.hasPrefix("```") {
                if inCode {
                    blocks.append(.code(codeLang, code.joined(separator: "\n")))
                    code = []; codeLang = nil; inCode = false
                } else {
                    flush()
                    let lang = trimmed.dropFirst(3).trimmingCharacters(in: .whitespaces)
                    codeLang = lang.isEmpty ? nil : lang
                    inCode = true
                }
                continue
            }
            if inCode { code.append(line); continue }
            if trimmed.isEmpty { flush(); continue }
            if trimmed.hasPrefix("|"), trimmed.count > 1 {
                if !paragraph.isEmpty || !bullets.isEmpty || !numbered.isEmpty || !quote.isEmpty { flush() }
                tableLines.append(trimmed)
                continue
            }
            if !tableLines.isEmpty { flushTable() }
            if trimmed == "---" || trimmed == "***" { flush(); blocks.append(.rule); continue }
            if let h = headingLevel(trimmed) {
                flush()
                blocks.append(.heading(h, String(trimmed.drop(while: { $0 == "#" })).trimmingCharacters(in: .whitespaces)))
                continue
            }
            if trimmed.hasPrefix("- ") || trimmed.hasPrefix("* ") || trimmed.hasPrefix("• ") || trimmed.hasPrefix("+ ") {
                if !paragraph.isEmpty || !numbered.isEmpty || !quote.isEmpty { flush() }
                let indent = line.prefix { $0 == " " || $0 == "\t" }.reduce(0) { $0 + ($1 == "\t" ? 4 : 1) }
                let level = min(3, indent / 2)
                bullets.append((level: level, text: String(trimmed.dropFirst(2)).trimmingCharacters(in: .whitespaces)))
                continue
            }
            if let m = numberedItem.firstMatch(in: trimmed, range: NSRange(trimmed.startIndex..., in: trimmed)),
               let full = Range(m.range, in: trimmed), let num = Range(m.range(at: 1), in: trimmed) {
                if !paragraph.isEmpty || !bullets.isEmpty || !quote.isEmpty { flush() }
                numbered.append((label: trimmed[num] + ".", text: String(trimmed[full.upperBound...])))
                continue
            }
            if trimmed.hasPrefix("> ") || trimmed == ">" {
                if !paragraph.isEmpty || !bullets.isEmpty || !numbered.isEmpty { flush() }
                quote.append(String(trimmed.dropFirst(1)).trimmingCharacters(in: .whitespaces))
                continue
            }
            // Continuation of a list item (indented) stays with the item.
            if line.hasPrefix("  "), !bullets.isEmpty { bullets[bullets.count - 1].text += " " + trimmed; continue }
            if line.hasPrefix("  "), !numbered.isEmpty { numbered[numbered.count - 1].text += " " + trimmed; continue }
            if !bullets.isEmpty || !numbered.isEmpty || !quote.isEmpty { flush() }
            paragraph.append(trimmed)
        }
        if inCode { blocks.append(.code(codeLang, code.joined(separator: "\n"))) } // unterminated while streaming
        flush()
        return blocks
    }

    /// "| a | b |" → ["a", "b"] (a trailing pipe is optional).
    private static func splitTableRow(_ line: String) -> [String] {
        var cells: [String] = []
        var current = ""
        var inCode = false
        for ch in line.dropFirst() {          // skip the leading pipe
            if ch == "`" { inCode.toggle() }
            if ch == "|", !inCode { cells.append(current.trimmingCharacters(in: .whitespaces)); current = "" }
            else { current.append(ch) }
        }
        let tail = current.trimmingCharacters(in: .whitespaces)
        if !tail.isEmpty { cells.append(tail) }
        return cells.map { $0.replacingOccurrences(of: "<br>", with: " ").replacingOccurrences(of: "<br/>", with: " ") }
    }

    private static func headingLevel(_ s: String) -> Int? {
        var n = 0
        for ch in s { if ch == "#" { n += 1 } else { break } }
        guard n > 0, n <= 4, s.dropFirst(n).first == " " else { return nil }
        return n
    }
}

struct CodeBlock: View {
    let language: String?
    let code: String
    @State private var copied = false

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Text(language ?? "code").font(.caption2).foregroundStyle(.secondary)
                Spacer()
                Button {
                    UIPasteboard.general.string = code
                    copied = true
                    DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) { copied = false }
                } label: {
                    Label(copied ? "Copied" : "Copy", systemImage: copied ? "checkmark" : "doc.on.doc").font(.caption2)
                }
                .buttonStyle(.plain).foregroundStyle(.secondary)
            }
            .padding(.horizontal, 10).padding(.vertical, 6)
            Divider()
            ScrollView(.horizontal, showsIndicators: false) {
                Text(code)
                    .font(.system(.footnote, design: .monospaced))
                    .padding(10)
            }
        }
        .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 10, style: .continuous))
    }
}


/// Pipe table with fixed-width, wrapping columns; wide tables scroll sideways instead of squeezing the chat.
/// (SwiftUI's Grid does not grow row heights for wrapped text inside a horizontal ScrollView.)
struct MarkdownTable: View {
    let header: [String]
    let rows: [[String]]
    let inline: (String) -> Text

    private var columns: Int { max(header.count, rows.map(\.count).max() ?? 0) }
    /// Width per column from its longest cell: 6.5 pt per character, clamped to 70–220 pt.
    private var widths: [CGFloat] {
        (0..<columns).map { i in
            let longest = ([header] + rows).map { i < $0.count ? $0[i].count : 0 }.max() ?? 0
            return min(220, max(70, CGFloat(longest) * 6.5))
        }
    }

    var body: some View {
        let w = widths
        ScrollView(.horizontal, showsIndicators: false) {
            VStack(alignment: .leading, spacing: 0) {
                row(header, widths: w, bold: true)
                Divider().padding(.vertical, 4)
                ForEach(Array(rows.enumerated()), id: \.offset) { _, r in
                    row(r, widths: w, bold: false).padding(.vertical, 3)
                }
            }
            .padding(10)
        }
        .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 10, style: .continuous))
    }

    private func row(_ cells: [String], widths: [CGFloat], bold: Bool) -> some View {
        HStack(alignment: .top, spacing: 14) {
            ForEach(0..<widths.count, id: \.self) { i in
                inline(i < cells.count ? cells[i] : "")
                    .font(bold ? .footnote.weight(.semibold) : .footnote)
                    .fixedSize(horizontal: false, vertical: true)
                    .frame(width: widths[i], alignment: .topLeading)
            }
        }
    }
}
