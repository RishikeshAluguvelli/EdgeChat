import SwiftUI
import EdgeChatCore

struct MessageRow: View, Equatable {
    @Environment(AppModel.self) private var app
    let message: Message
    let conversationID: UUID
    let isStreaming: Bool
    let isLast: Bool
    @State private var showReasoning = false

    static func == (a: MessageRow, b: MessageRow) -> Bool {
        a.message == b.message && a.conversationID == b.conversationID && a.isStreaming == b.isStreaming && a.isLast == b.isLast
    }

    var body: some View {
        HStack(alignment: .bottom) {
            if message.role == .user { Spacer(minLength: 48) }
            VStack(alignment: message.role == .user ? .trailing : .leading, spacing: 6) {
                if !message.attachments.isEmpty {
                    AttachmentGallery(attachments: message.attachments, conversationID: conversationID)
                }
                if message.role == .user {
                    if let recalled = message.recalledContext, !recalled.isEmpty {
                        RecalledContextChip(text: recalled)
                    }
                    if !message.content.isEmpty {
                        Text(message.content)
                            .textSelection(.enabled)
                            .padding(.horizontal, 14).padding(.vertical, 10)
                            .background(Color.accentColor.opacity(0.9), in: RoundedRectangle(cornerRadius: 18, style: .continuous))
                            .foregroundStyle(.white)
                    }
                } else {
                    assistantBody
                }
            }
            if message.role == .assistant { Spacer(minLength: 32) }
        }
        .contextMenu {
            Button { UIPasteboard.general.string = message.content } label: { Label("Copy", systemImage: "doc.on.doc") }
            if message.role == .assistant, isLast, !app.engine.isGenerating {
                Button { Task { await app.engine.regenerate(conversationID: conversationID) } } label: { Label("Regenerate", systemImage: "arrow.clockwise") }
            }
        }
    }

    @ViewBuilder
    private var assistantBody: some View {
        VStack(alignment: .leading, spacing: 8) {
            if let r = message.reasoning, !r.isEmpty {
                DisclosureGroup(isExpanded: $showReasoning) {
                    Text(r).font(.footnote).foregroundStyle(.secondary).textSelection(.enabled).padding(.top, 4)
                } label: {
                    Label(isStreaming && message.content.isEmpty ? "Thinking…" : "Thought process", systemImage: "brain")
                        .font(.footnote).foregroundStyle(.secondary)
                }
            }
            if message.content.isEmpty && isStreaming && (message.reasoning ?? "").isEmpty {
                TypingIndicator()
            } else if !message.content.isEmpty {
                MarkdownText(message.content)
            }
            if let e = message.error {
                Label(e, systemImage: "exclamationmark.triangle").font(.footnote).foregroundStyle(.red)
            }
            if let s = message.stats, app.settings.showStats, !isStreaming {
                Text(statsLine(s)).font(.caption2).foregroundStyle(.tertiary)
            }
            if isLast, !isStreaming, !app.engine.isGenerating, app.engine.canContinue(conversationID: conversationID) {
                Button {
                    Task { await app.engine.continueReply(conversationID: conversationID) }
                } label: {
                    Label("Continue", systemImage: "arrow.turn.down.right").font(.footnote.weight(.medium))
                }
                .buttonStyle(.bordered).controlSize(.small)
                .padding(.top, 2)
            }
        }
        .padding(.horizontal, 2)
    }

    private func statsLine(_ s: GenerationStats) -> String {
        var parts: [String] = []
        if s.generatedTokens > 0 { parts.append(String(format: "%.1f tok/s", s.tokensPerSecond)) }
        parts.append("\(s.generatedTokens) tokens")
        if s.promptTokens > 0 { parts.append("prompt \(s.promptTokens) (\(s.cachedTokens) cached)") }
        if s.prefillSeconds > 0.05 { parts.append(String(format: "prefill %.1fs", s.prefillSeconds)) }
        if let n = s.contextShifts, n > 0 { parts.append("context shifted ×\(n)") }
        if let c = s.contextLength { parts.append("ctx \(TextUtils.formatContext(c))") }
        switch s.stopReason {
        case .cancelled: parts.append("stopped")
        case .maxTokens: parts.append("reply length limit reached")
        case .contextFull: parts.append("paused: context window full")
        case .eos: break
        }
        return parts.joined(separator: " · ")
    }
}

struct TypingIndicator: View {
    @State private var phase = 0
    var body: some View {
        HStack(spacing: 4) {
            ForEach(0..<3) { i in
                Circle().fill(.secondary).frame(width: 7, height: 7).opacity(phase == i ? 1 : 0.35)
            }
        }
        .padding(.vertical, 6)
        .onAppear {
            Timer.scheduledTimer(withTimeInterval: 0.3, repeats: true) { _ in phase = (phase + 1) % 3 }
        }
    }
}


/// Shows that older snippets were retrieved for this turn; tap to read them.
struct RecalledContextChip: View {
    let text: String
    @State private var show = false
    private var count: Int { text.components(separatedBy: "\n- ").count - 1 + (text.contains("\n- ") ? 0 : 0) }

    var body: some View {
        Button { show = true } label: {
            Label("Recalled \(max(1, count)) earlier snippet\(count == 1 ? "" : "s")", systemImage: "magnifyingglass")
                .font(.caption2).foregroundStyle(.secondary)
        }
        .buttonStyle(.plain)
        .sheet(isPresented: $show) {
            NavigationStack {
                ScrollView { Text(text).font(.footnote).textSelection(.enabled).padding() }
                    .navigationTitle("Recalled context").navigationBarTitleDisplayMode(.inline)
                    .toolbar { ToolbarItem(placement: .confirmationAction) { Button("Done") { show = false } } }
            }
        }
    }
}
