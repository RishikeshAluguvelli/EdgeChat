import SwiftUI
import EdgeChatCore

struct ChatView: View {
    @Environment(AppModel.self) private var app
    let conversationID: UUID
    @State private var showRename = false
    @State private var newTitle = ""
    @State private var showContextInfo = false

    private var conversation: Conversation? { app.store.conversation(conversationID) }

    /// (used cells, context length) after the most recent completed reply in this chat.
    private var contextUsage: (used: Int, total: Int)? {
        guard let c = conversation,
              let stats = c.messages.last(where: { $0.role == .assistant && $0.stats?.contextLength != nil })?.stats,
              let total = stats.contextLength, total > 0 else { return nil }
        return (min(total, stats.promptTokens + stats.generatedTokens), total)
    }

    var body: some View {
        VStack(spacing: 0) {
            if !app.engine.isReady { ModelBanner() }
            ScrollView {
                // Not lazy: with estimated row heights the initial jump to the bottom of a long chat lands short and
                // leaves a blank gap; rows are Equatable and their Markdown is memoized, so eager layout is cheap.
                VStack(alignment: .leading, spacing: 14) {
                    if let c = conversation {
                        ForEach(Array(c.messages.enumerated()), id: \.element.id) { i, m in
                            if i == c.summaryCoversMessages, let summary = c.summary, c.summaryCoversMessages > 0 {
                                SummaryDivider(count: c.summaryCoversMessages, summary: summary)
                            }
                            MessageRow(message: m, conversationID: conversationID,
                                       isStreaming: app.engine.isGenerating && app.engine.generatingConversationID == conversationID && m.id == c.messages.last?.id,
                                       isLast: m.id == c.messages.last?.id)
                                .equatable()
                        }
                    }
                    Color.clear.frame(height: 4).id("bottom")
                }
                .padding(.horizontal, 12)
                .padding(.top, 12)
            }
            .defaultScrollAnchor(.bottom)
            .scrollDismissesKeyboard(.interactively)
            if app.engine.activity == .retrieving || app.engine.activity == .summarizing || app.engine.activity == .saving {
                HStack(spacing: 8) {
                    ProgressView().controlSize(.small)
                    Text(app.engine.activity == .summarizing ? (app.engine.isCompacting ? "Compacting older messages into memory…" : "Summarizing earlier conversation…")
                         : app.engine.activity == .retrieving ? "Recalling relevant context…" : "Saving conversation cache…")
                        .font(.caption).foregroundStyle(.secondary)
                    Spacer()
                }
                .padding(.horizontal, 14).padding(.vertical, 6)
            }
            ComposerView(conversationID: conversationID)
        }
        .navigationTitle(conversation?.title ?? "Chat")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            if let u = contextUsage {
                ToolbarItem(placement: .topBarTrailing) {
                    Button { showContextInfo = true } label: {
                        ContextGauge(fraction: Double(u.used) / Double(u.total))
                    }
                    .accessibilityLabel("Context window \(Int(Double(u.used) / Double(u.total) * 100)) percent used")
                }
            }
            ToolbarItem(placement: .topBarTrailing) {
                Menu {
                    Button { newTitle = conversation?.title ?? ""; showRename = true } label: { Label("Rename", systemImage: "pencil") }
                    Button { Task { await app.engine.clearCache() } } label: { Label("Clear KV cache", systemImage: "memorychip") }
                    Button(role: .destructive) {
                        app.engine.forgetConversation(conversationID)
                        app.store.delete(conversationID)
                        app.selectedConversationID = nil
                    } label: { Label("Delete chat", systemImage: "trash") }
                } label: { Image(systemName: "ellipsis.circle") }
            }
        }
        .alert("Context window", isPresented: $showContextInfo) {
            Button("OK") {}
        } message: {
            if let u = contextUsage {
                let pct = Int(Double(u.used) / Double(u.total) * 100)
                Text("\(u.used.formatted()) of \(u.total.formatted()) tokens (\(pct)%) were in use after the last reply, including any photos. When usage passes about 75%, the oldest messages are folded into a memory summary automatically\(app.settings.autoCompact ? " right after the reply" : " before the next reply"), and relevant older messages are recalled per turn.")
            } else {
                Text("No reply yet.")
            }
        }
        .alert("Rename chat", isPresented: $showRename) {
            TextField("Title", text: $newTitle)
            Button("Save") { app.store.modify(conversationID, touch: false) { $0.title = newTitle } }
            Button("Cancel", role: .cancel) {}
        }
        .alert("Attachment problem", isPresented: Binding(get: { app.engine.lastError != nil }, set: { if !$0 { app.engine.lastError = nil } })) {
            Button("OK") { app.engine.lastError = nil }
        } message: { Text(app.engine.lastError ?? "") }
    }
}

/// Shown above the transcript while no model is ready.
struct ModelBanner: View {
    @Environment(AppModel.self) private var app
    var body: some View {
        HStack(spacing: 10) {
            switch app.engine.state {
            case .loading(let p, let name):
                ProgressView(value: p).frame(width: 80)
                Text("Loading \(name)…").font(.footnote)
            case .failed(let e):
                Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.red)
                Text(e).font(.footnote).lineLimit(2)
                Spacer()
                Button("Models") { app.showModels = true }.font(.footnote)
            default:
                Image(systemName: "arrow.down.circle").foregroundStyle(.tint)
                Text(app.models.installed.isEmpty ? "Download a model to start chatting." : "Pick a model to load.").font(.footnote)
                Spacer()
                Button("Models") { app.showModels = true }.font(.footnote).buttonStyle(.borderedProminent).controlSize(.small)
            }
        }
        .padding(.horizontal, 14).padding(.vertical, 8)
        .background(.ultraThinMaterial)
    }
}


/// Marks where older messages were folded into the model-written memory summary.
struct SummaryDivider: View {
    let count: Int
    let summary: String
    @State private var show = false

    var body: some View {
        Button { show = true } label: {
            HStack(spacing: 8) {
                Rectangle().fill(.quaternary).frame(height: 1)
                Label("\(count) earlier messages summarized", systemImage: "brain.head.profile")
                    .font(.caption2).foregroundStyle(.secondary).fixedSize()
                Rectangle().fill(.quaternary).frame(height: 1)
            }
        }
        .buttonStyle(.plain)
        .sheet(isPresented: $show) {
            NavigationStack {
                ScrollView { Text(summary).textSelection(.enabled).padding() }
                    .navigationTitle("Conversation memory").navigationBarTitleDisplayMode(.inline)
                    .toolbar { ToolbarItem(placement: .confirmationAction) { Button("Done") { show = false } } }
            }
        }
    }
}


/// Small ring showing how full the context window was after the last reply.
struct ContextGauge: View {
    let fraction: Double
    private var color: Color { fraction >= 0.9 ? .red : fraction >= 0.7 ? .orange : .secondary }
    var body: some View {
        ZStack {
            Circle().stroke(Color.secondary.opacity(0.25), lineWidth: 2.5)
            Circle().trim(from: 0, to: min(1, max(0.02, fraction)))
                .stroke(color, style: StrokeStyle(lineWidth: 2.5, lineCap: .round))
                .rotationEffect(.degrees(-90))
            Text("\(Int((fraction * 100).rounded()))")
                .font(.system(size: 7, weight: .semibold, design: .rounded))
                .foregroundStyle(color)
        }
        .frame(width: 22, height: 22)
    }
}
