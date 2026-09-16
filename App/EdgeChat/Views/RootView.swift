import SwiftUI
import EdgeChatCore

struct RootView: View {
    @Environment(AppModel.self) private var app

    var body: some View {
        @Bindable var app = app
        NavigationSplitView {
            ConversationListView()
        } detail: {
            if let id = app.selectedConversationID, app.store.conversation(id) != nil {
                ChatView(conversationID: id)
                    .id(id)
            } else {
                EmptyChatView()
            }
        }
        .sheet(isPresented: $app.showModels) { ModelsView() }
        .sheet(isPresented: $app.showSettings) { SettingsView() }
    }
}

struct ConversationListView: View {
    @Environment(AppModel.self) private var app

    var body: some View {
        @Bindable var app = app
        List(selection: $app.selectedConversationID) {
            Section {
                ModelStatusRow()
            }
            Section("Chats") {
                ForEach(app.store.conversations) { c in
                    NavigationLink(value: c.id) {
                        VStack(alignment: .leading, spacing: 2) {
                            Text(c.title).lineLimit(1)
                            Text(c.updatedAt, style: .relative).font(.caption).foregroundStyle(.secondary)
                        }
                    }
                }
                .onDelete { idx in
                    for i in idx {
                        let id = app.store.conversations[i].id
                        app.engine.forgetConversation(id)
                        app.store.delete(id)
                    }
                    if let sel = app.selectedConversationID, app.store.conversation(sel) == nil { app.selectedConversationID = nil }
                }
            }
        }
        .navigationTitle("EdgeChat")
        .toolbar {
            ToolbarItem(placement: .topBarLeading) {
                Button { app.showSettings = true } label: { Image(systemName: "gearshape") }
            }
            ToolbarItem(placement: .topBarTrailing) {
                Button { app.newConversation() } label: { Image(systemName: "square.and.pencil") }
            }
        }
        .overlay {
            if app.store.conversations.isEmpty {
                ContentUnavailableView {
                    Label("No chats yet", systemImage: "bubble.left.and.text.bubble.right")
                } description: {
                    Text("Everything runs on this device. No internet needed once a model is downloaded.")
                } actions: {
                    Button("New chat") { app.newConversation() }.buttonStyle(.borderedProminent)
                }
            }
        }
    }
}

/// Compact status of the loaded model; tap to open the model picker.
struct ModelStatusRow: View {
    @Environment(AppModel.self) private var app

    var body: some View {
        Button { app.showModels = true } label: {
            HStack(spacing: 12) {
                Image(systemName: "cpu")
                    .font(.title3)
                    .foregroundStyle(tint)
                    .frame(width: 28)
                VStack(alignment: .leading, spacing: 2) {
                    Text(title).font(.subheadline.weight(.semibold)).lineLimit(1)
                    Text(subtitle).font(.caption).foregroundStyle(.secondary).lineLimit(2)
                }
                Spacer()
                if case .loading(let p, _) = app.engine.state {
                    ProgressView(value: p).frame(width: 60)
                } else {
                    Image(systemName: "chevron.right").font(.caption).foregroundStyle(.tertiary)
                }
            }
        }
        .buttonStyle(.plain)
    }

    private var tint: Color {
        switch app.engine.state {
        case .ready: return .green
        case .loading: return .orange
        case .failed: return .red
        case .idle: return .secondary
        }
    }

    private var title: String {
        switch app.engine.state {
        case .ready: return app.engine.loadedModel?.name ?? "Model ready"
        case .loading(_, let name): return "Loading \(name)…"
        case .failed: return "Model failed to load"
        case .idle: return app.models.installed.isEmpty ? "Download a model to start" : "Choose a model"
        }
    }

    private var subtitle: String {
        switch app.engine.state {
        case .ready(let info):
            var bits = ["\(TextUtils.formatContext(info.contextLength)) context", info.gpuOffloaded ? "Metal" : "CPU"]
            if info.hasVision { bits.append("vision") }
            return bits.joined(separator: " · ")
        case .loading: return "Mapping weights into memory"
        case .failed(let e): return e
        case .idle: return "Runs fully offline"
        }
    }
}

struct EmptyChatView: View {
    @Environment(AppModel.self) private var app
    var body: some View {
        ContentUnavailableView {
            Label("EdgeChat", systemImage: "iphone.gen3.radiowaves.left.and.right")
        } description: {
            Text("Private, offline AI chat. Attach PDFs, text files and photos.")
        } actions: {
            Button("New chat") { app.newConversation() }.buttonStyle(.borderedProminent)
        }
    }
}
