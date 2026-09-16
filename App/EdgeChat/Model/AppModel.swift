import Foundation
import SwiftUI
import EdgeChatCore

/// Root object: settings, model files, conversations, and the inference controller.
@MainActor @Observable
final class AppModel {
    var settings: AppSettings { didSet { settings.save() } }
    let models: ModelManager
    let store: ConversationStore
    let engine: EngineController

    var selectedConversationID: UUID?
    var showModels = false
    var showSettings = false

    init() {
        DiagnosticsLog.install()
        let settings = AppSettings.load()
        self.settings = settings
        self.models = ModelManager()
        self.store = ConversationStore()
        self.engine = EngineController()
        engine.app = self
        models.app = self
    }

    func start() async {
        store.loadAll()
        models.refresh()
        if selectedConversationID == nil { selectedConversationID = store.conversations.first?.id }
        if settings.autoLoadLastModel, let installed = models.activeInstalledModel() {
            await engine.load(installed)
        } else if models.installed.isEmpty {
            showModels = true
        }
        #if DEBUG
        await runAutomationIfRequested()
        #endif
    }

    func newConversation() {
        let c = store.create()
        selectedConversationID = c.id
    }

    var selectedConversation: Conversation? {
        guard let id = selectedConversationID else { return nil }
        return store.conversation(id)
    }

    var physicalMemory: UInt64 { ProcessInfo.processInfo.physicalMemory }

    #if DEBUG
    /// Simulator automation: `-EdgeChatAutoPrompt "first ||| second ||| third"` loads the first installed model,
    /// creates a chat and sends each prompt in turn. `-EdgeChatAutoContext 512` overrides the context length first.
    private func runAutomationIfRequested() async {
        // `-EdgeChatShowScreen list|models|settings` opens a screen at launch (used for README screenshots).
        switch UserDefaults.standard.string(forKey: "EdgeChatShowScreen") {
        case "list": selectedConversationID = nil
        case "models": showModels = true
        case "settings": showSettings = true
        default: break
        }
        let ctx = UserDefaults.standard.integer(forKey: "EdgeChatAutoContext")
        if ctx > 0 { settings.engine.contextLength = UInt32(ctx) }
        guard let raw = UserDefaults.standard.string(forKey: "EdgeChatAutoPrompt") else { return }
        if !engine.isReady || engine.loadedConfig != settings.engine, let first = models.installed.first { await engine.load(first) }
        newConversation()
        guard let id = selectedConversationID else { return }
        for prompt in raw.components(separatedBy: "|||").map({ $0.trimmingCharacters(in: .whitespaces) }) where !prompt.isEmpty {
            await engine.send(conversationID: id, text: prompt, attachments: [])
        }
    }
    #endif
}


/// Appends engine warnings/errors to Documents/edgechat.log so problems on a phone can be shared.
enum DiagnosticsLog {
    static let url: URL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent("edgechat.log")
    private static let queue = DispatchQueue(label: "com.edgechat.log", qos: .utility)
    private static let formatter: ISO8601DateFormatter = { let f = ISO8601DateFormatter(); f.formatOptions = [.withInternetDateTime, .withFractionalSeconds]; return f }()

    static func install() {
        LlamaEngine.logHandler = { level, text in
            guard level.rawValue >= LlamaLogLevel.warn.rawValue else { return }
            append("[llama \(level)] " + text)
        }
        append("--- launch \(ProcessInfo.processInfo.processName) \(Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "") RAM \(ProcessInfo.processInfo.physicalMemory / 1_048_576) MB\n")
    }

    static func append(_ line: String) {
        queue.async {
            let entry = formatter.string(from: Date()) + " " + line + (line.hasSuffix("\n") ? "" : "\n")
            guard let data = entry.data(using: .utf8) else { return }
            if let size = try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize, size > 2_000_000 {
                try? FileManager.default.removeItem(at: url) // simple rotation
            }
            if let h = try? FileHandle(forWritingTo: url) {
                defer { try? h.close() }
                _ = try? h.seekToEnd(); try? h.write(contentsOf: data)
            } else {
                try? data.write(to: url)
            }
        }
    }
}
