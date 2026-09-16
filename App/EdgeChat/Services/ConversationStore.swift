import Foundation
import EdgeChatCore

/// JSON-on-disk conversation persistence: Documents/Conversations/<id>.json (+ <id>/ for attachments).
@MainActor @Observable
final class ConversationStore {
    private(set) var conversations: [Conversation] = []
    let root: URL

    init() {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        root = docs.appendingPathComponent("Conversations", isDirectory: true)
        try? FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    func loadAll() {
        let files = (try? FileManager.default.contentsOfDirectory(at: root, includingPropertiesForKeys: nil)) ?? []
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        conversations = files.filter { $0.pathExtension == "json" }
            .compactMap { try? decoder.decode(Conversation.self, from: Data(contentsOf: $0)) }
            .map { c in
                // A reply the app was killed in the middle of leaves an empty bubble; drop it.
                var c = c
                c.messages.removeAll { $0.role == .assistant && $0.content.isEmpty && $0.error == nil && $0.stats == nil }
                return c
            }
            .sorted { $0.updatedAt > $1.updatedAt }
    }

    func conversation(_ id: UUID) -> Conversation? { conversations.first { $0.id == id } }

    @discardableResult
    func create() -> Conversation {
        let c = Conversation()
        conversations.insert(c, at: 0)
        save(c)
        return c
    }

    /// `persist: false` updates memory only (used for streaming token flushes); call again with `persist: true` to write.
    func update(_ conversation: Conversation, touch: Bool = true, persist: Bool = true) {
        var c = conversation
        if touch { c.updatedAt = Date() }
        if let i = conversations.firstIndex(where: { $0.id == c.id }) {
            conversations[i] = c
        } else {
            conversations.insert(c, at: 0)
        }
        if touch { conversations.sort { $0.updatedAt > $1.updatedAt } }
        if persist { save(c) }
    }

    func modify(_ id: UUID, touch: Bool = true, persist: Bool = true, _ body: (inout Conversation) -> Void) {
        guard var c = conversation(id) else { return }
        body(&c)
        update(c, touch: touch, persist: persist)
    }

    func delete(_ id: UUID) {
        conversations.removeAll { $0.id == id }
        try? FileManager.default.removeItem(at: fileURL(id))
        try? FileManager.default.removeItem(at: attachmentsDirectory(id))
    }

    func attachmentsDirectory(_ id: UUID) -> URL {
        let dir = root.appendingPathComponent(id.uuidString, isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    func memoryIndexURL(_ id: UUID) -> URL {
        attachmentsDirectory(id).appendingPathComponent("memory-index.json")
    }

    func attachmentURL(conversationID: UUID, attachment: Attachment) -> URL {
        attachmentsDirectory(conversationID).appendingPathComponent(attachment.storedFileName)
    }

    private func fileURL(_ id: UUID) -> URL { root.appendingPathComponent("\(id.uuidString).json") }

    /// Serial so writes land in the order they were requested (a concurrent queue let a stale flush overwrite the final stats).
    private static let saveQueue = DispatchQueue(label: "edgechat.conversation-save", qos: .utility)

    private func save(_ c: Conversation) {
        let url = fileURL(c.id)
        Self.saveQueue.async {
            let encoder = JSONEncoder()
            encoder.dateEncodingStrategy = .iso8601
            encoder.outputFormatting = [.sortedKeys]
            if let data = try? encoder.encode(c) { try? data.write(to: url, options: .atomic) }
        }
    }
}
