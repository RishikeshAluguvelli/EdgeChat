import Foundation
import UniformTypeIdentifiers
import EdgeChatCore

/// Something the user picked but has not sent yet.
struct PendingAttachment: Identifiable, Hashable {
    enum Source: Hashable {
        case imageData(Data)
        case fileURL(URL)
    }
    let id = UUID()
    let name: String
    let source: Source
    var isImage: Bool {
        switch source {
        case .imageData: return true
        case .fileURL(let url): return (UTType(filenameExtension: url.pathExtension) ?? .data).conforms(to: .image)
        }
    }
}

enum AttachmentIngest {
    /// Stores files under the conversation folder and pre-computes the text the model will see.
    static func ingest(_ pending: [PendingAttachment], attachmentsDirectory dir: URL,
                       settings: AppSettings) async -> (attachments: [Attachment], errors: [String]) {
        var out: [Attachment] = []
        var errors: [String] = []
        let docs = pending.filter { !$0.isImage }
        // Character budget shared by all documents in this message (~3.5 chars/token, leave room for the reply).
        let ctx = Int(settings.engine.contextLength)
        let perDocChars = max(2_000, ((ctx - settings.sampling.maxTokens - 600) * 7 / 2) / max(1, docs.count))

        for item in pending {
            do {
                switch item.source {
                case .imageData(let data):
                    out.append(try await storeImage(data: data, name: item.name, dir: dir, settings: settings))
                case .fileURL(let url):
                    let scoped = url.startAccessingSecurityScopedResource()
                    defer { if scoped { url.stopAccessingSecurityScopedResource() } }
                    if item.isImage {
                        let data = try Data(contentsOf: url)
                        out.append(try await storeImage(data: data, name: item.name, dir: dir, settings: settings))
                    } else {
                        let id = UUID()
                        let stored = "\(id.uuidString).\(url.pathExtension.isEmpty ? "txt" : url.pathExtension)"
                        let dest = dir.appendingPathComponent(stored)
                        try? FileManager.default.removeItem(at: dest)
                        try FileManager.default.copyItem(at: url, to: dest)
                        let extraction = try await AttachmentProcessor.extractText(from: dest, maxCharacters: perDocChars, ocrFallback: settings.ocrForImages)
                        let size = Int64((try? dest.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0)
                        out.append(Attachment(id: id, kind: .document, fileName: item.name, storedFileName: stored, byteCount: size,
                                              extractedText: extraction.text, textTruncated: extraction.truncated,
                                              pageCount: extraction.pageCount, ingestion: extraction.method))
                    }
                }
            } catch {
                errors.append("\(item.name): \(error.localizedDescription)")
            }
        }
        return (out, errors)
    }

    private static func storeImage(data: Data, name: String, dir: URL, settings: AppSettings) async throws -> Attachment {
        let id = UUID()
        let (input, cg) = try AttachmentProcessor.makeImageInput(from: data, id: id.uuidString, maxDimension: settings.maxImageDimension)
        guard let jpeg = AttachmentProcessor.jpegData(cg) else { throw AttachmentError.imageDecodeFailed }
        let stored = "\(id.uuidString).jpg"
        try jpeg.write(to: dir.appendingPathComponent(stored), options: .atomic)
        var text: String?
        if settings.ocrForImages {
            text = await AttachmentProcessor.describeImageAsText(cg, fileName: name)
        }
        return Attachment(id: id, kind: .image, fileName: name, storedFileName: stored, byteCount: Int64(jpeg.count),
                          extractedText: text, imageWidth: input.width, imageHeight: input.height)
    }

    /// The exact text the engine sees for a message (deterministic, so the KV prefix cache keeps matching).
    static func engineText(for message: Message, modelHasVision: Bool) -> String {
        var parts: [String] = []
        if let recalled = message.recalledContext, !recalled.isEmpty { parts.append(recalled) }
        for a in message.attachments {
            switch a.kind {
            case .document:
                let pages = a.pageCount.map { " pages=\"\($0)\"" } ?? ""
                parts.append("<document name=\"\(a.fileName)\"\(pages)>\n\(a.extractedText ?? "(no text)")\n</document>")
            case .image:
                if !modelHasVision, let t = a.extractedText { parts.append(t) }
            }
        }
        parts.append(message.content)
        return parts.filter { !$0.isEmpty }.joined(separator: "\n\n")
    }

    /// Rebuilds the vision inputs for a message from the stored JPEGs.
    static func imageInputs(for message: Message, attachmentsDirectory dir: URL, settings: AppSettings) -> [ImageInput] {
        message.attachments.filter { $0.kind == .image }.compactMap { a in
            let url = dir.appendingPathComponent(a.storedFileName)
            guard let data = try? Data(contentsOf: url) else { return nil }
            return try? AttachmentProcessor.makeImageInput(from: data, id: a.id.uuidString, maxDimension: settings.maxImageDimension).input
        }
    }
}
