import SwiftUI
import EdgeChatCore

/// Thumbnails and chips for a sent message's attachments.
struct AttachmentGallery: View {
    @Environment(AppModel.self) private var app
    let attachments: [Attachment]
    let conversationID: UUID
    @State private var preview: Attachment?

    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                ForEach(attachments) { a in
                    switch a.kind {
                    case .image:
                        StoredImage(url: app.store.attachmentURL(conversationID: conversationID, attachment: a))
                            .frame(width: 120, height: 120)
                            .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
                            .onTapGesture { preview = a }
                    case .document:
                        DocumentChip(name: a.fileName, detail: docDetail(a))
                    }
                }
            }
        }
        .sheet(item: $preview) { a in
            NavigationStack {
                StoredImage(url: app.store.attachmentURL(conversationID: conversationID, attachment: a), contentMode: .fit)
                    .navigationTitle(a.fileName).navigationBarTitleDisplayMode(.inline)
            }
        }
    }

    private func docDetail(_ a: Attachment) -> String {
        var parts: [String] = []
        if let p = a.pageCount { parts.append("\(p) page\(p == 1 ? "" : "s")") }
        parts.append(TextUtils.formatBytes(a.byteCount))
        if a.textTruncated { parts.append("trimmed to fit") }
        if a.ingestion == "pdf+ocr" { parts.append("OCR") }
        return parts.joined(separator: " · ")
    }
}

struct DocumentChip: View {
    let name: String
    let detail: String
    var onRemove: (() -> Void)? = nil

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: "doc.text.fill").foregroundStyle(.tint)
            VStack(alignment: .leading, spacing: 1) {
                Text(name).font(.footnote.weight(.medium)).lineLimit(1)
                Text(detail).font(.caption2).foregroundStyle(.secondary).lineLimit(1)
            }
            if let onRemove {
                Button(action: onRemove) { Image(systemName: "xmark.circle.fill").foregroundStyle(.secondary) }.buttonStyle(.plain)
            }
        }
        .padding(.horizontal, 10).padding(.vertical, 8)
        .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 12, style: .continuous))
        .frame(maxWidth: 220)
    }
}

/// Loads a JPEG from disk off the main thread.
struct StoredImage: View {
    let url: URL
    var contentMode: ContentMode = .fill
    @State private var image: UIImage?

    var body: some View {
        Group {
            if let image {
                Image(uiImage: image).resizable().aspectRatio(contentMode: contentMode)
            } else {
                Color(.secondarySystemBackground).overlay(ProgressView())
            }
        }
        .task(id: url) {
            let u = url
            image = await Task.detached(priority: .utility) { UIImage(contentsOfFile: u.path) }.value
        }
    }
}

/// Pending attachments in the composer.
struct PendingAttachmentStrip: View {
    @Binding var items: [PendingAttachment]

    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                ForEach(items) { item in
                    switch item.source {
                    case .imageData(let data):
                        ZStack(alignment: .topTrailing) {
                            if let ui = UIImage(data: data) {
                                Image(uiImage: ui).resizable().scaledToFill().frame(width: 64, height: 64)
                                    .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
                            }
                            removeButton(item)
                        }
                    case .fileURL(let url):
                        DocumentChip(name: item.name, detail: url.pathExtension.uppercased()) { items.removeAll { $0.id == item.id } }
                    }
                }
            }
            .padding(.horizontal, 12).padding(.top, 8)
        }
    }

    private func removeButton(_ item: PendingAttachment) -> some View {
        Button { items.removeAll { $0.id == item.id } } label: {
            Image(systemName: "xmark.circle.fill").font(.body).foregroundStyle(.white, .black.opacity(0.6))
        }
        .buttonStyle(.plain).offset(x: 4, y: -4)
    }
}
