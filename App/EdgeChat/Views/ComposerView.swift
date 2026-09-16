import SwiftUI
import PhotosUI
import UniformTypeIdentifiers
import EdgeChatCore

struct ComposerView: View {
    @Environment(AppModel.self) private var app
    let conversationID: UUID
    @State private var text = ""
    @State private var pending: [PendingAttachment] = []
    @State private var photoItems: [PhotosPickerItem] = []
    @State private var showFiles = false
    @State private var showCamera = false
    @State private var isPreparing = false
    @FocusState private var focused: Bool

    private var isGeneratingHere: Bool { app.engine.isGenerating && app.engine.generatingConversationID == conversationID }
    private var canSend: Bool {
        app.engine.isReady && !app.engine.isGenerating && !isPreparing && (!text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || !pending.isEmpty)
    }

    var body: some View {
        VStack(spacing: 0) {
            Divider()
            if !pending.isEmpty { PendingAttachmentStrip(items: $pending) }
            HStack(alignment: .bottom, spacing: 8) {
                Menu {
                    Button { showFiles = true } label: { Label("Document or file", systemImage: "doc") }
                    Button { showCamera = true } label: { Label("Take photo", systemImage: "camera") }
                        .disabled(!UIImagePickerController.isSourceTypeAvailable(.camera))
                } label: {
                    Image(systemName: "plus.circle.fill").font(.title2).foregroundStyle(.secondary)
                }
                PhotosPicker(selection: $photoItems, maxSelectionCount: 4, matching: .images) {
                    Image(systemName: "photo.on.rectangle").font(.title3).foregroundStyle(.secondary)
                }
                TextField(placeholder, text: $text, axis: .vertical)
                    .lineLimit(1...6)
                    .focused($focused)
                    .padding(.horizontal, 12).padding(.vertical, 8)
                    .background(Color(.secondarySystemBackground), in: RoundedRectangle(cornerRadius: 18, style: .continuous))
                if isGeneratingHere {
                    Button { app.engine.stop() } label: {
                        Image(systemName: "stop.circle.fill").font(.title).foregroundStyle(.red)
                    }
                } else {
                    Button(action: send) {
                        Image(systemName: "arrow.up.circle.fill").font(.title).foregroundStyle(canSend ? Color.accentColor : .secondary)
                    }
                    .disabled(!canSend)
                }
            }
            .padding(.horizontal, 12).padding(.vertical, 8)
            if !app.engine.hasVision, pending.contains(where: \.isImage), app.engine.isReady {
                Text("This model has no vision encoder. Images are read with on-device OCR; pick a “Vision” model for real image understanding.")
                    .font(.caption2).foregroundStyle(.secondary).padding(.horizontal, 14).padding(.bottom, 6)
            }
        }
        .background(.bar)
        .fileImporter(isPresented: $showFiles, allowedContentTypes: AttachmentProcessor.supportedDocumentTypes + AttachmentProcessor.supportedImageTypes, allowsMultipleSelection: true) { result in
            if case .success(let urls) = result {
                for url in urls { pending.append(PendingAttachment(name: url.lastPathComponent, source: .fileURL(url))) }
            }
        }
        .sheet(isPresented: $showCamera) {
            CameraPicker { image in
                if let data = image.jpegData(compressionQuality: 0.9) {
                    pending.append(PendingAttachment(name: "Photo \(Date().formatted(date: .omitted, time: .shortened)).jpg", source: .imageData(data)))
                }
            }
            .ignoresSafeArea()
        }
        .onChange(of: photoItems) { _, items in
            guard !items.isEmpty else { return }
            isPreparing = true
            Task {
                for item in items {
                    if let data = try? await item.loadTransferable(type: Data.self) {
                        pending.append(PendingAttachment(name: "Photo \(pending.count + 1).jpg", source: .imageData(data)))
                    }
                }
                photoItems = []
                isPreparing = false
            }
        }
    }

    private var placeholder: String {
        if !app.engine.isReady { return "Load a model to chat" }
        return "Message (offline)"
    }

    private func send() {
        guard canSend else { return }
        let t = text.trimmingCharacters(in: .whitespacesAndNewlines)
        let items = pending
        text = ""
        pending = []
        Task { await app.engine.send(conversationID: conversationID, text: t, attachments: items) }
    }
}

struct CameraPicker: UIViewControllerRepresentable {
    let onImage: (UIImage) -> Void
    @Environment(\.dismiss) private var dismiss

    func makeUIViewController(context: Context) -> UIImagePickerController {
        let picker = UIImagePickerController()
        picker.sourceType = .camera
        picker.delegate = context.coordinator
        return picker
    }
    func updateUIViewController(_ uiViewController: UIImagePickerController, context: Context) {}
    func makeCoordinator() -> Coordinator { Coordinator(self) }

    final class Coordinator: NSObject, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
        let parent: CameraPicker
        init(_ parent: CameraPicker) { self.parent = parent }
        func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey: Any]) {
            if let image = info[.originalImage] as? UIImage { parent.onImage(image) }
            parent.dismiss()
        }
        func imagePickerControllerDidCancel(_ picker: UIImagePickerController) { parent.dismiss() }
    }
}
