import SwiftUI
import UniformTypeIdentifiers
import EdgeChatCore

struct ModelsView: View {
    @Environment(AppModel.self) private var app
    @Environment(\.dismiss) private var dismiss
    @State private var showImporter = false

    private var ram: UInt64 { app.physicalMemory }

    var body: some View {
        NavigationStack {
            List {
                Section {
                    LabeledContent("Device memory", value: TextUtils.formatBytes(Int64(ram)))
                    LabeledContent("Free storage", value: TextUtils.formatBytes(app.models.freeDiskSpace))
                    LabeledContent("Models on device", value: TextUtils.formatBytes(app.models.storageUsed))
                } footer: {
                    Text("Models run entirely on this device. Download once over Wi-Fi; after that no connection is needed. 4B models need an 8 GB iPhone (15 Pro or newer); 6 GB phones should use the 2B/1.7B tier.")
                }
                ForEach(ModelSpec.Tier.allCases, id: \.self) { tier in
                    Section(tier.rawValue) {
                        ForEach(ModelCatalog.models.filter { $0.tier == tier }) { spec in
                            CatalogRow(spec: spec)
                                .swipeActions(edge: .trailing) {
                                    if let installed = app.models.installed(for: spec), !app.engine.isGenerating {
                                        Button(role: .destructive) {
                                            Task { if app.engine.loadedModel?.id == spec.id { await app.engine.unload() }; app.models.delete(installed) }
                                        } label: { Label("Delete", systemImage: "trash") }
                                    }
                                }
                        }
                    }
                }
                let imported = app.models.installed.filter { $0.spec == nil }
                Section {
                    ForEach(imported) { model in ImportedRow(model: model) }
                    Button { showImporter = true } label: { Label("Import GGUF from Files", systemImage: "square.and.arrow.down") }
                } header: {
                    Text("Your own GGUFs")
                } footer: {
                    Text("Any llama.cpp-compatible GGUF works. You can also drop files into EdgeChat's folder in the Files app or via Finder. Import an mmproj-*.gguf alongside a vision model and pair it from the row's menu.")
                }
                if let err = app.models.lastError {
                    Section { Text(err).foregroundStyle(.red).font(.footnote) }
                }
            }
            .navigationTitle("Models")
            .toolbar { ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } } }
            .fileImporter(isPresented: $showImporter, allowedContentTypes: [UTType(filenameExtension: "gguf") ?? .data, .data], allowsMultipleSelection: true) { result in
                if case .success(let urls) = result { for u in urls { app.models.importModel(from: u) } }
            }
            .onAppear { app.models.refresh() }
        }
    }
}

struct CatalogRow: View {
    @Environment(AppModel.self) private var app
    let spec: ModelSpec
    @State private var confirmDelete = false

    private var installed: InstalledModel? { app.models.installed(for: spec) }
    private var isActive: Bool { app.engine.loadedModel?.id == spec.id }
    private var isLoading: Bool { if case .loading = app.engine.state, app.settings.activeModelID == spec.id { return true }; return false }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .firstTextBaseline) {
                Text(spec.name).font(.headline)
                Spacer()
                actionView
            }
            HStack(spacing: 6) {
                badge(spec.parameters + " · " + spec.quantization, color: .secondary)
                if spec.hasVision { badge("Vision", color: .purple) }
                if spec.capabilities.contains(.reasoning) { badge("Reasoning", color: .orange) }
                if spec.id == ModelCatalog.recommendedID { badge("Recommended", color: .green) }
            }
            Text(spec.summary).font(.footnote).foregroundStyle(.secondary)
            HStack {
                Text(TextUtils.formatBytes(spec.totalBytes)).font(.caption).foregroundStyle(.secondary)
                Text("·").foregroundStyle(.secondary)
                fitLabel
            }
            if case .downloading(let w, let t) = app.models.downloads[spec.id] {
                ProgressView(value: t > 0 ? Double(w) / Double(t) : 0) {
                    Text("\(TextUtils.formatBytes(w)) of \(TextUtils.formatBytes(t))").font(.caption2).foregroundStyle(.secondary)
                }
            } else if case .paused(let w, let t) = app.models.downloads[spec.id] {
                ProgressView(value: t > 0 ? Double(w) / Double(t) : 0) { Text("Paused").font(.caption2).foregroundStyle(.secondary) }
            } else if case .failed(let e) = app.models.downloads[spec.id] {
                Text(e).font(.caption2).foregroundStyle(.red)
            }
        }
        .padding(.vertical, 4)
    }

    @ViewBuilder
    private var actionView: some View {
        switch app.models.downloads[spec.id] {
        case .downloading:
            HStack(spacing: 12) {
                Button { app.models.pause(spec) } label: { Image(systemName: "pause.circle") }
                Button(role: .destructive) { app.models.cancel(spec) } label: { Image(systemName: "xmark.circle") }
            }.buttonStyle(.borderless)
        case .paused:
            HStack(spacing: 12) {
                Button { app.models.resume(spec) } label: { Image(systemName: "play.circle") }
                Button(role: .destructive) { app.models.cancel(spec) } label: { Image(systemName: "xmark.circle") }
            }.buttonStyle(.borderless)
        case .failed:
            Button("Retry") { app.models.download(spec) }.buttonStyle(.bordered).controlSize(.small)
        case nil:
            if let installed {
                HStack(spacing: 12) {
                    if isActive {
                        Label("Active", systemImage: "checkmark.circle.fill").foregroundStyle(.green).font(.subheadline)
                    } else if isLoading {
                        ProgressView()
                    } else {
                        Button("Use") { Task { await app.engine.load(installed) } }.buttonStyle(.borderedProminent).controlSize(.small)
                    }
                    Button(role: .destructive) { confirmDelete = true } label: { Image(systemName: "trash") }
                        .buttonStyle(.borderless).tint(.red)
                        .disabled(isLoading || app.engine.isGenerating)
                }
                .confirmationDialog("Delete \(spec.name)?", isPresented: $confirmDelete, titleVisibility: .visible) {
                    Button("Delete \(TextUtils.formatBytes(installed.sizeBytes))", role: .destructive) {
                        Task { if isActive { await app.engine.unload() }; app.models.delete(installed) }
                    }
                } message: { Text("Frees the storage on this phone. You can download it again any time.") }
            } else {
                Button("Get") { app.models.download(spec) }.buttonStyle(.bordered).controlSize(.small)
            }
        }
    }

    @ViewBuilder
    private var fitLabel: some View {
        switch spec.fit(physicalMemory: app.physicalMemory, contextLength: Int(app.settings.engine.contextLength)) {
        case .good: Label("Fits this device", systemImage: "checkmark.circle").font(.caption).foregroundStyle(.green)
        case .tight: Label("Tight on memory", systemImage: "exclamationmark.circle").font(.caption).foregroundStyle(.orange)
        case .tooLarge: Label("Likely too large", systemImage: "xmark.circle").font(.caption).foregroundStyle(.red)
        }
    }

    private func badge(_ text: String, color: Color) -> some View {
        Text(text).font(.caption2.weight(.medium)).padding(.horizontal, 6).padding(.vertical, 2)
            .background(color.opacity(0.15), in: Capsule()).foregroundStyle(color)
    }
}

struct ImportedRow: View {
    @Environment(AppModel.self) private var app
    let model: InstalledModel
    private var isActive: Bool { app.engine.loadedModel?.id == model.id }

    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 3) {
                Text(model.name).font(.headline).lineLimit(1)
                Text(TextUtils.formatBytes(model.sizeBytes) + (model.mmprojURL.map { " · vision: \($0.lastPathComponent)" } ?? "")).font(.caption).foregroundStyle(.secondary).lineLimit(1)
            }
            Spacer()
            if isActive {
                Label("Active", systemImage: "checkmark.circle.fill").foregroundStyle(.green).labelStyle(.iconOnly)
            } else {
                Button("Use") { Task { await app.engine.load(model) } }.buttonStyle(.borderedProminent).controlSize(.small)
            }
            Menu {
                if !app.models.importedMmprojFiles.isEmpty {
                    Menu("Pair vision projector") {
                        ForEach(app.models.importedMmprojFiles, id: \.self) { u in
                            Button(u.lastPathComponent) { app.models.pair(model: model, mmproj: u) }
                        }
                        Button("None") { app.models.pair(model: model, mmproj: nil) }
                    }
                }
                Button(role: .destructive) { Task { if isActive { await app.engine.unload() }; app.models.delete(model) } } label: { Label("Delete", systemImage: "trash") }
            } label: { Image(systemName: "ellipsis.circle") }
        }
    }
}
