import Foundation
import EdgeChatCore

struct InstalledModel: Identifiable, Hashable {
    var id: String            // catalog id, or "file:<name>" for imported GGUFs
    var name: String
    var modelURL: URL
    var mmprojURL: URL?
    var spec: ModelSpec?
    var sizeBytes: Int64
    var hasVision: Bool { mmprojURL != nil }
}

enum DownloadState: Equatable {
    case downloading(written: Int64, total: Int64)
    case paused(written: Int64, total: Int64)
    case failed(String)

    var fraction: Double {
        switch self {
        case .downloading(let w, let t), .paused(let w, let t): return t > 0 ? Double(w) / Double(t) : 0
        case .failed: return 0
        }
    }
}

/// Tracks GGUF files on disk (Application Support/Models + Documents for Finder/Files drops) and downloads.
@MainActor @Observable
final class ModelManager {
    weak var app: AppModel?
    let modelsDir: URL
    let documentsDir: URL
    private(set) var installed: [InstalledModel] = []
    private(set) var importedFiles: [URL] = []      // GGUFs not matching the catalog (incl. mmproj files)
    private(set) var downloads: [String: DownloadState] = [:]
    var lastError: String?

    private let downloader = DownloadManager()
    private var progressByKey: [String: (Int64, Int64)] = [:]
    /// Imported model file name → mmproj file name (user pairing).
    private var pairings: [String: String] {
        get { UserDefaults.standard.dictionary(forKey: "EdgeChat.mmprojPairings") as? [String: String] ?? [:] }
        set { UserDefaults.standard.set(newValue, forKey: "EdgeChat.mmprojPairings") }
    }

    init() {
        let support = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        modelsDir = support.appendingPathComponent("Models", isDirectory: true)
        documentsDir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        try? FileManager.default.createDirectory(at: modelsDir, withIntermediateDirectories: true)
        var values = URLResourceValues(); values.isExcludedFromBackup = true
        var dir = modelsDir; try? dir.setResourceValues(values)

        downloader.onProgress = { [weak self] key, written, total in
            Task { @MainActor in self?.progress(key: key, written: written, total: total) }
        }
        downloader.onFinished = { [weak self] key, result in
            Task { @MainActor in self?.finished(key: key, result: result) }
        }
        downloader.onPaused = { [weak self] key in
            Task { @MainActor in self?.paused(key: key) }
        }
        downloader.reattach { [weak self] keys in
            Task { @MainActor in
                for key in keys { self?.progress(key: key, written: 0, total: 0) }
            }
        }
    }

    // MARK: Inventory

    func refresh() {
        var files: [URL] = []
        for dir in [modelsDir, documentsDir] {
            let items = (try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: [.fileSizeKey])) ?? []
            files += items.filter { $0.pathExtension.lowercased() == "gguf" }
        }
        var result: [InstalledModel] = []
        var matched: Set<URL> = []
        for spec in ModelCatalog.models {
            guard let model = files.first(where: { $0.lastPathComponent == spec.fileName }) else { continue }
            matched.insert(model)
            var mmproj: URL?
            if let name = spec.mmprojFileName, let m = files.first(where: { $0.lastPathComponent == name }) {
                mmproj = m; matched.insert(m)
            }
            result.append(InstalledModel(id: spec.id, name: spec.name, modelURL: model, mmprojURL: mmproj, spec: spec, sizeBytes: size(of: model) + (mmproj.map(size(of:)) ?? 0)))
        }
        let others = files.filter { !matched.contains($0) }.sorted { $0.lastPathComponent < $1.lastPathComponent }
        importedFiles = others
        for file in others where !file.lastPathComponent.lowercased().hasPrefix("mmproj") {
            let name = file.deletingPathExtension().lastPathComponent
            let mmproj = pairings[file.lastPathComponent].flatMap { p in others.first { $0.lastPathComponent == p } }
            result.append(InstalledModel(id: "file:" + file.lastPathComponent, name: name, modelURL: file, mmprojURL: mmproj, spec: nil, sizeBytes: size(of: file) + (mmproj.map(size(of:)) ?? 0)))
        }
        installed = result
    }

    var importedMmprojFiles: [URL] { importedFiles.filter { $0.lastPathComponent.lowercased().hasPrefix("mmproj") } }

    func pair(model: InstalledModel, mmproj: URL?) {
        var p = pairings
        p[model.modelURL.lastPathComponent] = mmproj?.lastPathComponent
        pairings = p
        refresh()
    }

    func installed(for spec: ModelSpec) -> InstalledModel? { installed.first { $0.id == spec.id } }

    func activeInstalledModel() -> InstalledModel? {
        guard let id = app?.settings.activeModelID else { return nil }
        return installed.first { $0.id == id }
    }

    private func size(of url: URL) -> Int64 {
        Int64((try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0)
    }

    var storageUsed: Int64 { installed.reduce(0) { $0 + $1.sizeBytes } }

    var freeDiskSpace: Int64 {
        (try? modelsDir.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey]).volumeAvailableCapacityForImportantUsage) ?? 0
    }

    // MARK: Downloads

    func download(_ spec: ModelSpec) {
        lastError = nil
        downloads[spec.id] = .downloading(written: 0, total: spec.totalBytes)
        progressByKey[spec.id + "|model"] = (0, spec.sizeBytes)
        downloader.start(key: spec.id + "|model", url: spec.url, destination: modelsDir.appendingPathComponent(spec.fileName))
        if let mm = spec.mmprojURL, let name = spec.mmprojFileName {
            progressByKey[spec.id + "|mmproj"] = (0, spec.mmprojSizeBytes)
            downloader.start(key: spec.id + "|mmproj", url: mm, destination: modelsDir.appendingPathComponent(name))
        }
    }

    func pause(_ spec: ModelSpec) {
        downloader.pause(key: spec.id + "|model")
        downloader.pause(key: spec.id + "|mmproj")
    }

    func resume(_ spec: ModelSpec) { download(spec) }

    func cancel(_ spec: ModelSpec) {
        downloader.cancel(key: spec.id + "|model")
        downloader.cancel(key: spec.id + "|mmproj")
        downloads[spec.id] = nil
        progressByKey = progressByKey.filter { !$0.key.hasPrefix(spec.id + "|") }
    }

    func delete(_ model: InstalledModel) {
        try? FileManager.default.removeItem(at: model.modelURL)
        if let m = model.mmprojURL { try? FileManager.default.removeItem(at: m) }
        if app?.settings.activeModelID == model.id { app?.settings.activeModelID = nil }
        refresh()
    }

    private func progress(key: String, written: Int64, total: Int64) {
        let id = String(key.split(separator: "|")[0])
        var entry = progressByKey[key] ?? (0, 0)
        entry.0 = written
        if total > 0 { entry.1 = total }
        progressByKey[key] = entry
        let all = progressByKey.filter { $0.key.hasPrefix(id + "|") }.values
        let w = all.reduce(0) { $0 + $1.0 }, t = all.reduce(0) { $0 + $1.1 }
        if case .paused = downloads[id] { return }
        downloads[id] = .downloading(written: w, total: t)
    }

    private func paused(key: String) {
        let id = String(key.split(separator: "|")[0])
        if case .downloading(let w, let t) = downloads[id] { downloads[id] = .paused(written: w, total: t) }
    }

    private func finished(key: String, result: Result<URL, Error>) {
        let id = String(key.split(separator: "|")[0])
        switch result {
        case .success:
            progressByKey[key] = (progressByKey[key]?.1 ?? 0, progressByKey[key]?.1 ?? 0)
            let stillRunning = progressByKey.keys.filter { $0.hasPrefix(id + "|") }.contains { downloader.isActive(key: $0) }
            if !stillRunning {
                downloads[id] = nil
                progressByKey = progressByKey.filter { !$0.key.hasPrefix(id + "|") }
                refresh()
                if app?.settings.activeModelID == nil, let m = installed.first(where: { $0.id == id }) {
                    Task { await app?.engine.load(m) }
                }
            }
        case .failure(let error):
            downloads[id] = .failed(error.localizedDescription)
            lastError = error.localizedDescription
        }
    }

    // MARK: Import

    /// Copies a GGUF picked from Files (or opened via "Open in EdgeChat") into the models folder.
    func importModel(from url: URL) {
        guard url.pathExtension.lowercased() == "gguf" else { lastError = "Only .gguf files can be imported."; return }
        let scoped = url.startAccessingSecurityScopedResource()
        defer { if scoped { url.stopAccessingSecurityScopedResource() } }
        let dest = modelsDir.appendingPathComponent(url.lastPathComponent)
        do {
            if FileManager.default.fileExists(atPath: dest.path) { try FileManager.default.removeItem(at: dest) }
            try FileManager.default.copyItem(at: url, to: dest)
            var values = URLResourceValues(); values.isExcludedFromBackup = true
            var d = dest; try? d.setResourceValues(values)
            refresh()
        } catch {
            lastError = "Import failed: \(error.localizedDescription)"
        }
    }
}
