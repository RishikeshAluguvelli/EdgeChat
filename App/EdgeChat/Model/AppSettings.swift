import Foundation
import EdgeChatCore

/// User-tunable settings, persisted as JSON in UserDefaults. Unknown/missing keys fall back to defaults.
struct AppSettings: Codable, Equatable {
    var systemPrompt = AppSettings.defaultSystemPrompt
    var sampling = SamplingConfig()
    var engine = EngineConfig()
    var activeModelID: String?          // catalog id or "file:<name>" for imported models
    var ocrForImages = true             // also run OCR so text-only models can read images
    var maxImageDimension = 768        // ~576 image tokens on Qwen3-VL; 1024 px costs ~1000
    var showStats = true
    var autoLoadLastModel = true
    /// Long-conversation memory.
    var summarizeDroppedTurns = true    // fold dropped history into a model-written summary
    var memoryRetrieval = true          // RAG: recall relevant older messages/documents into the current turn
    var kvSnapshots = true              // save/restore the KV cache per conversation
    var autoCompact = true              // summarize early (after a reply crosses ~75% of the window) instead of waiting for it to fill

    static let defaultSystemPrompt = """
You are a helpful assistant running entirely on the user's iPhone, offline.
Answer the current message directly. Match the length to the question: short questions get short answers, and reply to acknowledgements like "thanks" or "good to know" in one line. Do not restate or re-answer earlier messages unless asked.
If the message is unclear, ask a brief clarifying question instead of guessing what was meant.
If you are unsure or do not know something (a specific paper, product, person or recent event), say so plainly rather than inventing details.
When documents or images are attached, ground your answers in them. Continue the conversation naturally.
A "Memory of earlier conversation" section or a <recalled_context> block, when present, is background from earlier in this same chat: use it silently for continuity and never mention or repeat it.
"""

    /// Earlier defaults; a stored prompt equal to one of these is upgraded to the current default on load.
    static let legacyDefaultSystemPrompts: Set<String> = [
        "You are a helpful, concise assistant running entirely on the user's iPhone, with no internet access. When documents or images are attached, ground your answers in them. If a <recalled_context> block or a memory summary is present, treat it as earlier parts of this same conversation.",
        "You are a helpful, concise assistant running entirely on the user's iPhone, with no internet access. When documents or images are attached, ground your answers in them. If a <recalled_context> block or a memory summary is present, treat it as earlier parts of this same conversation. If you are unsure or do not know something (for example a specific paper, product, or recent event), say so plainly instead of guessing or inventing details.",
    ]
    /// Bumped when stored values need migrating (v2: reply length 1024 → unlimited).
    static let currentSettingsVersion = 2
    var settingsVersion = AppSettings.currentSettingsVersion

    static let key = "EdgeChat.settings.v1"

    init() {
        engine.contextLength = EngineConfig.defaultContextLength(physicalMemory: ProcessInfo.processInfo.physicalMemory)
    }

    private enum CodingKeys: String, CodingKey {
        case systemPrompt, sampling, engine, activeModelID, ocrForImages, maxImageDimension, showStats, autoLoadLastModel,
             summarizeDroppedTurns, memoryRetrieval, kvSnapshots, autoCompact, settingsVersion
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let d = AppSettings()
        let storedPrompt = try c.decodeIfPresent(String.self, forKey: .systemPrompt) ?? d.systemPrompt
        systemPrompt = Self.legacyDefaultSystemPrompts.contains(storedPrompt) ? d.systemPrompt : storedPrompt
        sampling = try c.decodeIfPresent(SamplingConfig.self, forKey: .sampling) ?? d.sampling
        let version = try c.decodeIfPresent(Int.self, forKey: .settingsVersion) ?? 1
        if version < 2, sampling.maxTokens == 1024 { sampling.maxTokens = 0 }   // old default → unlimited
        settingsVersion = Self.currentSettingsVersion
        engine = try c.decodeIfPresent(EngineConfig.self, forKey: .engine) ?? d.engine
        activeModelID = try c.decodeIfPresent(String.self, forKey: .activeModelID)
        ocrForImages = try c.decodeIfPresent(Bool.self, forKey: .ocrForImages) ?? d.ocrForImages
        maxImageDimension = try c.decodeIfPresent(Int.self, forKey: .maxImageDimension) ?? d.maxImageDimension
        showStats = try c.decodeIfPresent(Bool.self, forKey: .showStats) ?? d.showStats
        autoLoadLastModel = try c.decodeIfPresent(Bool.self, forKey: .autoLoadLastModel) ?? d.autoLoadLastModel
        summarizeDroppedTurns = try c.decodeIfPresent(Bool.self, forKey: .summarizeDroppedTurns) ?? d.summarizeDroppedTurns
        memoryRetrieval = try c.decodeIfPresent(Bool.self, forKey: .memoryRetrieval) ?? d.memoryRetrieval
        kvSnapshots = try c.decodeIfPresent(Bool.self, forKey: .kvSnapshots) ?? d.kvSnapshots
        autoCompact = try c.decodeIfPresent(Bool.self, forKey: .autoCompact) ?? d.autoCompact
    }

    static func load() -> AppSettings {
        guard let data = UserDefaults.standard.data(forKey: key),
              let s = try? JSONDecoder().decode(AppSettings.self, from: data) else { return AppSettings() }
        return s
    }

    func save() {
        if let data = try? JSONEncoder().encode(self) { UserDefaults.standard.set(data, forKey: Self.key) }
    }
}
