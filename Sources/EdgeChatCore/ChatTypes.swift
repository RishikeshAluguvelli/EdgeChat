import Foundation

// MARK: - Persistent chat model

public enum Role: String, Codable, Sendable, Hashable {
    case system, user, assistant
}

public struct Attachment: Codable, Hashable, Sendable, Identifiable {
    public enum Kind: String, Codable, Sendable { case image, document }

    public var id: UUID
    public var kind: Kind
    /// Original display name (e.g. "report.pdf", "IMG_0421.jpeg").
    public var fileName: String
    /// File stored inside the conversation's attachment folder.
    public var storedFileName: String
    public var byteCount: Int64
    /// Text extracted from a document, or OCR/labels for an image (used for text-only models).
    public var extractedText: String?
    public var textTruncated: Bool
    public var pageCount: Int?
    public var imageWidth: Int?
    public var imageHeight: Int?
    /// How the attachment was fed to the model on send ("vision", "ocr", "text").
    public var ingestion: String?

    public init(id: UUID = UUID(), kind: Kind, fileName: String, storedFileName: String, byteCount: Int64,
                extractedText: String? = nil, textTruncated: Bool = false, pageCount: Int? = nil,
                imageWidth: Int? = nil, imageHeight: Int? = nil, ingestion: String? = nil) {
        self.id = id; self.kind = kind; self.fileName = fileName; self.storedFileName = storedFileName
        self.byteCount = byteCount; self.extractedText = extractedText; self.textTruncated = textTruncated
        self.pageCount = pageCount; self.imageWidth = imageWidth; self.imageHeight = imageHeight
        self.ingestion = ingestion
    }
}

public struct GenerationStats: Codable, Hashable, Sendable {
    /// Total prompt tokens (including image tokens).
    public var promptTokens: Int
    /// Prompt positions that were reused from the KV cache (no recompute).
    public var cachedTokens: Int
    public var generatedTokens: Int
    public var prefillSeconds: Double
    public var decodeSeconds: Double
    public var stopReason: StopReason
    public var modelID: String?
    /// Number of mid-generation context shifts (oldest tokens discarded to keep going).
    public var contextShifts: Int?
    /// Context window the reply was generated with.
    public var contextLength: Int?

    public var tokensPerSecond: Double {
        decodeSeconds > 0 ? Double(generatedTokens) / decodeSeconds : 0
    }
    public var promptTokensPerSecond: Double {
        let fresh = promptTokens - cachedTokens
        return prefillSeconds > 0 && fresh > 0 ? Double(fresh) / prefillSeconds : 0
    }

    public init(promptTokens: Int, cachedTokens: Int, generatedTokens: Int, prefillSeconds: Double,
                decodeSeconds: Double, stopReason: StopReason, modelID: String? = nil, contextShifts: Int? = nil) {
        self.promptTokens = promptTokens; self.cachedTokens = cachedTokens; self.generatedTokens = generatedTokens
        self.prefillSeconds = prefillSeconds; self.decodeSeconds = decodeSeconds; self.stopReason = stopReason
        self.modelID = modelID; self.contextShifts = contextShifts
    }
}

public enum StopReason: String, Codable, Sendable, Hashable {
    case eos, maxTokens, contextFull, cancelled
}

public struct Message: Codable, Hashable, Sendable, Identifiable {
    public var id: UUID
    public var role: Role
    public var content: String
    /// Chain-of-thought emitted inside <think>…</think> (Qwen3 etc.), shown collapsed.
    public var reasoning: String?
    public var attachments: [Attachment]
    public var createdAt: Date
    public var stats: GenerationStats?
    public var error: String?
    /// Snippets retrieved from earlier in the conversation and shown to the model with this user turn.
    public var recalledContext: String?

    public init(id: UUID = UUID(), role: Role, content: String, reasoning: String? = nil,
                attachments: [Attachment] = [], createdAt: Date = Date(),
                stats: GenerationStats? = nil, error: String? = nil, recalledContext: String? = nil) {
        self.id = id; self.role = role; self.content = content; self.reasoning = reasoning
        self.attachments = attachments; self.createdAt = createdAt; self.stats = stats; self.error = error
        self.recalledContext = recalledContext
    }
}

public struct Conversation: Codable, Hashable, Sendable, Identifiable {
    public var id: UUID
    public var title: String
    public var messages: [Message]
    public var createdAt: Date
    public var updatedAt: Date
    /// Per-conversation override; nil uses the global default.
    public var systemPrompt: String?
    public var modelID: String?
    /// Model-written memory of messages that no longer fit the context window.
    public var summary: String?
    /// Number of leading messages folded into `summary` (they are not sent to the model any more).
    public var summaryCoversMessages: Int

    public init(id: UUID = UUID(), title: String = "New chat", messages: [Message] = [],
                createdAt: Date = Date(), updatedAt: Date = Date(), systemPrompt: String? = nil, modelID: String? = nil,
                summary: String? = nil, summaryCoversMessages: Int = 0) {
        self.id = id; self.title = title; self.messages = messages; self.createdAt = createdAt
        self.updatedAt = updatedAt; self.systemPrompt = systemPrompt; self.modelID = modelID
        self.summary = summary; self.summaryCoversMessages = summaryCoversMessages
    }

    private enum CodingKeys: String, CodingKey { case id, title, messages, createdAt, updatedAt, systemPrompt, modelID, summary, summaryCoversMessages }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decode(UUID.self, forKey: .id)
        title = try c.decode(String.self, forKey: .title)
        messages = try c.decode([Message].self, forKey: .messages)
        createdAt = try c.decode(Date.self, forKey: .createdAt)
        updatedAt = try c.decode(Date.self, forKey: .updatedAt)
        systemPrompt = try c.decodeIfPresent(String.self, forKey: .systemPrompt)
        modelID = try c.decodeIfPresent(String.self, forKey: .modelID)
        summary = try c.decodeIfPresent(String.self, forKey: .summary)
        summaryCoversMessages = try c.decodeIfPresent(Int.self, forKey: .summaryCoversMessages) ?? 0
    }
}

// MARK: - Engine configuration

public struct SamplingConfig: Codable, Hashable, Sendable {
    public var temperature: Float = 0.7
    public var topK: Int32 = 40
    public var topP: Float = 0.9
    public var minP: Float = 0.05
    public var repeatPenalty: Float = 1.1
    public var repeatLastN: Int32 = 64
    /// Maximum tokens to generate per reply; 0 = no limit (the reply ends when the model stops or the window is full).
    /// Also bounds the space reserved for the reply in the context budget (see `LlamaEngine.replyReserve`).
    public var maxTokens: Int = 0
    /// Effective per-reply cap used by the generation loop.
    public var effectiveMaxTokens: Int { maxTokens > 0 ? maxTokens : Int.max }
    /// 0xFFFFFFFF = random seed each call.
    public var seed: UInt32 = 0xFFFF_FFFF

    public init() {}
}

public struct EngineConfig: Codable, Hashable, Sendable {
    public var contextLength: UInt32 = 4096
    public var batchSize: UInt32 = 512
    /// Layers offloaded to the GPU (Metal). 99 = all. Ignored (0) on the Simulator.
    public var gpuLayers: Int32 = 99
    /// 0 = auto (physical cores minus two, at least two).
    public var threads: Int32 = 0
    public var flashAttention: Bool = true
    /// Quantize the KV cache to Q8_0 (halves cache memory, enables longer contexts).
    public var kvCacheQ8: Bool = false
    /// For hybrid reasoning models (Qwen3): suppress <think> blocks for faster replies.
    public var disableThinking: Bool = true
    /// When a reply would overflow the context, discard the oldest half of the history and keep generating.
    public var contextShift: Bool = true

    public init() {}

    private enum CodingKeys: String, CodingKey { case contextLength, batchSize, gpuLayers, threads, flashAttention, kvCacheQ8, disableThinking, contextShift }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let d = EngineConfig()
        contextLength = try c.decodeIfPresent(UInt32.self, forKey: .contextLength) ?? d.contextLength
        batchSize = try c.decodeIfPresent(UInt32.self, forKey: .batchSize) ?? d.batchSize
        gpuLayers = try c.decodeIfPresent(Int32.self, forKey: .gpuLayers) ?? d.gpuLayers
        threads = try c.decodeIfPresent(Int32.self, forKey: .threads) ?? d.threads
        flashAttention = try c.decodeIfPresent(Bool.self, forKey: .flashAttention) ?? d.flashAttention
        kvCacheQ8 = try c.decodeIfPresent(Bool.self, forKey: .kvCacheQ8) ?? d.kvCacheQ8
        disableThinking = try c.decodeIfPresent(Bool.self, forKey: .disableThinking) ?? d.disableThinking
        contextShift = try c.decodeIfPresent(Bool.self, forKey: .contextShift) ?? d.contextShift
    }

    /// Sensible context length for a device: 8k on 8 GB phones, 4k on 6 GB, 2k below.
    public static func defaultContextLength(physicalMemory: UInt64) -> UInt32 {
        let gb = Double(physicalMemory) / 1_073_741_824
        if gb >= 7.5 { return 8192 }
        if gb >= 5.5 { return 4096 }
        return 2048
    }

    public var resolvedThreads: Int32 {
        if threads > 0 { return threads }
        let cores = Int32(ProcessInfo.processInfo.activeProcessorCount)
        return max(2, cores - 2)
    }
}

// MARK: - Engine I/O

/// An RGB888 bitmap ready for the vision encoder.
public struct ImageInput: Sendable, Hashable {
    /// Stable identifier (used for KV-cache prefix reuse across turns).
    public let id: String
    public let width: Int
    public let height: Int
    public let rgb: Data

    public init(id: String, width: Int, height: Int, rgb: Data) {
        precondition(rgb.count == width * height * 3, "rgb must be width*height*3 bytes")
        self.id = id; self.width = width; self.height = height; self.rgb = rgb
    }
}

/// One turn as fed to the engine. `text` already contains any document text the app injected.
public struct ChatTurn: Sendable, Hashable {
    public var role: Role
    public var text: String
    public var images: [ImageInput]

    public init(role: Role, text: String, images: [ImageInput] = []) {
        self.role = role; self.text = text; self.images = images
    }
}

public enum GenerationEvent: Sendable {
    /// Prompt has been evaluated; generation starts.
    case promptProcessed(promptTokens: Int, cachedTokens: Int, droppedTurns: Int, seconds: Double)
    case token(String)
    case finished(GenerationStats)
}

public enum EngineError: Error, LocalizedError, Sendable {
    case notLoaded
    case modelLoadFailed(String)
    case contextCreateFailed
    case mmprojLoadFailed(String)
    case templateFailed
    case tokenizeFailed
    case decodeFailed(Int32)
    case contextFull
    case contextTooSmall(Int)
    case promptTooLong
    case imageEncodeFailed(Int32)
    case busy
    case cancelled
    case stateFailed
    case summaryFailed

    public var errorDescription: String? {
        switch self {
        case .notLoaded: return "No model is loaded."
        case .modelLoadFailed(let p): return "Could not load model at \(p). The file may be corrupt or unsupported."
        case .contextCreateFailed: return "Could not create the inference context (out of memory?). Try a smaller context length."
        case .mmprojLoadFailed(let p): return "Could not load the vision projector at \(p)."
        case .templateFailed: return "The model's chat template could not be applied."
        case .tokenizeFailed: return "Tokenization failed."
        case .decodeFailed(let rc): return "Inference failed (llama_decode returned \(rc))."
        case .contextFull: return "The context window is full."
        case .contextTooSmall(let n): return "Context length \(n) is too small for the reply budget. Increase context length or lower max tokens."
        case .promptTooLong: return "The message is too long to fit in the context window."
        case .imageEncodeFailed(let rc): return "The vision encoder failed (code \(rc))."
        case .busy: return "The engine is busy."
        case .cancelled: return "Generation was cancelled."
        case .stateFailed: return "Could not save the conversation cache."
        case .summaryFailed: return "The model produced an empty summary."
        }
    }
}
