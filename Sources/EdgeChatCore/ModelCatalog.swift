import Foundation

public enum ModelCapability: String, Codable, Sendable, Hashable {
    case vision      // accepts images via an mmproj vision encoder
    case reasoning   // hybrid thinking model (<think> blocks)
}

/// A downloadable GGUF model (plus optional vision projector).
public struct ModelSpec: Identifiable, Codable, Hashable, Sendable {
    public let id: String
    public let name: String
    public let family: String
    public let parameters: String
    public let quantization: String
    public let url: URL
    public let sizeBytes: Int64
    public let mmprojURL: URL?
    public let mmprojSizeBytes: Int64
    public let capabilities: Set<ModelCapability>
    public let contextTrain: Int
    public let summary: String
    public let tier: Tier

    public enum Tier: String, Codable, Sendable, Hashable, CaseIterable {
        case flagship = "Best for 8 GB iPhones"
        case balanced = "Good on 6 GB iPhones"
        case tiny = "Tiny (testing / older phones)"
    }

    public var fileName: String { url.lastPathComponent }
    public var mmprojFileName: String? { mmprojURL.map { "\(id)-\($0.lastPathComponent)" } }
    public var totalBytes: Int64 { sizeBytes + mmprojSizeBytes }
    public var hasVision: Bool { capabilities.contains(.vision) }

    /// Rough working-set estimate: weights + projector + KV cache/activations headroom.
    public func estimatedRAMBytes(contextLength: Int) -> Int64 {
        let kv = Int64(contextLength) * 64 * 1024 // ~64 KB per token is typical for 3-4B models at f16
        return Int64(Double(totalBytes) * 1.1) + kv + 300 * 1024 * 1024
    }

    public enum Fit: Sendable { case good, tight, tooLarge }

    public func fit(physicalMemory: UInt64, contextLength: Int) -> Fit {
        let need = Double(estimatedRAMBytes(contextLength: contextLength))
        let ram = Double(physicalMemory)
        if need < ram * 0.5 { return .good }
        if need < ram * 0.68 { return .tight }
        return .tooLarge
    }
}

public enum ModelCatalog {
    private static func hf(_ repo: String, _ file: String) -> URL {
        URL(string: "https://huggingface.co/\(repo)/resolve/main/\(file)")!
    }
    private static let GB: Double = 1_073_741_824

    /// Curated, verified GGUFs. Ordered by recommendation within each tier.
    public static let models: [ModelSpec] = [
        ModelSpec(
            id: "qwen3-vl-4b-instruct",
            name: "Qwen3-VL 4B Instruct", family: "Qwen3-VL", parameters: "4B", quantization: "Q4_K_M",
            url: hf("unsloth/Qwen3-VL-4B-Instruct-GGUF", "Qwen3-VL-4B-Instruct-Q4_K_M.gguf"),
            sizeBytes: Int64(2.33 * GB),
            mmprojURL: hf("unsloth/Qwen3-VL-4B-Instruct-GGUF", "mmproj-F16.gguf"),
            mmprojSizeBytes: Int64(0.78 * GB),
            capabilities: [.vision], contextTrain: 262_144,
            summary: "Recommended. Best all-rounder that fits an iPhone: strong chat, coding and math plus genuine image understanding (photos, screenshots, charts, documents).",
            tier: .flagship),
        ModelSpec(
            id: "qwen3-4b-instruct-2507",
            name: "Qwen3 4B Instruct (2507)", family: "Qwen3", parameters: "4B", quantization: "Q4_K_M",
            url: hf("unsloth/Qwen3-4B-Instruct-2507-GGUF", "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"),
            sizeBytes: Int64(2.33 * GB), mmprojURL: nil, mmprojSizeBytes: 0,
            capabilities: [], contextTrain: 262_144,
            summary: "Strongest text-only 4B model. Images are handled with on-device OCR instead of a vision encoder.",
            tier: .flagship),
        ModelSpec(
            id: "gemma-3-4b-it",
            name: "Gemma 3 4B IT", family: "Gemma 3", parameters: "4B", quantization: "Q4_K_M",
            url: hf("unsloth/gemma-3-4b-it-GGUF", "gemma-3-4b-it-Q4_K_M.gguf"),
            sizeBytes: Int64(2.32 * GB),
            mmprojURL: hf("unsloth/gemma-3-4b-it-GGUF", "mmproj-F16.gguf"),
            mmprojSizeBytes: Int64(0.79 * GB),
            capabilities: [.vision], contextTrain: 131_072,
            summary: "Google's multimodal 4B model. Friendly writing style, good multilingual support, image input.",
            tier: .flagship),
        ModelSpec(
            id: "qwen3-vl-4b-instruct-q3",
            name: "Qwen3-VL 4B Instruct (compact)", family: "Qwen3-VL", parameters: "4B", quantization: "Q3_K_M",
            url: hf("unsloth/Qwen3-VL-4B-Instruct-GGUF", "Qwen3-VL-4B-Instruct-Q3_K_M.gguf"),
            sizeBytes: Int64(1.93 * GB),
            mmprojURL: hf("unsloth/Qwen3-VL-4B-Instruct-GGUF", "mmproj-F16.gguf"),
            mmprojSizeBytes: Int64(0.78 * GB),
            capabilities: [.vision], contextTrain: 262_144,
            summary: "The 4B vision model squeezed to 3-bit for 6 GB phones (iPhone 13–15, 15 Plus). Slightly less accurate than Q4 but far more capable than the 2B. Use a 4k context and the 8-bit KV cache.",
            tier: .balanced),
        ModelSpec(
            id: "llama-3.2-3b-instruct",
            name: "Llama 3.2 3B Instruct", family: "Llama 3.2", parameters: "3B", quantization: "Q4_K_M",
            url: hf("unsloth/Llama-3.2-3B-Instruct-GGUF", "Llama-3.2-3B-Instruct-Q4_K_M.gguf"),
            sizeBytes: Int64(1.88 * GB), mmprojURL: nil, mmprojSizeBytes: 0,
            capabilities: [], contextTrain: 131_072,
            summary: "Meta's compact instruct model. Fast and reliable for everyday chat and summarization.",
            tier: .balanced),
        ModelSpec(
            id: "qwen3-vl-2b-instruct",
            name: "Qwen3-VL 2B Instruct", family: "Qwen3-VL", parameters: "2B", quantization: "Q4_K_M",
            url: hf("unsloth/Qwen3-VL-2B-Instruct-GGUF", "Qwen3-VL-2B-Instruct-Q4_K_M.gguf"),
            sizeBytes: Int64(1.03 * GB),
            mmprojURL: hf("unsloth/Qwen3-VL-2B-Instruct-GGUF", "mmproj-F16.gguf"),
            mmprojSizeBytes: Int64(0.76 * GB),
            capabilities: [.vision], contextTrain: 262_144,
            summary: "Smallest model with real image understanding. Good choice for 6 GB phones that need vision.",
            tier: .balanced),
        ModelSpec(
            id: "qwen3-1.7b",
            name: "Qwen3 1.7B", family: "Qwen3", parameters: "1.7B", quantization: "Q4_K_M",
            url: hf("unsloth/Qwen3-1.7B-GGUF", "Qwen3-1.7B-Q4_K_M.gguf"),
            sizeBytes: Int64(1.03 * GB), mmprojURL: nil, mmprojSizeBytes: 0,
            capabilities: [.reasoning], contextTrain: 40_960,
            summary: "Very fast hybrid reasoning model. Thinking can be toggled in Settings.",
            tier: .balanced),
        ModelSpec(
            id: "qwen3-0.6b",
            name: "Qwen3 0.6B", family: "Qwen3", parameters: "0.6B", quantization: "Q4_K_M",
            url: hf("unsloth/Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q4_K_M.gguf"),
            sizeBytes: Int64(0.37 * GB), mmprojURL: nil, mmprojSizeBytes: 0,
            capabilities: [.reasoning], contextTrain: 40_960,
            summary: "Tiny and instant. Limited knowledge; handy for testing and the Simulator.",
            tier: .tiny),
        ModelSpec(
            id: "smolvlm-256m-instruct",
            name: "SmolVLM 256M Instruct", family: "SmolVLM", parameters: "256M", quantization: "Q8_0",
            url: hf("ggml-org/SmolVLM-256M-Instruct-GGUF", "SmolVLM-256M-Instruct-Q8_0.gguf"),
            sizeBytes: Int64(0.16 * GB),
            mmprojURL: hf("ggml-org/SmolVLM-256M-Instruct-GGUF", "mmproj-SmolVLM-256M-Instruct-Q8_0.gguf"),
            mmprojSizeBytes: Int64(0.10 * GB),
            capabilities: [.vision], contextTrain: 8_192,
            summary: "Tiny vision model for testing image input end to end. Very limited quality.",
            tier: .tiny),
    ]

    public static let recommendedID = "qwen3-vl-4b-instruct"

    public static func spec(id: String) -> ModelSpec? { models.first { $0.id == id } }
}
