import Foundation
import EdgeChatCore

// edgechat-cli: exercise the same engine the iOS app uses, from the terminal (macOS, Metal).
//
//   swift run edgechat-cli --model path/to/model.gguf [--mmproj path/to/mmproj.gguf]
//        [--image photo.jpg] [--doc report.pdf] [--ctx 4096] [--system "..."]
//        [--prompt "first turn" --prompt "second turn" ...]   # scripted turns; omit for a REPL
//        [--think]                                             # allow <think> on Qwen3 hybrids

var modelPath: String?
var mmprojPath: String?
var images: [String] = []
var docs: [String] = []
var prompts: [String] = []
var system = "You are a helpful assistant running fully on-device."
var ctxLen: UInt32 = 4096
var think = false
var maxTokens = 512
var verbose = false

var args = Array(CommandLine.arguments.dropFirst())
func next(_ flag: String) -> String {
    guard !args.isEmpty else { fputs("missing value for \(flag)\n", stderr); exit(2) }
    return args.removeFirst()
}
while !args.isEmpty {
    let a = args.removeFirst()
    switch a {
    case "--model": modelPath = next(a)
    case "--mmproj": mmprojPath = next(a)
    case "--image": images.append(next(a))
    case "--doc": docs.append(next(a))
    case "--prompt": prompts.append(next(a))
    case "--system": system = next(a)
    case "--ctx": ctxLen = UInt32(next(a)) ?? 4096
    case "--max-tokens": maxTokens = Int(next(a)) ?? 512
    case "--think": think = true
    case "--verbose": verbose = true
    default: fputs("unknown argument \(a)\n", stderr); exit(2)
    }
}
guard let modelPath else {
    fputs("usage: edgechat-cli --model <gguf> [--mmproj <gguf>] [--image <file>] [--doc <file>] [--prompt <text> ...]\n", stderr)
    exit(2)
}

if !verbose { LlamaEngine.logHandler = { level, text in if level.rawValue >= LlamaLogLevel.error.rawValue { fputs(text, stderr) } } }

let engine = LlamaEngine()
var config = EngineConfig()
config.contextLength = ctxLen
config.disableThinking = !think

let semaphore = DispatchSemaphore(value: 0)
Task {
    do {
        let t0 = Date()
        let info = try await engine.load(modelPath: modelPath, mmprojPath: mmprojPath, config: config) { p in
            if verbose { fputs("\rloading \(Int(p * 100))%", stderr) }
        }
        fputs("Loaded \(info.name) (\(info.architectureDescription)) in \(String(format: "%.1f", Date().timeIntervalSince(t0)))s — ctx \(info.contextLength), vision: \(info.hasVision), gpu: \(info.gpuOffloaded), threads: \(info.threads)\n", stderr)

        var turns: [ChatTurn] = []
        var pendingImages: [ImageInput] = []
        var pendingDocText = ""
        for path in images {
            let data = try Data(contentsOf: URL(fileURLWithPath: path))
            let (input, cg) = try AttachmentProcessor.makeImageInput(from: data, id: path)
            if info.hasVision {
                pendingImages.append(input)
                fputs("attached image \(path) (\(input.width)x\(input.height))\n", stderr)
            } else {
                let desc = await AttachmentProcessor.describeImageAsText(cg, fileName: (path as NSString).lastPathComponent)
                pendingDocText += desc + "\n\n"
                fputs("model has no vision; used OCR/labels for \(path)\n", stderr)
            }
        }
        for path in docs {
            let ex = try await AttachmentProcessor.extractText(from: URL(fileURLWithPath: path), maxCharacters: Int(ctxLen) * 3)
            pendingDocText += "<document name=\"\((path as NSString).lastPathComponent)\">\n\(ex.text)\n</document>\n\n"
            fputs("attached document \(path) (\(ex.method), pages: \(ex.pageCount.map(String.init) ?? "-"), truncated: \(ex.truncated))\n", stderr)
        }

        func runTurn(_ userText: String) async throws {
            var text = userText
            if !pendingDocText.isEmpty { text = pendingDocText + text; pendingDocText = "" }
            turns.append(ChatTurn(role: .user, text: text, images: pendingImages))
            pendingImages = []
            var sampling = SamplingConfig()
            sampling.maxTokens = maxTokens
            var reply = ""
            var splitter = ThinkTagSplitter()
            var inReasoning = false
            for try await event in engine.respond(systemPrompt: system, turns: turns, sampling: sampling) {
                switch event {
                case .promptProcessed(let n, let cached, let dropped, let secs):
                    fputs("[prompt \(n) tokens, \(cached) cached, \(dropped) turns dropped, \(String(format: "%.2f", secs))s]\n", stderr)
                case .token(let s):
                    for part in splitter.feed(s) {
                        switch part {
                        case .reasoning(let r):
                            if !inReasoning { fputs("\u{1B}[2m<think>", stdout); inReasoning = true }
                            fputs(r, stdout)
                        case .text(let t):
                            if inReasoning { fputs("</think>\u{1B}[0m\n", stdout); inReasoning = false }
                            fputs(t, stdout); reply += t
                        }
                    }
                    fflush(stdout)
                case .finished(let stats):
                    for part in splitter.flush() { if case .text(let t) = part { fputs(t, stdout); reply += t } }
                    fputs("\n[\(stats.generatedTokens) tokens, \(String(format: "%.1f", stats.tokensPerSecond)) tok/s, stop: \(stats.stopReason.rawValue)]\n", stderr)
                }
            }
            turns.append(ChatTurn(role: .assistant, text: reply))
        }

        if prompts.isEmpty {
            fputs("Interactive mode. Type a message, or /quit.\n", stderr)
            while true {
                fputs("\n> ", stdout); fflush(stdout)
                guard let line = readLine(), line != "/quit" else { break }
                if line.trimmingCharacters(in: .whitespaces).isEmpty { continue }
                try await runTurn(line)
            }
        } else {
            for p in prompts {
                fputs("\n> \(p)\n", stdout)
                try await runTurn(p)
            }
        }
        await engine.unload()
        semaphore.signal()
    } catch {
        fputs("error: \(error.localizedDescription)\n", stderr)
        exit(1)
    }
}
semaphore.wait()
