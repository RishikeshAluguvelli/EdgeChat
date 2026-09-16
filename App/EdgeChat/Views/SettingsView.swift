import SwiftUI
import EdgeChatCore

struct SettingsView: View {
    @Environment(AppModel.self) private var app
    @Environment(\.dismiss) private var dismiss

    private let contextChoices: [UInt32] = [2048, 4096, 8192, 16384, 32768]
    private let maxTokenChoices = [0, 256, 512, 1024, 2048, 4096]   // 0 = until the model stops

    var body: some View {
        @Bindable var app = app
        NavigationStack {
            Form {
                Section {
                    Picker("Context length", selection: $app.settings.engine.contextLength) {
                        ForEach(contextChoices, id: \.self) { Text(TextUtils.formatContext(Int($0))).tag($0) }
                    }
                    Toggle("Use GPU (Metal)", isOn: Binding(get: { app.settings.engine.gpuLayers > 0 }, set: { app.settings.engine.gpuLayers = $0 ? 99 : 0 }))
                    Toggle("Flash attention", isOn: $app.settings.engine.flashAttention)
                    Toggle("8-bit KV cache", isOn: $app.settings.engine.kvCacheQ8)
                    Stepper("Threads: \(app.settings.engine.threads == 0 ? "auto (\(app.settings.engine.resolvedThreads))" : "\(app.settings.engine.threads)")",
                            value: $app.settings.engine.threads, in: 0...Int32(ProcessInfo.processInfo.activeProcessorCount))
                    Toggle("Skip thinking (Qwen3 hybrids)", isOn: $app.settings.engine.disableThinking)
                    if let info = app.engine.info {
                        LabeledContent("KV cache at \(TextUtils.formatContext(Int(app.settings.engine.contextLength)))", value: TextUtils.formatBytes(Int64(info.kvCacheBytes(contextLength: Int(app.settings.engine.contextLength)))))
                    }
                    if app.engine.loadedModel != nil, app.engine.loadedConfig != app.settings.engine {
                        Button("Apply and reload model") { Task { await app.engine.applyEngineSettingsIfNeeded() } }
                    }
                } header: {
                    Text("Inference")
                } footer: {
                    Text("Longer contexts let you attach bigger documents but use more memory (roughly 64 KB per token for a 4B model; 8-bit KV cache halves that). Changes apply when the model is reloaded.")
                }

                Section {
                    Toggle("Summarize dropped history", isOn: $app.settings.summarizeDroppedTurns)
                    Toggle("Compact before the window fills", isOn: $app.settings.autoCompact).disabled(!app.settings.summarizeDroppedTurns)
                    Toggle("Recall older context (RAG)", isOn: $app.settings.memoryRetrieval)
                    Toggle("Shift context mid-reply", isOn: $app.settings.engine.contextShift)
                    Toggle("Cache snapshots per chat", isOn: $app.settings.kvSnapshots)
                } header: {
                    Text("Long conversations")
                } footer: {
                    Text("When a chat outgrows the context window, the model writes a running summary of the oldest messages, and relevant older messages or document passages are retrieved (on-device embeddings + keyword search) into each new turn. Compacting early runs that summary right after a reply once the window is about three-quarters full, so the next message is not delayed. Snapshots keep switching between chats instant at the cost of some storage.")
                }

                Section {
                    sliderRow("Temperature", value: $app.settings.sampling.temperature, range: 0...1.5, step: 0.05)
                    sliderRow("Top-p", value: $app.settings.sampling.topP, range: 0.1...1, step: 0.05)
                    sliderRow("Min-p", value: $app.settings.sampling.minP, range: 0...0.3, step: 0.01)
                    sliderRow("Repeat penalty", value: $app.settings.sampling.repeatPenalty, range: 1...1.5, step: 0.01)
                    Picker("Reply length", selection: $app.settings.sampling.maxTokens) {
                        ForEach(maxTokenChoices, id: \.self) { Text($0 == 0 ? "Until the model stops" : "\($0) tokens").tag($0) }
                    }
                } header: {
                    Text("Generation")
                } footer: {
                    Text("With no limit, a reply ends when the model finishes. If it reaches the edge of the context window, the oldest messages are summarized away and the reply continues automatically; you can also tap Continue under a reply that stopped early.")
                }

                Section("System prompt") {
                    TextEditor(text: $app.settings.systemPrompt).frame(minHeight: 110).font(.footnote)
                    Button("Reset to default") { app.settings.systemPrompt = AppSettings().systemPrompt }.font(.footnote)
                }

                Section {
                    Toggle("OCR images on attach", isOn: $app.settings.ocrForImages)
                    Picker("Max image size", selection: $app.settings.maxImageDimension) {
                        ForEach([512, 768, 1024, 1536], id: \.self) { Text("\($0) px").tag($0) }
                    }
                } header: {
                    Text("Attachments")
                } footer: {
                    Text("OCR lets text-only models read screenshots and scanned pages. Vision models see the image directly; larger images cost more prompt tokens.")
                }

                Section("Display") {
                    Toggle("Show generation stats", isOn: $app.settings.showStats)
                    Toggle("Load last model on launch", isOn: $app.settings.autoLoadLastModel)
                }

                Section("About") {
                    LabeledContent("Engine", value: "llama.cpp b10988 (Metal, mtmd)")
                    ShareLink(item: DiagnosticsLog.url) { Label("Share diagnostics log", systemImage: "doc.text.magnifyingglass") }
                    LabeledContent("Version", value: Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "-")
                    if let info = app.engine.info {
                        LabeledContent("Loaded model", value: info.name)
                        LabeledContent("Architecture", value: info.architectureDescription)
                        LabeledContent("Parameters", value: String(format: "%.2fB", Double(info.parameterCount) / 1e9))
                        LabeledContent("Training context", value: "\(info.trainingContext)")
                    }
                }
            }
            .navigationTitle("Settings")
            .toolbar { ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } } }
        }
    }

    private func sliderRow(_ title: String, value: Binding<Float>, range: ClosedRange<Float>, step: Float) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack { Text(title); Spacer(); Text(String(format: "%.2f", value.wrappedValue)).foregroundStyle(.secondary).monospacedDigit() }
            Slider(value: value, in: range, step: step)
        }
    }
}
