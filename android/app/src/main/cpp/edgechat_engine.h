// EdgeChat native engine: a single-sequence llama.cpp engine with KV-cache prefix reuse across turns.
// Mirrors Sources/EdgeChatCore/LlamaEngine.swift (iOS) so both apps behave the same.
#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

namespace edgechat {

struct EngineConfig {
    uint32_t contextLength = 4096;
    uint32_t batchSize = 512;
    int32_t gpuLayers = 0;          // CPU-only on Android for now (no Metal); kept for parity
    int32_t threads = 0;            // 0 = auto
    bool flashAttention = true;
    bool kvCacheQ8 = false;
    bool disableThinking = true;
    bool contextShift = true;
};

struct SamplingConfig {
    float temperature = 0.7f;
    int32_t topK = 40;
    float topP = 0.9f;
    float minP = 0.05f;
    float repeatPenalty = 1.1f;
    int32_t repeatLastN = 64;
    int32_t maxTokens = 0;          // 0 = until the model stops (ceiling max(4096, 2*n_ctx))
    uint32_t seed = 0xFFFFFFFFu;
    int32_t effectiveMaxTokens() const { return maxTokens > 0 ? maxTokens : INT32_MAX; }
};

struct ImageInput {
    std::string id;
    int width = 0;
    int height = 0;
    std::vector<uint8_t> rgb;       // width*height*3
};

struct ChatTurn {
    std::string role;               // "user" | "assistant" | "system"
    std::string text;
    std::vector<ImageInput> images;
};

struct LoadedModelInfo {
    std::string name;
    std::string architectureDescription;
    uint64_t parameterCount = 0;
    uint64_t sizeBytes = 0;
    int trainingContext = 0;
    int contextLength = 0;
    bool hasVision = false;
    bool supportsThinkingToggle = false;
    int threads = 0;
    int kvBytesPerToken = 0;
};

enum class StopReason { eos, maxTokens, contextFull, cancelled };

struct GenerationStats {
    int promptTokens = 0;
    int cachedTokens = 0;
    int generatedTokens = 0;
    double prefillSeconds = 0;
    double decodeSeconds = 0;
    StopReason stopReason = StopReason::eos;
    int contextShifts = 0;
    int contextLength = 0;
};

struct EngineError : std::runtime_error {
    enum Code { notLoaded, modelLoadFailed, contextCreateFailed, mmprojLoadFailed, templateFailed, tokenizeFailed,
                decodeFailed, contextFull, contextTooSmall, promptTooLong, imageEncodeFailed, busy, cancelled, stateFailed };
    Code code;
    EngineError(Code c, const std::string & msg) : std::runtime_error(msg), code(c) {}
};

struct GenerationCallbacks {
    std::function<void(int promptTokens, int cachedTokens, int droppedTurns, double seconds)> onPromptProcessed;
    std::function<void(const std::string & piece)> onToken;
};

class Engine {
public:
    Engine();
    ~Engine();

    static void setLogCallback(std::function<void(int level, const std::string &)> cb);

    LoadedModelInfo load(const std::string & modelPath, const std::string & mmprojPath, const EngineConfig & config,
                         const std::function<void(float)> & progress);
    void unload();
    bool isLoaded() const { return ctx_ != nullptr; }
    const LoadedModelInfo & info() const { return info_; }

    void clearCache();
    int countTokens(const std::string & text);
    int promptBudget(const SamplingConfig & sampling);
    int planTruncation(const std::string & systemPrompt, const std::vector<ChatTurn> & turns,
                       const SamplingConfig & sampling, double budgetFraction);

    /// Streams a reply. Runs synchronously on the calling thread; `cancel()` may be called from any thread.
    GenerationStats respond(const std::string & systemPrompt, const std::vector<ChatTurn> & turns,
                            const SamplingConfig & sampling, bool continuing, const GenerationCallbacks & cb);
    void cancel() { cancel_.store(true); }

    std::vector<std::vector<float>> embed(const std::vector<std::string> & texts, int maxTokens = 384);

    /// KV snapshot in EdgeChat's own container (same format as iOS): magic, version, n_ctx, cells, length, payload.
    size_t saveState(const std::string & path);
    bool loadState(const std::string & path);

    static int replyReserve(int contextLength, const SamplingConfig & sampling);

private:
    struct Unit {
        llama_token token = -1;      // >= 0 for text tokens
        std::string imageId;         // for images
        int32_t positions = 1;
        int32_t cells = 1;
        bool isImage() const { return token < 0; }
        bool operator==(const Unit & o) const {
            return token == o.token && imageId == o.imageId && positions == o.positions && cells == o.cells;
        }
    };

    struct PreparedPrompt {
        std::vector<Unit> units;
        std::vector<llama_token> tokens;         // text-only path
        mtmd_input_chunks * chunks = nullptr;    // multimodal path
        std::vector<mtmd_bitmap *> bitmaps;
        int tokenCount = 0;
        int positionCount = 0;
        int32_t systemTokenCount = 0;
        int cellCount() const { return tokenCount; }
        ~PreparedPrompt();
    };

    struct Message { std::string role; std::string content; };

    void unloadInternal();
    PreparedPrompt buildPrompt(const std::string & systemPrompt, const std::vector<ChatTurn> & turns, bool continuing);
    PreparedPrompt truncatedHistory(const std::string & systemPrompt, std::vector<ChatTurn> turns, int budget,
                                    bool continuing, int & dropped);
    void evaluate(PreparedPrompt & prepared, size_t matched, int32_t & nPast);
    void decodeTokens(const llama_token * tokens, size_t count, int32_t startPos, bool logitsLast);
    bool shiftContext(int32_t & nPast, int32_t keep);
    llama_sampler * makeSampler(const SamplingConfig & s);
    std::vector<llama_token> tokenize(const std::string & text, bool addSpecial, bool parseSpecial);
    std::string pieceBytes(llama_token token);
    std::string applyChatTemplate(const std::vector<Message> & messages, bool addAssistant);
    int32_t resolvedThreads() const;
    int cachedCells() const;

    llama_model * model_ = nullptr;
    llama_context * ctx_ = nullptr;
    const llama_vocab * vocab_ = nullptr;
    mtmd_context * mctx_ = nullptr;
    llama_batch batch_{};
    bool hasBatch_ = false;
    int32_t nBatch_ = 512;
    EngineConfig config_;
    std::string chatTemplate_;
    bool hasTemplate_ = false;
    std::vector<Unit> cachedUnits_;
    std::atomic<bool> cancel_{false};
    LoadedModelInfo info_;
};

/// Emits only complete UTF-8 sequences (multi-byte characters can be split across tokens).
class UTF8Accumulator {
public:
    bool append(const std::string & bytes, std::string & out);
    bool flush(std::string & out);
private:
    std::string buffer_;
};

} // namespace edgechat
