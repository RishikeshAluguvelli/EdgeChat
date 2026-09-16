#include "edgechat_engine.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <mutex>
#include <sstream>
#include <thread>

namespace edgechat {

// MARK: Logging

static std::function<void(int, const std::string &)> g_log;
static std::mutex g_logLock;

static void ggmlLog(ggml_log_level level, const char * text, void *) {
    if (!text) return;
    std::lock_guard<std::mutex> lock(g_logLock);
    if (g_log) g_log((int) level, text);
}

void Engine::setLogCallback(std::function<void(int, const std::string &)> cb) {
    std::lock_guard<std::mutex> lock(g_logLock);
    g_log = std::move(cb);
}

static void backendInit() {
    static std::once_flag once;
    std::call_once(once, [] {
        llama_backend_init();
        llama_log_set(ggmlLog, nullptr);
        mtmd_helper_log_set(ggmlLog, nullptr);
    });
}

static double secondsSince(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
}

// MARK: UTF-8 accumulator

bool UTF8Accumulator::append(const std::string & bytes, std::string & out) {
    buffer_ += bytes;
    size_t cut = buffer_.size();
    int i = (int) buffer_.size() - 1;
    int trailing = 0;
    while (i >= 0 && trailing < 4) {
        unsigned char b = (unsigned char) buffer_[i];
        if ((b & 0xC0) == 0x80) { i--; trailing++; continue; }
        int need = b < 0x80 ? 1 : (b & 0xE0) == 0xC0 ? 2 : (b & 0xF0) == 0xE0 ? 3 : (b & 0xF8) == 0xF0 ? 4 : 1;
        if (trailing + 1 < need) cut = (size_t) i;
        break;
    }
    if (cut == 0) return false;
    out = buffer_.substr(0, cut);
    buffer_.erase(0, cut);
    return !out.empty();
}

bool UTF8Accumulator::flush(std::string & out) {
    if (buffer_.empty()) return false;
    out = buffer_;
    buffer_.clear();
    return true;
}

// MARK: Lifecycle

Engine::Engine() {}
Engine::~Engine() { unloadInternal(); }

Engine::PreparedPrompt::~PreparedPrompt() {
    if (chunks) mtmd_input_chunks_free(chunks);
    for (auto * b : bitmaps) if (b) mtmd_bitmap_free(b);
}

int32_t Engine::resolvedThreads() const {
    if (config_.threads > 0) return config_.threads;
    int cores = (int) std::thread::hardware_concurrency();
    if (cores <= 0) cores = 4;
    // Big.LITTLE phones: too many threads lands work on slow cores; half the cores (at least 2) is a good default.
    return std::max(2, std::min(cores - 2, 6));
}

LoadedModelInfo Engine::load(const std::string & modelPath, const std::string & mmprojPath, const EngineConfig & config,
                             const std::function<void(float)> & progress) {
    backendInit();
    unloadInternal();
    config_ = config;
    const int32_t gpuLayers = llama_supports_gpu_offload() ? config.gpuLayers : 0;
    const int32_t threads = resolvedThreads();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = gpuLayers;
    mparams.use_extra_bufts = true;
    struct ProgressBox { const std::function<void(float)> * cb; } box{&progress};
    if (progress) {
        mparams.progress_callback = [](float p, void * ud) -> bool {
            auto * b = (ProgressBox *) ud;
            if (b && b->cb && *b->cb) (*b->cb)(p);
            return true;
        };
        mparams.progress_callback_user_data = &box;
    }
    model_ = llama_model_load_from_file(modelPath.c_str(), mparams);
    if (!model_) throw EngineError(EngineError::modelLoadFailed, "Could not load model at " + modelPath);
    vocab_ = llama_model_get_vocab(model_);

    const int nCtxTrain = llama_model_n_ctx_train(model_);
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = std::min(config.contextLength, (uint32_t) std::max(nCtxTrain, 2048));
    cparams.n_batch = std::min(config.batchSize, cparams.n_ctx);
    cparams.n_ubatch = std::min(cparams.n_batch, 512u);
    cparams.n_seq_max = 1;
    cparams.n_threads = threads;
    cparams.n_threads_batch = threads;
    cparams.offload_kqv = gpuLayers > 0;
    cparams.flash_attn_type = config.flashAttention ? LLAMA_FLASH_ATTN_TYPE_AUTO : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    if (config.kvCacheQ8) {
        cparams.type_k = GGML_TYPE_Q8_0;
        cparams.type_v = GGML_TYPE_Q8_0;
        cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    }
    ctx_ = llama_init_from_model(model_, cparams);
    if (!ctx_) { unloadInternal(); throw EngineError(EngineError::contextCreateFailed, "Could not create the inference context (out of memory?). Try a smaller context length."); }
    nBatch_ = (int32_t) cparams.n_batch;
    batch_ = llama_batch_init(nBatch_, 0, 1);
    hasBatch_ = true;

    const char * tmpl = llama_model_chat_template(model_, nullptr);
    hasTemplate_ = tmpl != nullptr;
    chatTemplate_ = tmpl ? tmpl : "";

    bool hasVision = false;
    if (!mmprojPath.empty()) {
        mtmd_context_params mp = mtmd_context_params_default();
        mp.use_gpu = gpuLayers > 0;
        mp.n_threads = threads;
        mp.print_timings = false;
        mp.warmup = false;
        mp.flash_attn_type = cparams.flash_attn_type;
        mctx_ = mtmd_init_from_file(mmprojPath.c_str(), model_, mp);
        if (!mctx_) { unloadInternal(); throw EngineError(EngineError::mmprojLoadFailed, "Could not load the vision projector at " + mmprojPath); }
        hasVision = mtmd_support_vision(mctx_);
    }

    char desc[256] = {0};
    llama_model_desc(model_, desc, sizeof(desc));
    char nameBuf[256] = {0};
    int n = llama_model_meta_val_str(model_, "general.name", nameBuf, sizeof(nameBuf));
    std::string name = n > 0 ? nameBuf : modelPath.substr(modelPath.find_last_of('/') + 1);

    const int nHead = std::max(1, llama_model_n_head(model_));
    const int headDim = llama_model_n_embd(model_) / nHead;
    const int kvPerToken = llama_model_n_layer(model_) * 2 * llama_model_n_head_kv(model_) * headDim * (config.kvCacheQ8 ? 1 : 2);

    info_ = LoadedModelInfo{};
    info_.name = name;
    info_.architectureDescription = desc;
    info_.parameterCount = llama_model_n_params(model_);
    info_.sizeBytes = llama_model_size(model_);
    info_.trainingContext = nCtxTrain;
    info_.contextLength = (int) llama_n_ctx(ctx_);
    info_.hasVision = hasVision;
    info_.supportsThinkingToggle = chatTemplate_.find("enable_thinking") != std::string::npos;
    info_.threads = threads;
    info_.kvBytesPerToken = kvPerToken;
    return info_;
}

void Engine::unload() { unloadInternal(); }

void Engine::unloadInternal() {
    if (hasBatch_) { llama_batch_free(batch_); hasBatch_ = false; }
    if (mctx_) { mtmd_free(mctx_); mctx_ = nullptr; }
    if (ctx_) { llama_free(ctx_); ctx_ = nullptr; }
    if (model_) { llama_model_free(model_); model_ = nullptr; }
    vocab_ = nullptr;
    chatTemplate_.clear();
    hasTemplate_ = false;
    cachedUnits_.clear();
    info_ = LoadedModelInfo{};
}

void Engine::clearCache() {
    if (ctx_) llama_memory_clear(llama_get_memory(ctx_), true);
    cachedUnits_.clear();
}

int Engine::cachedCells() const {
    int n = 0;
    for (auto & u : cachedUnits_) n += u.cells;
    return n;
}

int Engine::countTokens(const std::string & text) {
    if (!vocab_) throw EngineError(EngineError::notLoaded, "No model is loaded.");
    return (int) tokenize(text, false, true).size();
}

int Engine::replyReserve(int contextLength, const SamplingConfig & sampling) {
    return std::min(sampling.effectiveMaxTokens(), std::max(128, contextLength / 4));
}

int Engine::promptBudget(const SamplingConfig & sampling) {
    if (!ctx_) throw EngineError(EngineError::notLoaded, "No model is loaded.");
    const int nCtx = (int) llama_n_ctx(ctx_);
    return nCtx - replyReserve(nCtx, sampling) - 8;
}

int Engine::planTruncation(const std::string & systemPrompt, const std::vector<ChatTurn> & turns,
                           const SamplingConfig & sampling, double budgetFraction) {
    if (!ctx_) throw EngineError(EngineError::notLoaded, "No model is loaded.");
    const int nCtx = (int) llama_n_ctx(ctx_);
    const int full = nCtx - replyReserve(nCtx, sampling) - 8;
    if (full < 128) throw EngineError(EngineError::contextTooSmall, "Context length is too small for the reply budget.");
    const int budget = std::max(128, (int) (full * std::min(1.0, std::max(0.1, budgetFraction))));
    int dropped = 0;
    truncatedHistory(systemPrompt, turns, budget, false, dropped);
    return dropped;
}

// MARK: Snapshots

static const uint32_t kSnapshotMagic = 0x45434B56; // "ECKV"
static const uint32_t kSnapshotVersion = 2;
static const size_t kSnapshotHeader = 4 + 4 + 4 + 4 + 8;

template <typename T> static void putLE(std::string & out, T v) {
    for (size_t i = 0; i < sizeof(T); i++) out.push_back((char) ((uint64_t) v >> (8 * i)));
}
template <typename T> static T getLE(const std::string & in, size_t & off) {
    uint64_t v = 0;
    for (size_t i = 0; i < sizeof(T); i++) v |= (uint64_t) (unsigned char) in[off + i] << (8 * i);
    off += sizeof(T);
    return (T) v;
}

static bool writeAtomic(const std::string & path, const std::string & data) {
    std::string tmp = path + ".tmp";
    { std::ofstream f(tmp, std::ios::binary); if (!f) return false; f.write(data.data(), (std::streamsize) data.size()); if (!f) return false; }
    return std::rename(tmp.c_str(), path.c_str()) == 0;
}

static bool readAll(const std::string & path, std::string & out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::ostringstream ss; ss << f.rdbuf();
    out = ss.str();
    return true;
}

size_t Engine::saveState(const std::string & path) {
    if (!ctx_) throw EngineError(EngineError::notLoaded, "No model is loaded.");
    const size_t size = llama_state_seq_get_size(ctx_, 0);
    if (size == 0) throw EngineError(EngineError::stateFailed, "Could not save the conversation cache.");
    std::vector<uint8_t> payload(size);
    const size_t written = llama_state_seq_get_data(ctx_, payload.data(), size, 0);
    if (written == 0 || written > size) throw EngineError(EngineError::stateFailed, "Could not save the conversation cache.");
    std::string data;
    data.reserve(kSnapshotHeader + written);
    putLE<uint32_t>(data, kSnapshotMagic);
    putLE<uint32_t>(data, kSnapshotVersion);
    putLE<uint32_t>(data, llama_n_ctx(ctx_));
    putLE<uint32_t>(data, (uint32_t) std::max(0, cachedCells()));
    putLE<uint64_t>(data, (uint64_t) written);
    data.append((const char *) payload.data(), written);
    std::string units;
    for (auto & u : cachedUnits_) {
        if (u.isImage()) units += "i:" + std::to_string(u.positions) + ":" + std::to_string(u.cells) + ":" + u.imageId + "\n";
        else units += "t:" + std::to_string(u.token) + "\n";
    }
    if (!writeAtomic(path + ".units", units) || !writeAtomic(path, data)) throw EngineError(EngineError::stateFailed, "Could not save the conversation cache.");
    return written;
}

bool Engine::loadState(const std::string & path) {
    bool ok = false;
    if (ctx_) {
        llama_memory_t mem = llama_get_memory(ctx_);
        llama_memory_clear(mem, true);
        cachedUnits_.clear();
        std::string data, sidecar;
        std::vector<Unit> units;
        bool parsed = readAll(path, data) && data.size() > kSnapshotHeader && readAll(path + ".units", sidecar);
        if (parsed) {
            std::istringstream lines(sidecar);
            std::string line;
            while (std::getline(lines, line)) {
                if (line.empty()) continue;
                if (line.rfind("t:", 0) == 0) { Unit u; u.token = (llama_token) std::stol(line.substr(2)); units.push_back(u); }
                else if (line.rfind("i:", 0) == 0) {
                    size_t a = line.find(':', 2), b = a == std::string::npos ? a : line.find(':', a + 1);
                    if (a == std::string::npos || b == std::string::npos) { parsed = false; break; }
                    Unit u; u.token = -1;
                    u.positions = std::stoi(line.substr(2, a - 2));
                    u.cells = std::stoi(line.substr(a + 1, b - a - 1));
                    u.imageId = line.substr(b + 1);
                    units.push_back(u);
                } else { parsed = false; break; }
            }
        }
        if (parsed) {
            size_t off = 0;
            uint32_t magic = getLE<uint32_t>(data, off), version = getLE<uint32_t>(data, off), nCtx = getLE<uint32_t>(data, off), cells = getLE<uint32_t>(data, off);
            uint64_t length = getLE<uint64_t>(data, off);
            int32_t unitCells = 0, unitPos = 0;
            for (auto & u : units) { unitCells += u.cells; unitPos += u.positions; }
            if (magic == kSnapshotMagic && version == kSnapshotVersion && nCtx == llama_n_ctx(ctx_) &&
                length == data.size() - kSnapshotHeader && (int32_t) cells == unitCells && cells > 0 && cells <= nCtx) {
                size_t read = llama_state_seq_set_data(ctx_, (const uint8_t *) data.data() + kSnapshotHeader, (size_t) length, 0);
                if (read == length && llama_memory_seq_pos_max(mem, 0) == unitPos - 1) {
                    cachedUnits_ = units;
                    ok = true;
                } else {
                    llama_memory_clear(mem, true);
                }
            }
        }
    }
    if (!ok) { std::remove(path.c_str()); std::remove((path + ".units").c_str()); }
    return ok;
}

// MARK: Embeddings

std::vector<std::vector<float>> Engine::embed(const std::vector<std::string> & texts, int maxTokens) {
    if (!model_ || !vocab_) throw EngineError(EngineError::notLoaded, "No model is loaded.");
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = maxTokens + 8;
    cp.n_batch = maxTokens + 8;
    cp.n_ubatch = maxTokens + 8;
    cp.n_seq_max = 1;
    cp.n_threads = resolvedThreads();
    cp.n_threads_batch = resolvedThreads();
    cp.embeddings = true;
    cp.pooling_type = LLAMA_POOLING_TYPE_MEAN;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
    llama_context * ectx = llama_init_from_model(model_, cp);
    if (!ectx) throw EngineError(EngineError::contextCreateFailed, "Could not create the embedding context.");
    const int nEmbd = llama_model_n_embd(model_);
    llama_batch batch = llama_batch_init(maxTokens + 8, 0, 1);
    std::vector<std::vector<float>> out;
    for (auto & text : texts) {
        std::vector<llama_token> tokens = tokenize(text, true, false);
        if ((int) tokens.size() > maxTokens) tokens.resize(maxTokens);
        if (tokens.empty()) { out.emplace_back(); continue; }
        llama_memory_clear(llama_get_memory(ectx), true);
        batch.n_tokens = (int32_t) tokens.size();
        for (size_t i = 0; i < tokens.size(); i++) {
            batch.token[i] = tokens[i]; batch.pos[i] = (llama_pos) i; batch.n_seq_id[i] = 1; batch.seq_id[i][0] = 0; batch.logits[i] = 1;
        }
        const float * e = nullptr;
        if (llama_decode(ectx, batch) != 0 || !(e = llama_get_embeddings_seq(ectx, 0))) { out.emplace_back(); continue; }
        std::vector<float> v(e, e + nEmbd);
        float norm = 0; for (float x : v) norm += x * x; norm = std::sqrt(norm);
        if (norm > 0) for (float & x : v) x /= norm;
        out.push_back(std::move(v));
    }
    llama_batch_free(batch);
    llama_free(ectx);
    return out;
}

// MARK: Generation

GenerationStats Engine::respond(const std::string & systemPrompt, const std::vector<ChatTurn> & turns,
                                const SamplingConfig & sampling, bool continuing, const GenerationCallbacks & cb) {
    if (!ctx_ || !vocab_) throw EngineError(EngineError::notLoaded, "No model is loaded.");
    cancel_.store(false);
    const int nCtx = (int) llama_n_ctx(ctx_);
    const int budget = nCtx - replyReserve(nCtx, sampling) - 8;
    if (budget < 128) throw EngineError(EngineError::contextTooSmall, "Context length is too small for the reply budget.");
    if (continuing && (turns.empty() || turns.back().role != "assistant")) throw EngineError(EngineError::tokenizeFailed, "Nothing to continue.");

    // 1. Build the prompt, dropping the oldest turns until it fits the budget.
    int dropped = 0;
    PreparedPrompt prepared = truncatedHistory(systemPrompt, turns, budget, continuing, dropped);
    const auto & units = prepared.units;
    if (units.empty()) throw EngineError(EngineError::tokenizeFailed, "Tokenization failed.");

    // 2. Reuse the longest common prefix already in the KV cache.
    size_t matched = 0;
    while (matched < units.size() && matched < cachedUnits_.size() && units[matched] == cachedUnits_[matched]) matched++;
    if (matched == units.size()) matched--;   // always re-decode the last token to get logits
    int32_t nPastMatched = 0; int cachedCellsN = 0;
    for (size_t i = 0; i < matched; i++) { nPastMatched += units[i].positions; cachedCellsN += units[i].cells; }
    cachedUnits_.resize(matched);
    llama_memory_t mem = llama_get_memory(ctx_);
    llama_memory_seq_rm(mem, 0, nPastMatched, -1);

    // 3. Evaluate the remaining prompt.
    auto tPrefill = std::chrono::steady_clock::now();
    int32_t nPast = nPastMatched;
    evaluate(prepared, matched, nPast);
    int nCells = cachedCells();
    const double prefillSeconds = secondsSince(tPrefill);
    if (cb.onPromptProcessed) cb.onPromptProcessed(prepared.tokenCount, cachedCellsN, dropped, prefillSeconds);

    // 4. Sample tokens. "Unlimited" still has a ceiling so a looping model cannot shift the context forever.
    const int replyCap = sampling.maxTokens > 0 ? sampling.maxTokens : std::max(4096, nCtx * 2);
    llama_sampler * sampler = makeSampler(sampling);
    UTF8Accumulator acc;
    int generated = 0, shifts = 0;
    StopReason reason = StopReason::eos;
    auto tDecode = std::chrono::steady_clock::now();
    std::string piece;
    while (true) {
        if (cancel_.load()) { reason = StopReason::cancelled; break; }
        llama_token tok = llama_sampler_sample(sampler, ctx_, -1);
        if (llama_vocab_is_eog(vocab_, tok)) { reason = StopReason::eos; break; }
        if (acc.append(pieceBytes(tok), piece) && cb.onToken) cb.onToken(piece);
        generated++;
        if (generated >= replyCap) { reason = StopReason::maxTokens; break; }
        if (nCells + 1 >= nCtx) {
            if (config_.contextShift && shiftContext(nPast, prepared.systemTokenCount)) { nCells = cachedCells(); shifts++; }
            else { reason = StopReason::contextFull; break; }
        }
        try {
            decodeTokens(&tok, 1, nPast, true);
        } catch (const EngineError & e) {
            if (e.code == EngineError::cancelled) { reason = StopReason::cancelled; break; }
            if (e.code != EngineError::contextFull) { llama_sampler_free(sampler); throw; }
            bool recovered = false;
            if (config_.contextShift && shiftContext(nPast, prepared.systemTokenCount)) {
                try { decodeTokens(&tok, 1, nPast, true); recovered = true; nCells = cachedCells(); shifts++; } catch (...) {}
            }
            if (!recovered) { reason = StopReason::contextFull; break; }
        }
        nPast += 1;
        nCells += 1;
    }
    if (acc.flush(piece) && cb.onToken) cb.onToken(piece);
    llama_sampler_free(sampler);

    GenerationStats stats;
    stats.promptTokens = prepared.tokenCount;
    stats.cachedTokens = cachedCellsN;
    stats.generatedTokens = generated;
    stats.prefillSeconds = prefillSeconds;
    stats.decodeSeconds = secondsSince(tDecode);
    stats.stopReason = reason;
    stats.contextShifts = shifts;
    stats.contextLength = nCtx;
    return stats;
}

// UTF-8 aware character count / cut helpers for over-long turns.
static size_t utf8Length(const std::string & s) {
    size_t n = 0;
    for (unsigned char c : s) if ((c & 0xC0) != 0x80) n++;
    return n;
}
static size_t utf8Offset(const std::string & s, size_t chars) {
    size_t n = 0, i = 0;
    for (; i < s.size(); i++) { if (((unsigned char) s[i] & 0xC0) != 0x80) { if (n == chars) break; n++; } }
    return i;
}
static std::string truncateMiddle(const std::string & text, size_t maxChars) {
    const size_t len = utf8Length(text);
    if (len <= maxChars || maxChars <= 64) return text;
    std::string marker = "\n\n[… " + std::to_string(len - maxChars) + " characters omitted …]\n\n";
    const size_t budget = maxChars > utf8Length(marker) ? maxChars - utf8Length(marker) : 0;
    const size_t head = (size_t) (budget * 0.7), tail = budget - head;
    return text.substr(0, utf8Offset(text, head)) + marker + text.substr(utf8Offset(text, len - tail));
}

Engine::PreparedPrompt Engine::truncatedHistory(const std::string & systemPrompt, std::vector<ChatTurn> history, int budget,
                                                bool continuing, int & dropped) {
    dropped = 0;
    while (true) {
        PreparedPrompt prepared = buildPrompt(systemPrompt, history, continuing);
        if (prepared.cellCount() <= budget) return prepared;
        if (history.size() > 1) {
            size_t n = std::min<size_t>(2, history.size() - 1);
            history.erase(history.begin(), history.begin() + (long) n);
            dropped += (int) n;
            continue;
        }
        // A single over-long turn (big document, or a long partial reply being continued): shrink its text.
        std::string & text = history[0].text;
        const size_t len = utf8Length(text);
        if (len <= 512) throw EngineError(EngineError::promptTooLong, "The message is too long to fit in the context window.");
        const double ratio = (double) budget / (double) prepared.cellCount() * 0.9;
        const size_t keep = (size_t) (len * ratio);
        if (continuing) text = text.substr(utf8Offset(text, len - keep));
        else text = truncateMiddle(text, keep);
    }
}

bool Engine::shiftContext(int32_t & nPast, int32_t keep) {
    if (!ctx_) return false;
    if (cachedUnits_.size() != (size_t) nPast) return false;
    for (auto & u : cachedUnits_) if (u.isImage()) return false;
    const int32_t nKeep = std::min(std::max(0, keep), nPast / 2);
    const int32_t nDiscard = (nPast - nKeep) / 2;
    if (nDiscard <= 0) return false;
    llama_memory_t mem = llama_get_memory(ctx_);
    if (!llama_memory_seq_rm(mem, 0, nKeep, nKeep + nDiscard)) return false;
    llama_memory_seq_add(mem, 0, nKeep + nDiscard, nPast, -nDiscard);
    cachedUnits_.erase(cachedUnits_.begin() + nKeep, cachedUnits_.begin() + nKeep + nDiscard);
    nPast -= nDiscard;
    return true;
}

static const char * kContinueSentinel = "\x01\x02" "EDGECHAT_CONTINUE" "\x02\x01";

Engine::PreparedPrompt Engine::buildPrompt(const std::string & systemPrompt, const std::vector<ChatTurn> & turns, bool continuing) {
    PreparedPrompt prepared;
    std::vector<Message> messages;
    bool systemBlank = systemPrompt.find_first_not_of(" \t\r\n") == std::string::npos;
    if (!systemBlank) messages.push_back({"system", systemPrompt});
    const char * marker = mctx_ ? mtmd_default_marker() : nullptr;
    for (auto & turn : turns) {
        std::string content = turn.text;
        if (marker && !turn.images.empty()) {
            std::string prefix;
            for (size_t i = 0; i < turn.images.size(); i++) { if (i) prefix += "\n"; prefix += marker; }
            content = prefix + "\n" + turn.text;
            for (auto & image : turn.images) {
                mtmd_bitmap * bitmap = mtmd_bitmap_init((uint32_t) image.width, (uint32_t) image.height, image.rgb.data());
                if (bitmap) mtmd_bitmap_set_id(bitmap, image.id.c_str());
                prepared.bitmaps.push_back(bitmap);
            }
        }
        messages.push_back({turn.role, content});
    }
    const bool thinkOff = config_.disableThinking && chatTemplate_.find("enable_thinking") != std::string::npos;
    std::string text;
    if (continuing && !messages.empty() && messages.back().role == "assistant") {
        std::string partial = messages.back().content;
        if (thinkOff) partial = "<think>\n\n</think>\n\n" + partial;
        messages.back().content = partial + kContinueSentinel;
        std::string formatted = applyChatTemplate(messages, false);
        size_t cut = formatted.find(kContinueSentinel);
        if (cut == std::string::npos) throw EngineError(EngineError::templateFailed, "The model's chat template could not be applied.");
        text = formatted.substr(0, cut);
    } else {
        text = applyChatTemplate(messages, true);
        if (thinkOff) text += "<think>\n\n</think>\n\n";
    }

    if (mctx_) {
        prepared.chunks = mtmd_input_chunks_init();
        if (!prepared.chunks) throw EngineError(EngineError::tokenizeFailed, "Tokenization failed.");
        mtmd_input_text input{text.c_str(), text.size(), true, true};
        int32_t rc = mtmd_tokenize(mctx_, prepared.chunks, &input, (const mtmd_bitmap **) prepared.bitmaps.data(), prepared.bitmaps.size());
        if (rc != 0) throw EngineError(EngineError::tokenizeFailed, "Tokenization failed.");
        const size_t n = mtmd_input_chunks_size(prepared.chunks);
        for (size_t i = 0; i < n; i++) {
            const mtmd_input_chunk * chunk = mtmd_input_chunks_get(prepared.chunks, i);
            if (!chunk) continue;
            switch (mtmd_input_chunk_get_type(chunk)) {
                case MTMD_INPUT_CHUNK_TYPE_TEXT: {
                    size_t count = 0;
                    const llama_token * toks = mtmd_input_chunk_get_tokens_text(chunk, &count);
                    for (size_t j = 0; j < count; j++) { Unit u; u.token = toks[j]; prepared.units.push_back(u); }
                    break;
                }
                case MTMD_INPUT_CHUNK_TYPE_IMAGE: {
                    const char * id = mtmd_input_chunk_get_id(chunk);
                    Unit u; u.token = -1; u.imageId = id ? id : "img-" + std::to_string(i);
                    u.positions = mtmd_input_chunk_get_n_pos(chunk);
                    u.cells = (int32_t) mtmd_input_chunk_get_n_tokens(chunk);
                    prepared.units.push_back(u);
                    break;
                }
                default: throw EngineError(EngineError::tokenizeFailed, "Unsupported input chunk.");
            }
        }
        prepared.tokenCount = (int) mtmd_helper_get_n_tokens(prepared.chunks);
        prepared.positionCount = (int) mtmd_helper_get_n_pos(prepared.chunks);
    } else {
        prepared.tokens = tokenize(text, true, true);
        for (auto t : prepared.tokens) { Unit u; u.token = t; prepared.units.push_back(u); }
        prepared.tokenCount = (int) prepared.tokens.size();
        prepared.positionCount = prepared.tokenCount;
        if (!messages.empty() && messages.front().role == "system") {
            try {
                std::string sysText = applyChatTemplate({messages.front()}, false);
                std::vector<llama_token> sysTokens = tokenize(sysText, true, true);
                if (sysTokens.size() < prepared.tokens.size() && std::equal(sysTokens.begin(), sysTokens.end(), prepared.tokens.begin()))
                    prepared.systemTokenCount = (int32_t) sysTokens.size();
            } catch (...) {}
        }
    }
    return prepared;
}

void Engine::evaluate(PreparedPrompt & prepared, size_t matched, int32_t & nPast) {
    if (!prepared.chunks) {
        const size_t count = prepared.tokens.size() - matched;
        decodeTokens(prepared.tokens.data() + matched, count, nPast, true);
        nPast += (int32_t) count;
        return;
    }
    if (!ctx_ || !mctx_) throw EngineError(EngineError::notLoaded, "No model is loaded.");
    const size_t n = mtmd_input_chunks_size(prepared.chunks);
    size_t unitIndex = 0;
    for (size_t i = 0; i < n; i++) {
        const mtmd_input_chunk * chunk = mtmd_input_chunks_get(prepared.chunks, i);
        if (!chunk) continue;
        const bool isLast = i == n - 1;
        switch (mtmd_input_chunk_get_type(chunk)) {
            case MTMD_INPUT_CHUNK_TYPE_TEXT: {
                size_t count = 0;
                const llama_token * toks = mtmd_input_chunk_get_tokens_text(chunk, &count);
                if (unitIndex + count <= matched) { unitIndex += count; continue; }
                const size_t skip = matched > unitIndex ? matched - unitIndex : 0;
                decodeTokens(toks + skip, count - skip, nPast, isLast);
                nPast += (int32_t) (count - skip);
                unitIndex += count;
                break;
            }
            case MTMD_INPUT_CHUNK_TYPE_IMAGE: {
                if (unitIndex < matched) { unitIndex++; continue; }
                if (cancel_.load()) throw EngineError(EngineError::cancelled, "Generation was cancelled.");
                llama_pos newPast = nPast;
                int32_t rc = mtmd_helper_eval_chunk_single(mctx_, ctx_, chunk, nPast, 0, nBatch_, isLast, &newPast);
                if (rc != 0) throw EngineError(EngineError::imageEncodeFailed, "The vision encoder failed (code " + std::to_string(rc) + ").");
                const char * id = mtmd_input_chunk_get_id(chunk);
                Unit u; u.token = -1; u.imageId = id ? id : "img-" + std::to_string(i);
                u.positions = mtmd_input_chunk_get_n_pos(chunk);
                u.cells = (int32_t) mtmd_input_chunk_get_n_tokens(chunk);
                cachedUnits_.push_back(u);
                nPast = newPast;
                unitIndex++;
                break;
            }
            default: throw EngineError(EngineError::tokenizeFailed, "Unsupported input chunk.");
        }
    }
}

void Engine::decodeTokens(const llama_token * tokens, size_t count, int32_t startPos, bool logitsLast) {
    if (!ctx_) throw EngineError(EngineError::notLoaded, "No model is loaded.");
    int32_t pos = startPos;
    size_t i = 0;
    while (i < count) {
        if (cancel_.load()) throw EngineError(EngineError::cancelled, "Generation was cancelled.");
        const size_t n = std::min<size_t>((size_t) nBatch_, count - i);
        batch_.n_tokens = (int32_t) n;
        for (size_t j = 0; j < n; j++) {
            batch_.token[j] = tokens[i + j];
            batch_.pos[j] = pos + (int32_t) j;
            batch_.n_seq_id[j] = 1;
            batch_.seq_id[j][0] = 0;
            batch_.logits[j] = 0;
        }
        if (logitsLast && i + n == count) batch_.logits[n - 1] = 1;
        const int rc = llama_decode(ctx_, batch_);
        if (rc != 0) {
            { std::lock_guard<std::mutex> lock(g_logLock); if (g_log) g_log(3, "llama_decode returned " + std::to_string(rc) + " at pos " + std::to_string(pos) + "\n"); }
            if (rc == 1) throw EngineError(EngineError::contextFull, "The context window is full.");
            throw EngineError(EngineError::decodeFailed, "Inference failed (llama_decode returned " + std::to_string(rc) + ").");
        }
        for (size_t j = 0; j < n; j++) { Unit u; u.token = tokens[i + j]; cachedUnits_.push_back(u); }
        pos += (int32_t) n;
        i += n;
    }
}

llama_sampler * Engine::makeSampler(const SamplingConfig & s) {
    llama_sampler_chain_params p = llama_sampler_chain_default_params();
    p.no_perf = true;
    llama_sampler * chain = llama_sampler_chain_init(p);
    if (s.repeatPenalty != 1.0f && s.repeatLastN != 0)
        llama_sampler_chain_add(chain, llama_sampler_init_penalties(llama_vocab_n_tokens(vocab_), s.repeatLastN, s.repeatPenalty, 0, 0));
    if (s.temperature <= 0) {
        llama_sampler_chain_add(chain, llama_sampler_init_greedy());
    } else {
        if (s.topK > 0) llama_sampler_chain_add(chain, llama_sampler_init_top_k(s.topK));
        if (s.topP < 1) llama_sampler_chain_add(chain, llama_sampler_init_top_p(s.topP, 1));
        if (s.minP > 0) llama_sampler_chain_add(chain, llama_sampler_init_min_p(s.minP, 1));
        llama_sampler_chain_add(chain, llama_sampler_init_temp(s.temperature));
        llama_sampler_chain_add(chain, llama_sampler_init_dist(s.seed));
    }
    return chain;
}

std::vector<llama_token> Engine::tokenize(const std::string & text, bool addSpecial, bool parseSpecial) {
    if (!vocab_) throw EngineError(EngineError::notLoaded, "No model is loaded.");
    std::vector<llama_token> tokens(text.size() + 16);
    int n = llama_tokenize(vocab_, text.c_str(), (int32_t) text.size(), tokens.data(), (int32_t) tokens.size(), addSpecial, parseSpecial);
    if (n < 0) {
        tokens.resize((size_t) -n);
        n = llama_tokenize(vocab_, text.c_str(), (int32_t) text.size(), tokens.data(), (int32_t) tokens.size(), addSpecial, parseSpecial);
    }
    if (n < 0) throw EngineError(EngineError::tokenizeFailed, "Tokenization failed.");
    tokens.resize((size_t) n);
    return tokens;
}

std::string Engine::pieceBytes(llama_token token) {
    char buf[128];
    int n = llama_token_to_piece(vocab_, token, buf, sizeof(buf), 0, true);
    if (n >= 0) return std::string(buf, (size_t) n);
    std::vector<char> big((size_t) -n);
    n = llama_token_to_piece(vocab_, token, big.data(), (int32_t) big.size(), 0, true);
    return n > 0 ? std::string(big.data(), (size_t) n) : std::string();
}

std::string Engine::applyChatTemplate(const std::vector<Message> & messages, bool addAssistant) {
    std::vector<llama_chat_message> cMessages;
    size_t size = 2048;
    for (auto & m : messages) { cMessages.push_back({m.role.c_str(), m.content.c_str()}); size += (m.role.size() + m.content.size()) * 2; }
    std::vector<char> buf(size);
    const char * tmpl = hasTemplate_ ? chatTemplate_.c_str() : "chatml";
    int n = llama_chat_apply_template(tmpl, cMessages.data(), cMessages.size(), addAssistant, buf.data(), (int32_t) buf.size());
    if (n < 0) {
        tmpl = "chatml";
        n = llama_chat_apply_template(tmpl, cMessages.data(), cMessages.size(), addAssistant, buf.data(), (int32_t) buf.size());
    }
    if (n > (int) buf.size()) {
        buf.resize((size_t) n + 1);
        n = llama_chat_apply_template(tmpl, cMessages.data(), cMessages.size(), addAssistant, buf.data(), (int32_t) buf.size());
    }
    if (n < 0) throw EngineError(EngineError::templateFailed, "The model's chat template could not be applied.");
    return std::string(buf.data(), (size_t) n);
}

} // namespace edgechat
