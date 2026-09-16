// JNI bridge for com.rishikesh.edgechat.engine.NativeEngine.
#include <jni.h>
#include <android/log.h>
#include <memory>
#include <string>
#include <vector>

#include "edgechat_engine.h"

using namespace edgechat;

namespace {

JavaVM * g_vm = nullptr;
jobject g_logger = nullptr;   // global ref to a NativeLogger, or null

std::string jstr(JNIEnv * env, jstring s) {
    if (!s) return "";
    const char * c = env->GetStringUTFChars(s, nullptr);
    std::string out = c ? c : "";
    env->ReleaseStringUTFChars(s, c);
    return out;
}

jstring jstring_utf8(JNIEnv * env, const std::string & s) {
    // NewStringUTF wants modified UTF-8; go through java.lang.String(byte[], "UTF-8") to be safe with emoji etc.
    jbyteArray bytes = env->NewByteArray((jsize) s.size());
    env->SetByteArrayRegion(bytes, 0, (jsize) s.size(), (const jbyte *) s.data());
    jclass strClass = env->FindClass("java/lang/String");
    jmethodID ctor = env->GetMethodID(strClass, "<init>", "([BLjava/lang/String;)V");
    jstring enc = env->NewStringUTF("UTF-8");
    jstring out = (jstring) env->NewObject(strClass, ctor, bytes, enc);
    env->DeleteLocalRef(bytes); env->DeleteLocalRef(enc); env->DeleteLocalRef(strClass);
    return out;
}

void throwEngineError(JNIEnv * env, const EngineError & e) {
    jclass cls = env->FindClass("com/rishikesh/edgechat/engine/EngineException");
    if (!cls) return;
    jmethodID ctor = env->GetMethodID(cls, "<init>", "(ILjava/lang/String;)V");
    jstring msg = jstring_utf8(env, e.what());
    jobject ex = env->NewObject(cls, ctor, (jint) e.code, msg);
    env->Throw((jthrowable) ex);
}

Engine * engine(jlong handle) { return reinterpret_cast<Engine *>(handle); }

std::vector<ChatTurn> buildTurns(JNIEnv * env, jintArray roles, jobjectArray texts, jintArray imageTurn, jobjectArray imageIds,
                                 jintArray imageW, jintArray imageH, jobjectArray imageRgb) {
    std::vector<ChatTurn> turns;
    const jsize n = env->GetArrayLength(roles);
    jint * r = env->GetIntArrayElements(roles, nullptr);
    for (jsize i = 0; i < n; i++) {
        ChatTurn t;
        t.role = r[i] == 0 ? "system" : r[i] == 1 ? "user" : "assistant";
        jstring s = (jstring) env->GetObjectArrayElement(texts, i);
        t.text = jstr(env, s);
        env->DeleteLocalRef(s);
        turns.push_back(std::move(t));
    }
    env->ReleaseIntArrayElements(roles, r, JNI_ABORT);
    if (imageTurn) {
        const jsize m = env->GetArrayLength(imageTurn);
        jint * it = env->GetIntArrayElements(imageTurn, nullptr);
        jint * w = env->GetIntArrayElements(imageW, nullptr);
        jint * h = env->GetIntArrayElements(imageH, nullptr);
        for (jsize i = 0; i < m; i++) {
            if (it[i] < 0 || it[i] >= (jint) turns.size()) continue;
            ImageInput img;
            jstring id = (jstring) env->GetObjectArrayElement(imageIds, i);
            img.id = jstr(env, id); env->DeleteLocalRef(id);
            img.width = w[i]; img.height = h[i];
            jbyteArray rgb = (jbyteArray) env->GetObjectArrayElement(imageRgb, i);
            const jsize len = env->GetArrayLength(rgb);
            img.rgb.resize((size_t) len);
            env->GetByteArrayRegion(rgb, 0, len, (jbyte *) img.rgb.data());
            env->DeleteLocalRef(rgb);
            if ((int) img.rgb.size() == img.width * img.height * 3) turns[(size_t) it[i]].images.push_back(std::move(img));
        }
        env->ReleaseIntArrayElements(imageTurn, it, JNI_ABORT);
        env->ReleaseIntArrayElements(imageW, w, JNI_ABORT);
        env->ReleaseIntArrayElements(imageH, h, JNI_ABORT);
    }
    return turns;
}

SamplingConfig sampling(jfloat temperature, jint topK, jfloat topP, jfloat minP, jfloat repeatPenalty, jint repeatLastN, jint maxTokens, jint seed) {
    SamplingConfig s;
    s.temperature = temperature; s.topK = topK; s.topP = topP; s.minP = minP;
    s.repeatPenalty = repeatPenalty; s.repeatLastN = repeatLastN; s.maxTokens = maxTokens; s.seed = (uint32_t) seed;
    return s;
}

} // namespace

extern "C" {

JNIEXPORT jint JNI_OnLoad(JavaVM * vm, void *) {
    g_vm = vm;
    Engine::setLogCallback([](int level, const std::string & text) {
        int prio = level >= 4 ? ANDROID_LOG_ERROR : level == 3 ? ANDROID_LOG_WARN : ANDROID_LOG_INFO;
        __android_log_write(prio, "llama", text.c_str());
        if (!g_logger || !g_vm || level < 3) return;
        JNIEnv * env = nullptr;
        bool attached = false;
        if (g_vm->GetEnv((void **) &env, JNI_VERSION_1_6) != JNI_OK) {
            if (g_vm->AttachCurrentThread(&env, nullptr) != JNI_OK) return;
            attached = true;
        }
        jclass cls = env->GetObjectClass(g_logger);
        jmethodID m = env->GetMethodID(cls, "log", "(ILjava/lang/String;)V");
        jstring s = jstring_utf8(env, text);
        env->CallVoidMethod(g_logger, m, (jint) level, s);
        env->DeleteLocalRef(s); env->DeleteLocalRef(cls);
        if (attached) g_vm->DetachCurrentThread();
    });
    return JNI_VERSION_1_6;
}

JNIEXPORT void JNICALL Java_com_rishikesh_edgechat_engine_NativeEngine_setLogger(JNIEnv * env, jobject, jobject logger) {
    if (g_logger) { env->DeleteGlobalRef(g_logger); g_logger = nullptr; }
    if (logger) g_logger = env->NewGlobalRef(logger);
}

JNIEXPORT jlong JNICALL Java_com_rishikesh_edgechat_engine_NativeEngine_create(JNIEnv *, jobject) {
    return reinterpret_cast<jlong>(new Engine());
}

JNIEXPORT void JNICALL Java_com_rishikesh_edgechat_engine_NativeEngine_destroy(JNIEnv *, jobject, jlong handle) {
    delete engine(handle);
}

JNIEXPORT jobject JNICALL Java_com_rishikesh_edgechat_engine_NativeEngine_load(
        JNIEnv * env, jobject, jlong handle, jstring modelPath, jstring mmprojPath,
        jint contextLength, jint batchSize, jint gpuLayers, jint threads, jboolean flashAttention, jboolean kvCacheQ8,
        jboolean disableThinking, jboolean contextShift, jobject progress) {
    EngineConfig cfg;
    cfg.contextLength = (uint32_t) contextLength; cfg.batchSize = (uint32_t) batchSize; cfg.gpuLayers = gpuLayers; cfg.threads = threads;
    cfg.flashAttention = flashAttention; cfg.kvCacheQ8 = kvCacheQ8; cfg.disableThinking = disableThinking; cfg.contextShift = contextShift;
    std::function<void(float)> cb;
    jmethodID onProgress = nullptr;
    if (progress) {
        jclass pc = env->GetObjectClass(progress);
        onProgress = env->GetMethodID(pc, "onProgress", "(F)V");
        env->DeleteLocalRef(pc);
        cb = [env, progress, onProgress](float p) { env->CallVoidMethod(progress, onProgress, (jfloat) p); };
    }
    try {
        LoadedModelInfo info = engine(handle)->load(jstr(env, modelPath), jstr(env, mmprojPath), cfg, cb);
        jclass cls = env->FindClass("com/rishikesh/edgechat/engine/NativeModelInfo");
        jmethodID ctor = env->GetMethodID(cls, "<init>", "(Ljava/lang/String;Ljava/lang/String;JJIIZZII)V");
        jstring name = jstring_utf8(env, info.name), arch = jstring_utf8(env, info.architectureDescription);
        return env->NewObject(cls, ctor, name, arch, (jlong) info.parameterCount, (jlong) info.sizeBytes, (jint) info.trainingContext,
                              (jint) info.contextLength, (jboolean) info.hasVision, (jboolean) info.supportsThinkingToggle,
                              (jint) info.threads, (jint) info.kvBytesPerToken);
    } catch (const EngineError & e) { throwEngineError(env, e); return nullptr; }
}

JNIEXPORT void JNICALL Java_com_rishikesh_edgechat_engine_NativeEngine_unload(JNIEnv *, jobject, jlong handle) { engine(handle)->unload(); }
JNIEXPORT void JNICALL Java_com_rishikesh_edgechat_engine_NativeEngine_clearCache(JNIEnv *, jobject, jlong handle) { engine(handle)->clearCache(); }
JNIEXPORT void JNICALL Java_com_rishikesh_edgechat_engine_NativeEngine_cancel(JNIEnv *, jobject, jlong handle) { engine(handle)->cancel(); }

JNIEXPORT jint JNICALL Java_com_rishikesh_edgechat_engine_NativeEngine_countTokens(JNIEnv * env, jobject, jlong handle, jstring text) {
    try { return engine(handle)->countTokens(jstr(env, text)); } catch (const EngineError & e) { throwEngineError(env, e); return 0; }
}

JNIEXPORT jint JNICALL Java_com_rishikesh_edgechat_engine_NativeEngine_promptBudget(JNIEnv * env, jobject, jlong handle, jint maxTokens) {
    SamplingConfig s; s.maxTokens = maxTokens;
    try { return engine(handle)->promptBudget(s); } catch (const EngineError & e) { throwEngineError(env, e); return 0; }
}

JNIEXPORT jint JNICALL Java_com_rishikesh_edgechat_engine_NativeEngine_planTruncation(
        JNIEnv * env, jobject, jlong handle, jstring systemPrompt, jintArray roles, jobjectArray texts,
        jintArray imageTurn, jobjectArray imageIds, jintArray imageW, jintArray imageH, jobjectArray imageRgb,
        jint maxTokens, jdouble budgetFraction) {
    try {
        auto turns = buildTurns(env, roles, texts, imageTurn, imageIds, imageW, imageH, imageRgb);
        SamplingConfig s; s.maxTokens = maxTokens;
        return engine(handle)->planTruncation(jstr(env, systemPrompt), turns, s, budgetFraction);
    } catch (const EngineError & e) { throwEngineError(env, e); return 0; }
}

JNIEXPORT jobject JNICALL Java_com_rishikesh_edgechat_engine_NativeEngine_respond(
        JNIEnv * env, jobject, jlong handle, jstring systemPrompt, jintArray roles, jobjectArray texts,
        jintArray imageTurn, jobjectArray imageIds, jintArray imageW, jintArray imageH, jobjectArray imageRgb,
        jfloat temperature, jint topK, jfloat topP, jfloat minP, jfloat repeatPenalty, jint repeatLastN, jint maxTokens, jint seed,
        jboolean continuing, jobject listener) {
    jclass lc = env->GetObjectClass(listener);
    jmethodID onPrompt = env->GetMethodID(lc, "onPromptProcessed", "(IIID)V");
    jmethodID onToken = env->GetMethodID(lc, "onToken", "(Ljava/lang/String;)V");
    env->DeleteLocalRef(lc);
    GenerationCallbacks cb;
    cb.onPromptProcessed = [&](int p, int c, int d, double s) { env->CallVoidMethod(listener, onPrompt, (jint) p, (jint) c, (jint) d, (jdouble) s); };
    cb.onToken = [&](const std::string & piece) {
        jstring s = jstring_utf8(env, piece);
        env->CallVoidMethod(listener, onToken, s);
        env->DeleteLocalRef(s);
    };
    try {
        auto turns = buildTurns(env, roles, texts, imageTurn, imageIds, imageW, imageH, imageRgb);
        GenerationStats st = engine(handle)->respond(jstr(env, systemPrompt), turns,
            sampling(temperature, topK, topP, minP, repeatPenalty, repeatLastN, maxTokens, seed), continuing, cb);
        jclass cls = env->FindClass("com/rishikesh/edgechat/engine/NativeStats");
        jmethodID ctor = env->GetMethodID(cls, "<init>", "(IIIDDIII)V");
        return env->NewObject(cls, ctor, (jint) st.promptTokens, (jint) st.cachedTokens, (jint) st.generatedTokens,
                              (jdouble) st.prefillSeconds, (jdouble) st.decodeSeconds, (jint) st.stopReason,
                              (jint) st.contextShifts, (jint) st.contextLength);
    } catch (const EngineError & e) { throwEngineError(env, e); return nullptr; }
}

JNIEXPORT jobjectArray JNICALL Java_com_rishikesh_edgechat_engine_NativeEngine_embed(JNIEnv * env, jobject, jlong handle, jobjectArray texts, jint maxTokens) {
    std::vector<std::string> in;
    const jsize n = env->GetArrayLength(texts);
    for (jsize i = 0; i < n; i++) { jstring s = (jstring) env->GetObjectArrayElement(texts, i); in.push_back(jstr(env, s)); env->DeleteLocalRef(s); }
    try {
        auto vectors = engine(handle)->embed(in, maxTokens);
        jclass fa = env->FindClass("[F");
        jobjectArray out = env->NewObjectArray((jsize) vectors.size(), fa, nullptr);
        for (size_t i = 0; i < vectors.size(); i++) {
            jfloatArray v = env->NewFloatArray((jsize) vectors[i].size());
            env->SetFloatArrayRegion(v, 0, (jsize) vectors[i].size(), vectors[i].data());
            env->SetObjectArrayElement(out, (jsize) i, v);
            env->DeleteLocalRef(v);
        }
        return out;
    } catch (const EngineError & e) { throwEngineError(env, e); return nullptr; }
}

JNIEXPORT jlong JNICALL Java_com_rishikesh_edgechat_engine_NativeEngine_saveState(JNIEnv * env, jobject, jlong handle, jstring path) {
    try { return (jlong) engine(handle)->saveState(jstr(env, path)); } catch (const EngineError & e) { throwEngineError(env, e); return 0; }
}

JNIEXPORT jboolean JNICALL Java_com_rishikesh_edgechat_engine_NativeEngine_loadState(JNIEnv * env, jobject, jlong handle, jstring path) {
    return (jboolean) engine(handle)->loadState(jstr(env, path));
}

} // extern "C"
