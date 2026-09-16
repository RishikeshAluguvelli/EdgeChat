package com.rishikesh.edgechat.data

import android.content.Context
import com.rishikesh.edgechat.engine.NativeEngine
import java.io.File
import java.util.concurrent.Executors

/** Appends llama warnings and per-turn stats to files/edgechat.log (Settings → Share diagnostics log). */
object DiagnosticsLog {
    private lateinit var file: File
    private val executor = Executors.newSingleThreadExecutor { r -> Thread(r, "edgechat-log") }
    private const val MAX_BYTES = 512L * 1024

    fun install(context: Context, appVersion: String, totalMemBytes: Long) {
        file = File(context.filesDir, "edgechat.log")
        append("--- launch EdgeChat $appVersion RAM ${totalMemBytes / (1024 * 1024)} MB")
    }

    val logger = object : NativeEngine.Logger {
        override fun log(level: Int, message: String) { if (level >= 3) append("[llama ${if (level >= 4) "error" else "warn"}] ${message.trimEnd()}") }
    }

    fun append(line: String) {
        if (!::file.isInitialized) return
        val stamp = java.time.format.DateTimeFormatter.ISO_INSTANT.format(java.time.Instant.now())
        executor.execute {
            runCatching {
                if (file.length() > MAX_BYTES) { val keep = file.readText().takeLast((MAX_BYTES / 2).toInt()); file.writeText(keep) }
                file.appendText("$stamp $line\n")
            }
        }
    }

    fun logFile(): File? = if (::file.isInitialized) file else null
}
