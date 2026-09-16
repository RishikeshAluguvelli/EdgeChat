package com.rishikesh.edgechat.data

import android.app.DownloadManager
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.database.Cursor
import android.net.Uri
import android.os.Build
import android.os.StatFs
import com.rishikesh.edgechat.model.ModelCatalog
import com.rishikesh.edgechat.model.ModelSpec
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import java.io.File

/** A model on disk (catalog download or imported GGUF), loadable by the engine. */
data class InstalledModel(
    /** Catalog id, or "file:<name>" for imports. */
    val id: String,
    val name: String,
    val modelFile: File,
    val mmprojFile: File?,
    val spec: ModelSpec?,
) {
    val sizeBytes: Long get() = modelFile.length() + (mmprojFile?.length() ?: 0)
    val hasVision: Boolean get() = mmprojFile != null
}

data class DownloadState(val id: String, val bytes: Long, val total: Long, val status: Int, val reason: Int) {
    val fraction: Float get() = if (total > 0) (bytes.toFloat() / total).coerceIn(0f, 1f) else 0f
    val isActive: Boolean get() = status == DownloadManager.STATUS_RUNNING || status == DownloadManager.STATUS_PENDING || status == DownloadManager.STATUS_PAUSED
}

/**
 * Models live in files/Models. Catalog downloads go through the system DownloadManager (background, resumable,
 * survives the app being killed) into files/Models/<file>.part and are renamed on completion.
 */
class ModelManager(private val context: Context) {
    /**
     * App-specific external storage (no permission needed, removed on uninstall). DownloadManager refuses to write
     * into internal storage ("Unsupported path"), so downloads and imports both live here.
     */
    val modelsDir: File = (context.getExternalFilesDir(MODELS_DIR) ?: File(context.filesDir, MODELS_DIR)).apply { mkdirs() }
    private val dm = context.getSystemService(Context.DOWNLOAD_SERVICE) as DownloadManager
    private val prefs = context.getSharedPreferences("edgechat.downloads", Context.MODE_PRIVATE)

    private val _installed = MutableStateFlow<List<InstalledModel>>(emptyList())
    val installed: StateFlow<List<InstalledModel>> = _installed
    private val _downloads = MutableStateFlow<Map<String, DownloadState>>(emptyMap())
    /** Keyed by model id; a vision model's projector download is folded into the same entry. */
    val downloads: StateFlow<Map<String, DownloadState>> = _downloads

    private val receiver = object : BroadcastReceiver() {
        override fun onReceive(c: Context?, intent: Intent?) { refresh() }
    }

    init {
        val filter = IntentFilter(DownloadManager.ACTION_DOWNLOAD_COMPLETE)
        if (Build.VERSION.SDK_INT >= 33) context.registerReceiver(receiver, filter, Context.RECEIVER_EXPORTED)
        else context.registerReceiver(receiver, filter)
        refresh()
    }

    fun refresh() {
        finalizeCompletedDownloads()
        val files = modelsDir.listFiles()?.filter { it.extension == "gguf" } ?: emptyList()
        val byName = files.associateBy { it.name }
        val list = mutableListOf<InstalledModel>()
        val claimed = mutableSetOf<String>()
        for (spec in ModelCatalog.models) {
            val model = byName[spec.fileName] ?: continue
            val mm = spec.mmprojFileName?.let { byName[it] }
            if (spec.mmprojURL != null && mm == null) continue   // projector still downloading
            claimed += model.name; mm?.let { claimed += it.name }
            list += InstalledModel(spec.id, spec.name, model, mm, spec)
        }
        for (f in files) if (f.name !in claimed && !f.name.startsWith("mmproj") && !f.name.contains("-mmproj-")) {
            list += InstalledModel("file:${f.name}", f.nameWithoutExtension, f, null, null)
        }
        _installed.value = list
        pollDownloads()
    }

    fun activeInstalledModel(activeID: String?): InstalledModel? = _installed.value.firstOrNull { it.id == activeID }

    /** Returns null on success or an error message (DownloadManager disabled, storage unavailable). */
    fun download(spec: ModelSpec): String? {
        if (_downloads.value[spec.id]?.isActive == true) return null
        return try {
            val ids = mutableListOf<Long>()
            ids += enqueue(spec.url, spec.fileName, "${spec.name} weights")
            if (spec.mmprojURL != null && spec.mmprojFileName != null) ids += enqueue(spec.mmprojURL, spec.mmprojFileName!!, "${spec.name} vision projector")
            prefs.edit().putString(spec.id, ids.joinToString(",")).apply()
            pollDownloads()
            null
        } catch (e: Exception) {
            DiagnosticsLog.append("[download-error] ${spec.id}: ${e.message}")
            e.message ?: "Could not start the download."
        }
    }

    private fun enqueue(url: String, fileName: String, title: String): Long {
        File(modelsDir, "$fileName.part").delete()
        val req = DownloadManager.Request(Uri.parse(url))
            .setTitle(title)
            .setDescription("EdgeChat model download")
            .setDestinationInExternalFilesDir(context, MODELS_DIR, "$fileName.part")
            .setNotificationVisibility(DownloadManager.Request.VISIBILITY_VISIBLE)
            .setAllowedOverMetered(true)
            .setAllowedOverRoaming(false)
        return dm.enqueue(req)
    }

    fun cancelDownload(specID: String) {
        val ids = prefs.getString(specID, null)?.split(",")?.mapNotNull { it.toLongOrNull() } ?: emptyList()
        if (ids.isNotEmpty()) dm.remove(*ids.toLongArray())
        prefs.edit().remove(specID).apply()
        pollDownloads()
    }

    fun delete(model: InstalledModel) {
        model.modelFile.delete()
        model.mmprojFile?.delete()
        refresh()
    }

    /** Copies a user-picked GGUF into the models folder (Files/"Open with"). */
    fun importModel(uri: Uri, displayName: String): InstalledModel? {
        val name = displayName.takeIf { it.endsWith(".gguf") } ?: "$displayName.gguf"
        val dest = File(modelsDir, name)
        runCatching {
            context.contentResolver.openInputStream(uri)?.use { input -> dest.outputStream().use { input.copyTo(it) } }
        }.onFailure { dest.delete(); return null }
        refresh()
        return _installed.value.firstOrNull { it.modelFile == dest }
    }

    /** Polls the DownloadManager for every tracked download and folds them per model id. */
    fun pollDownloads() {
        val out = HashMap<String, DownloadState>()
        for ((specID, raw) in prefs.all) {
            val ids = (raw as? String)?.split(",")?.mapNotNull { it.toLongOrNull() } ?: continue
            var bytes = 0L; var total = 0L; var status = DownloadManager.STATUS_SUCCESSFUL; var reason = 0; var any = false
            query(ids) { c ->
                any = true
                bytes += c.getLong(c.getColumnIndexOrThrow(DownloadManager.COLUMN_BYTES_DOWNLOADED_SO_FAR))
                total += c.getLong(c.getColumnIndexOrThrow(DownloadManager.COLUMN_TOTAL_SIZE_BYTES)).coerceAtLeast(0)
                val s = c.getInt(c.getColumnIndexOrThrow(DownloadManager.COLUMN_STATUS))
                if (s != DownloadManager.STATUS_SUCCESSFUL) { status = s; reason = c.getInt(c.getColumnIndexOrThrow(DownloadManager.COLUMN_REASON)) }
            }
            if (!any) { prefs.edit().remove(specID).apply(); continue }
            if (status == DownloadManager.STATUS_SUCCESSFUL) { prefs.edit().remove(specID).apply(); continue }
            out[specID] = DownloadState(specID, bytes, total, status, reason)
        }
        _downloads.value = out
    }

    private fun finalizeCompletedDownloads() {
        for ((specID, raw) in prefs.all) {
            val ids = (raw as? String)?.split(",")?.mapNotNull { it.toLongOrNull() } ?: continue
            var allDone = true
            query(ids) { c ->
                val s = c.getInt(c.getColumnIndexOrThrow(DownloadManager.COLUMN_STATUS))
                if (s == DownloadManager.STATUS_SUCCESSFUL) {
                    val local = c.getString(c.getColumnIndexOrThrow(DownloadManager.COLUMN_LOCAL_URI)) ?: return@query
                    val part = File(Uri.parse(local).path ?: return@query)
                    if (part.exists() && part.name.endsWith(".part")) {
                        val final = File(part.parentFile, part.name.removeSuffix(".part"))
                        part.renameTo(final)
                    }
                } else if (s == DownloadManager.STATUS_FAILED) {
                    allDone = allDone && true
                } else allDone = false
            }
            if (allDone) prefs.edit().remove(specID).apply()
            @Suppress("UNUSED_VARIABLE") val unused = specID
        }
    }

    private inline fun query(ids: List<Long>, block: (Cursor) -> Unit) {
        if (ids.isEmpty()) return
        val c = dm.query(DownloadManager.Query().setFilterById(*ids.toLongArray())) ?: return
        c.use { while (it.moveToNext()) block(it) }
    }

    fun freeBytes(): Long = StatFs(modelsDir.path).availableBytes

    companion object { const val MODELS_DIR = "Models" }
    fun modelsOnDeviceBytes(): Long = modelsDir.listFiles()?.sumOf { it.length() } ?: 0L
}
