package com.rishikesh.edgechat.data

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.pdf.PdfRenderer
import android.net.Uri
import android.provider.OpenableColumns
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import com.rishikesh.edgechat.engine.ImageInput
import com.rishikesh.edgechat.model.AppSettings
import com.rishikesh.edgechat.model.Attachment
import com.rishikesh.edgechat.model.AttachmentKind
import com.rishikesh.edgechat.model.Message
import com.rishikesh.edgechat.model.TextUtils
import com.tom_roush.pdfbox.pdmodel.PDDocument
import com.tom_roush.pdfbox.text.PDFTextStripper
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withContext
import java.io.ByteArrayOutputStream
import java.io.File
import java.util.UUID
import kotlin.coroutines.resume

/** Something the user picked but has not sent yet. */
data class PendingAttachment(val uri: Uri, val displayName: String, val mimeType: String?)

/** Turns picked files into stored [Attachment]s (PDF/text extraction, OCR, image downscaling) and engine inputs. */
object AttachmentIngest {
    private const val MAX_DOC_CHARS = 60_000
    private val textExtensions = setOf("txt", "md", "markdown", "csv", "json", "swift", "kt", "java", "py", "js", "ts", "c", "cpp", "h", "rs", "go", "rb", "sh", "yaml", "yml", "toml", "xml", "html", "htm", "rtf", "log")

    data class Result(val attachments: List<Attachment>, val errors: List<String>)

    suspend fun ingest(context: Context, pending: List<PendingAttachment>, dir: File, settings: AppSettings, modelHasVision: Boolean): Result =
        withContext(Dispatchers.IO) {
            val out = mutableListOf<Attachment>()
            val errors = mutableListOf<String>()
            for (p in pending) {
                runCatching { ingestOne(context, p, dir, settings, modelHasVision) }
                    .onSuccess { out += it }
                    .onFailure { errors += "${p.displayName}: ${it.message ?: "could not be read"}" }
            }
            Result(out, errors)
        }

    private suspend fun ingestOne(context: Context, p: PendingAttachment, dir: File, settings: AppSettings, modelHasVision: Boolean): Attachment {
        val bytes = context.contentResolver.openInputStream(p.uri)?.use { it.readBytes() } ?: error("cannot open")
        val ext = p.displayName.substringAfterLast('.', "").lowercase()
        val isImage = (p.mimeType?.startsWith("image/") == true) || ext in setOf("jpg", "jpeg", "png", "heic", "webp", "gif", "bmp")
        val id = UUID.randomUUID().toString().uppercase()
        if (isImage) {
            val bmp = BitmapFactory.decodeByteArray(bytes, 0, bytes.size) ?: error("unsupported image")
            val scaled = downscale(bmp, 1280)
            val stored = "$id.jpg"
            File(dir, stored).outputStream().use { scaled.compress(Bitmap.CompressFormat.JPEG, 88, it) }
            var text: String? = null
            if (!modelHasVision || settings.ocrForImages) {
                val ocr = ocr(scaled)
                text = "[Image \"${p.displayName}\", ${scaled.width}x${scaled.height}]" + if (ocr.isNotBlank()) " Text in image:\n$ocr" else " (no readable text)"
            }
            return Attachment(id = id, kind = AttachmentKind.IMAGE, fileName = p.displayName, storedFileName = stored, byteCount = File(dir, stored).length(),
                extractedText = text, imageWidth = scaled.width, imageHeight = scaled.height, ingestion = if (modelHasVision) "vision" else "ocr")
        }
        val stored = "$id.$ext".trimEnd('.')
        File(dir, stored).writeBytes(bytes)
        var pages: Int? = null
        var text: String = when {
            ext == "pdf" || p.mimeType == "application/pdf" -> {
                val (t, n) = pdfText(context, File(dir, stored)); pages = n; t
            }
            ext in textExtensions || p.mimeType?.startsWith("text/") == true -> String(bytes, Charsets.UTF_8)
            else -> error("unsupported file type")
        }
        if (ext == "html" || ext == "htm") text = text.replace(Regex("<[^>]+>"), " ")
        val (trimmed, truncated) = TextUtils.truncateMiddle(text.trim(), MAX_DOC_CHARS)
        return Attachment(id = id, kind = AttachmentKind.DOCUMENT, fileName = p.displayName, storedFileName = stored, byteCount = bytes.size.toLong(),
            extractedText = trimmed, textTruncated = truncated, pageCount = pages, ingestion = "text")
    }

    /** PDFBox text layer; pages without text fall back to rendering + OCR (scanned PDFs). */
    private suspend fun pdfText(context: Context, file: File): Pair<String, Int> {
        val sb = StringBuilder()
        var pageCount = 0
        runCatching {
            PDDocument.load(file).use { doc ->
                pageCount = doc.numberOfPages
                val stripper = PDFTextStripper()
                for (i in 1..minOf(pageCount, 200)) {
                    stripper.startPage = i; stripper.endPage = i
                    val t = stripper.getText(doc).trim()
                    if (t.isNotEmpty()) sb.append("\n\n[page $i]\n").append(t)
                }
            }
        }
        if (sb.length < 40 * maxOf(1, pageCount)) {
            // Little or no text layer: OCR the rendered pages.
            runCatching {
                android.os.ParcelFileDescriptor.open(file, android.os.ParcelFileDescriptor.MODE_READ_ONLY).use { pfd ->
                    PdfRenderer(pfd).use { renderer ->
                        pageCount = renderer.pageCount
                        for (i in 0 until minOf(renderer.pageCount, 30)) {
                            renderer.openPage(i).use { page ->
                                val scale = 1400f / maxOf(page.width, page.height)
                                val bmp = Bitmap.createBitmap((page.width * scale).toInt(), (page.height * scale).toInt(), Bitmap.Config.ARGB_8888)
                                bmp.eraseColor(android.graphics.Color.WHITE)
                                page.render(bmp, null, null, PdfRenderer.Page.RENDER_MODE_FOR_DISPLAY)
                                val t = ocr(bmp)
                                if (t.isNotBlank()) sb.append("\n\n[page ${i + 1} (OCR)]\n").append(t)
                            }
                        }
                    }
                }
            }
        }
        return sb.toString().trim() to pageCount
    }

    private suspend fun ocr(bitmap: Bitmap): String = suspendCancellableCoroutine { cont ->
        val recognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)
        recognizer.process(InputImage.fromBitmap(bitmap, 0))
            .addOnSuccessListener { cont.resume(it.text) }
            .addOnFailureListener { cont.resume("") }
    }

    private fun downscale(bmp: Bitmap, maxDim: Int): Bitmap {
        val m = maxOf(bmp.width, bmp.height)
        if (m <= maxDim) return bmp
        val s = maxDim.toFloat() / m
        return Bitmap.createScaledBitmap(bmp, (bmp.width * s).toInt().coerceAtLeast(1), (bmp.height * s).toInt().coerceAtLeast(1), true)
    }

    /** RGB888 input for the vision encoder from a stored JPEG. */
    fun makeImageInput(file: File, id: String, maxDimension: Int): ImageInput? {
        val bmp = BitmapFactory.decodeFile(file.path) ?: return null
        val scaled = downscale(bmp, maxDimension)
        val w = scaled.width; val h = scaled.height
        val pixels = IntArray(w * h)
        scaled.getPixels(pixels, 0, w, 0, 0, w, h)
        val rgb = ByteArray(w * h * 3)
        var j = 0
        for (p in pixels) { rgb[j++] = (p shr 16 and 0xff).toByte(); rgb[j++] = (p shr 8 and 0xff).toByte(); rgb[j++] = (p and 0xff).toByte() }
        return ImageInput(id, w, h, rgb)
    }

    /** The text the engine sees for a message: recalled context, document blocks, OCR (text-only models), the message. */
    fun engineText(message: Message, modelHasVision: Boolean): String {
        val parts = mutableListOf<String>()
        message.recalledContext?.takeIf { it.isNotEmpty() }?.let { parts += it }
        for (a in message.attachments) when (a.kind) {
            AttachmentKind.DOCUMENT -> {
                val pages = a.pageCount?.let { " pages=\"$it\"" } ?: ""
                parts += "<document name=\"${a.fileName}\"$pages>\n${a.extractedText ?: "(no text)"}\n</document>"
            }
            AttachmentKind.IMAGE -> if (!modelHasVision) a.extractedText?.let { parts += it }
        }
        parts += message.content
        return parts.filter { it.isNotEmpty() }.joinToString("\n\n")
    }

    fun imageInputs(message: Message, dir: File, settings: AppSettings): List<ImageInput> =
        message.attachments.filter { it.kind == AttachmentKind.IMAGE }.mapNotNull { a -> makeImageInput(File(dir, a.storedFileName), a.id, settings.maxImageDimension) }

    fun displayName(context: Context, uri: Uri): String {
        context.contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use { c ->
            if (c.moveToFirst()) c.getString(0)?.let { return it }
        }
        return uri.lastPathSegment?.substringAfterLast('/') ?: "file"
    }

    fun jpegBytes(bitmap: Bitmap): ByteArray = ByteArrayOutputStream().also { bitmap.compress(Bitmap.CompressFormat.JPEG, 88, it) }.toByteArray()
}
