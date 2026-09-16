package com.rishikesh.edgechat.data

import android.content.Context
import com.rishikesh.edgechat.model.Attachment
import com.rishikesh.edgechat.model.Conversation
import com.rishikesh.edgechat.model.Iso
import com.rishikesh.edgechat.model.Role
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update
import kotlinx.serialization.json.Json
import java.io.File
import java.util.concurrent.Executors

/** JSON-on-disk conversation persistence: files/Conversations/<id>.json (+ <id>/ for attachments). Main-thread API. */
class ConversationStore(context: Context) {
    val root: File = File(context.filesDir, "Conversations").apply { mkdirs() }
    private val _conversations = MutableStateFlow<List<Conversation>>(emptyList())
    val conversations: StateFlow<List<Conversation>> = _conversations
    private val json = Json { ignoreUnknownKeys = true; encodeDefaults = true; prettyPrint = false }
    /** Serial so writes land in the order they were requested. */
    private val saveExecutor = Executors.newSingleThreadExecutor { r -> Thread(r, "edgechat-save") }

    fun loadAll() {
        val files = root.listFiles { f -> f.extension == "json" } ?: emptyArray()
        val list = files.mapNotNull { f -> runCatching { json.decodeFromString<Conversation>(f.readText()) }.getOrNull() }
            .map { c ->
                // A reply the app was killed in the middle of leaves an empty bubble; drop it.
                c.copy(messages = c.messages.filterNot { it.role == Role.ASSISTANT && it.content.isEmpty() && it.error == null && it.stats == null })
            }
            .sortedByDescending { Iso.parse(it.updatedAt) }
        _conversations.value = list
    }

    fun conversation(id: String): Conversation? = _conversations.value.firstOrNull { it.id == id }

    fun create(): Conversation {
        val c = Conversation()
        _conversations.update { listOf(c) + it }
        save(c)
        return c
    }

    /** `persist = false` updates memory only (streaming token flushes); call again with `persist = true` to write. */
    fun update(conversation: Conversation, touch: Boolean = true, persist: Boolean = true) {
        val c = if (touch) conversation.copy(updatedAt = Iso.now()) else conversation
        _conversations.update { list ->
            val i = list.indexOfFirst { it.id == c.id }
            val next = if (i >= 0) list.toMutableList().also { it[i] = c } else (listOf(c) + list).toMutableList()
            if (touch) next.sortedByDescending { Iso.parse(it.updatedAt) } else next
        }
        if (persist) save(c)
    }

    fun modify(id: String, touch: Boolean = true, persist: Boolean = true, body: (Conversation) -> Conversation) {
        val c = conversation(id) ?: return
        update(body(c), touch, persist)
    }

    fun delete(id: String) {
        _conversations.update { list -> list.filterNot { it.id == id } }
        fileFor(id).delete()
        attachmentsDirectory(id).deleteRecursively()
    }

    fun attachmentsDirectory(id: String): File = File(root, id).apply { mkdirs() }
    fun memoryIndexFile(id: String): File = File(attachmentsDirectory(id), "memory-index.json")
    fun attachmentFile(conversationID: String, attachment: Attachment): File = File(attachmentsDirectory(conversationID), attachment.storedFileName)

    private fun fileFor(id: String) = File(root, "$id.json")

    private fun save(c: Conversation) {
        val target = fileFor(c.id)
        saveExecutor.execute {
            runCatching {
                val tmp = File(target.parentFile, "${target.name}.tmp")
                tmp.writeText(json.encodeToString(Conversation.serializer(), c))
                if (!tmp.renameTo(target)) { target.delete(); tmp.renameTo(target) }
            }
        }
    }
}
