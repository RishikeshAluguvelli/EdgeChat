package com.rishikesh.edgechat.ui

import android.net.Uri
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.ArrowUpward
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Description
import androidx.compose.material.icons.filled.Image
import androidx.compose.material.icons.filled.Memory
import androidx.compose.material.icons.filled.MoreVert
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.AssistChip
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilledIconButton
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import coil.compose.AsyncImage
import com.rishikesh.edgechat.AppModel
import com.rishikesh.edgechat.data.AttachmentIngest
import com.rishikesh.edgechat.data.PendingAttachment
import com.rishikesh.edgechat.engine.EngineController
import com.rishikesh.edgechat.model.AttachmentKind
import com.rishikesh.edgechat.model.GenerationStats
import com.rishikesh.edgechat.model.Message
import com.rishikesh.edgechat.model.Role
import com.rishikesh.edgechat.model.StopReason
import com.rishikesh.edgechat.model.TextUtils
import kotlinx.coroutines.launch
import java.util.Locale

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ChatScreen(app: AppModel, conversationID: String, onBack: () -> Unit) {
    val conversations by app.store.conversations.collectAsState()
    val c = conversations.firstOrNull { it.id == conversationID } ?: return
    val isGenerating by app.engine.isGenerating.collectAsState()
    val generatingID by app.engine.generatingConversationID.collectAsState()
    val activity by app.engine.activity.collectAsState()
    val compacting by app.engine.isCompacting.collectAsState()
    val engineState by app.engine.state.collectAsState()
    val settings by app.settings.collectAsState()
    val lastError by app.engine.lastError.collectAsState()
    val scope = rememberCoroutineScope()
    val listState = rememberLazyListState()
    var menu by remember { mutableStateOf(false) }
    var showContext by remember { mutableStateOf(false) }

    val usage = c.messages.lastOrNull { it.role == Role.ASSISTANT && it.stats?.contextLength != null }?.stats?.let { st ->
        val total = st.contextLength!!; minOf(total, st.promptTokens + st.generatedTokens) to total
    }

    LaunchedEffect(c.messages.size, c.messages.lastOrNull()?.content?.length) {
        if (c.messages.isNotEmpty()) listState.animateScrollToItem(c.messages.size)
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(c.title, maxLines = 1, style = MaterialTheme.typography.titleMedium) },
                navigationIcon = { IconButton(onClick = onBack) { Icon(Icons.AutoMirrored.Filled.ArrowBack, "Back") } },
                actions = {
                    usage?.let { (used, total) ->
                        val f = used.toFloat() / total
                        IconButton(onClick = { showContext = true }) {
                            Box(contentAlignment = Alignment.Center) {
                                CircularProgressIndicator(progress = { f }, modifier = Modifier.size(24.dp), strokeWidth = 2.5.dp,
                                    color = if (f >= 0.9f) MaterialTheme.colorScheme.error else if (f >= 0.7f) Color(0xFFE08A00) else MaterialTheme.colorScheme.onSurfaceVariant)
                                Text("${(f * 100).toInt()}", style = MaterialTheme.typography.labelSmall.copy(fontSize = androidx.compose.ui.unit.TextUnit(7f, androidx.compose.ui.unit.TextUnitType.Sp)))
                            }
                        }
                    }
                    IconButton(onClick = { menu = true }) { Icon(Icons.Default.MoreVert, "More") }
                    DropdownMenu(expanded = menu, onDismissRequest = { menu = false }) {
                        DropdownMenuItem(text = { Text("Clear KV cache") }, leadingIcon = { Icon(Icons.Default.Memory, null) }, onClick = { menu = false; scope.launch { app.engine.clearCache() } })
                        DropdownMenuItem(text = { Text("Delete chat") }, leadingIcon = { Icon(Icons.Default.Delete, null) }, onClick = { menu = false; app.deleteConversation(conversationID) })
                    }
                },
            )
        },
        bottomBar = {
            Column(Modifier.imePadding()) {
                if (activity == EngineController.Activity.RETRIEVING || activity == EngineController.Activity.SUMMARIZING || activity == EngineController.Activity.SAVING) {
                    Row(Modifier.padding(horizontal = 14.dp, vertical = 6.dp), verticalAlignment = Alignment.CenterVertically) {
                        CircularProgressIndicator(Modifier.size(14.dp), strokeWidth = 2.dp)
                        Spacer(Modifier.width(8.dp))
                        Text(when (activity) {
                            EngineController.Activity.SUMMARIZING -> if (compacting) "Compacting older messages into memory…" else "Summarizing earlier conversation…"
                            EngineController.Activity.RETRIEVING -> "Recalling relevant context…"
                            else -> "Saving conversation cache…"
                        }, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                    }
                }
                Composer(app, conversationID, enabled = engineState is EngineController.State.Ready, generatingHere = isGenerating && generatingID == conversationID)
            }
        },
    ) { padding ->
        LazyColumn(Modifier.fillMaxSize().padding(padding), state = listState, contentPadding = androidx.compose.foundation.layout.PaddingValues(12.dp), verticalArrangement = Arrangement.spacedBy(14.dp)) {
            if (engineState !is EngineController.State.Ready) item {
                Row(Modifier.fillMaxWidth().background(MaterialTheme.colorScheme.surfaceVariant, RoundedCornerShape(12.dp)).padding(12.dp), verticalAlignment = Alignment.CenterVertically) {
                    Text(when (val s = engineState) {
                        is EngineController.State.Loading -> "Loading ${s.name}…"
                        is EngineController.State.Failed -> s.message
                        else -> "Download or pick a model to start chatting."
                    }, modifier = Modifier.weight(1f), style = MaterialTheme.typography.bodyMedium)
                    TextButton(onClick = { app.showModels.value = true }) { Text("Models") }
                }
            }
            itemsIndexed(c.messages, key = { _, m -> m.id }) { i, m ->
                if (i == c.summaryCoversMessages && c.summary != null && c.summaryCoversMessages > 0) SummaryDivider(c.summaryCoversMessages, c.summary)
                MessageRow(app, conversationID, m,
                    isStreaming = isGenerating && generatingID == conversationID && m.id == c.messages.last().id,
                    isLast = m.id == c.messages.last().id, showStats = settings.showStats)
            }
            item { Spacer(Modifier.height(4.dp)) }
        }
    }

    if (showContext) AlertDialog(onDismissRequest = { showContext = false }, confirmButton = { TextButton(onClick = { showContext = false }) { Text("OK") } },
        title = { Text("Context window") },
        text = { usage?.let { (u, t) -> Text("${u} of ${t} tokens (${u * 100 / t}%) were in use after the last reply, including any photos. When usage passes about 75%, the oldest messages are folded into a memory summary automatically${if (settings.autoCompact) " right after the reply" else " before the next reply"}, and relevant older messages are recalled per turn.") } })
    lastError?.let { err ->
        AlertDialog(onDismissRequest = { app.engine.clearError() }, confirmButton = { TextButton(onClick = { app.engine.clearError() }) { Text("OK") } }, title = { Text("Attachment problem") }, text = { Text(err) })
    }
}

@Composable
private fun SummaryDivider(count: Int, summary: String?) {
    var show by remember { mutableStateOf(false) }
    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.Center) {
        AssistChip(onClick = { show = true }, label = { Text("$count earlier messages summarized") })
    }
    if (show) AlertDialog(onDismissRequest = { show = false }, confirmButton = { TextButton(onClick = { show = false }) { Text("Done") } },
        title = { Text("Conversation memory") }, text = { Text(summary ?: "", style = MaterialTheme.typography.bodySmall) })
}

@Composable
fun MessageRow(app: AppModel, conversationID: String, m: Message, isStreaming: Boolean, isLast: Boolean, showStats: Boolean) {
    val scope = rememberCoroutineScope()
    val dir = remember(conversationID) { app.store.attachmentsDirectory(conversationID) }
    Row(Modifier.fillMaxWidth(), horizontalArrangement = if (m.role == Role.USER) Arrangement.End else Arrangement.Start) {
        Column(Modifier.widthIn(max = if (m.role == Role.USER) 320.dp else 1000.dp), horizontalAlignment = if (m.role == Role.USER) Alignment.End else Alignment.Start, verticalArrangement = Arrangement.spacedBy(6.dp)) {
            if (m.attachments.isNotEmpty()) Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                for (a in m.attachments) {
                    if (a.kind == AttachmentKind.IMAGE) AsyncImage(model = java.io.File(dir, a.storedFileName), contentDescription = a.fileName, modifier = Modifier.size(96.dp).background(MaterialTheme.colorScheme.surfaceVariant, RoundedCornerShape(10.dp)))
                    else AssistChip(onClick = {}, label = { Text(a.fileName, maxLines = 1) }, leadingIcon = { Icon(Icons.Default.Description, null) })
                }
            }
            if (m.role == Role.USER) {
                m.recalledContext?.let { RecalledChip(it) }
                if (m.content.isNotEmpty()) Text(m.content, color = Color.White,
                    modifier = Modifier.background(MaterialTheme.colorScheme.primary, RoundedCornerShape(18.dp)).padding(horizontal = 14.dp, vertical = 10.dp))
            } else {
                m.reasoning?.takeIf { it.isNotEmpty() }?.let { r ->
                    var open by remember { mutableStateOf(false) }
                    TextButton(onClick = { open = !open }) { Text(if (isStreaming && m.content.isEmpty()) "Thinking…" else "Thought process") }
                    if (open) Text(r, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                }
                if (m.content.isEmpty() && isStreaming) Text("…", color = MaterialTheme.colorScheme.onSurfaceVariant)
                else if (m.content.isNotEmpty()) MarkdownText(m.content)
                m.error?.let { Text(it, color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodySmall) }
                if (showStats && !isStreaming) m.stats?.let { Text(statsLine(it), style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.outline) }
                if (isLast && !isStreaming) Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    if (app.engine.canContinue(conversationID)) OutlinedButton(onClick = { scope.launch { app.engine.continueReply(conversationID) } }) { Text("Continue") }
                    TextButton(onClick = { scope.launch { app.engine.regenerate(conversationID) } }) { Icon(Icons.Default.Refresh, null, Modifier.size(16.dp)); Spacer(Modifier.width(4.dp)); Text("Regenerate") }
                }
            }
        }
    }
}

private fun statsLine(s: GenerationStats): String {
    val parts = mutableListOf<String>()
    if (s.generatedTokens > 0) parts += String.format(Locale.US, "%.1f tok/s", s.tokensPerSecond)
    parts += "${s.generatedTokens} tokens"
    if (s.promptTokens > 0) parts += "prompt ${s.promptTokens} (${s.cachedTokens} cached)"
    if (s.prefillSeconds > 0.05) parts += String.format(Locale.US, "prefill %.1fs", s.prefillSeconds)
    s.contextShifts?.takeIf { it > 0 }?.let { parts += "context shifted ×$it" }
    s.contextLength?.let { parts += "ctx ${TextUtils.formatContext(it)}" }
    when (s.stopReason) {
        StopReason.CANCELLED -> parts += "stopped"
        StopReason.MAX_TOKENS -> parts += "reply length limit reached"
        StopReason.CONTEXT_FULL -> parts += "paused: context window full"
        StopReason.EOS -> {}
    }
    return parts.joinToString(" · ")
}

@Composable
private fun RecalledChip(text: String) {
    var show by remember { mutableStateOf(false) }
    val count = text.split("\n- ").size - 1
    TextButton(onClick = { show = true }) { Text("Recalled ${maxOf(1, count)} earlier snippet${if (count == 1) "" else "s"}", style = MaterialTheme.typography.labelSmall) }
    if (show) AlertDialog(onDismissRequest = { show = false }, confirmButton = { TextButton(onClick = { show = false }) { Text("Done") } }, title = { Text("Recalled context") }, text = { Text(text, style = MaterialTheme.typography.bodySmall) })
}

@Composable
private fun Composer(app: AppModel, conversationID: String, enabled: Boolean, generatingHere: Boolean) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    var text by remember { mutableStateOf("") }
    var pending by remember { mutableStateOf(listOf<PendingAttachment>()) }
    var pickMenu by remember { mutableStateOf(false) }
    val photoPicker = rememberLauncherForActivityResult(ActivityResultContracts.PickMultipleVisualMedia(4)) { uris: List<Uri> ->
        pending = pending + uris.map { PendingAttachment(it, AttachmentIngest.displayName(context, it), context.contentResolver.getType(it)) }
    }
    val filePicker = rememberLauncherForActivityResult(ActivityResultContracts.OpenMultipleDocuments()) { uris: List<Uri> ->
        pending = pending + uris.map { PendingAttachment(it, AttachmentIngest.displayName(context, it), context.contentResolver.getType(it)) }
    }
    val canSend = enabled && !generatingHere && (text.isNotBlank() || pending.isNotEmpty())
    Column(Modifier.fillMaxWidth().padding(horizontal = 10.dp, vertical = 8.dp)) {
        if (pending.isNotEmpty()) Row(Modifier.padding(bottom = 6.dp), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            for (p in pending) AssistChip(onClick = { pending = pending - p }, label = { Text(p.displayName, maxLines = 1) },
                leadingIcon = { Icon(if (p.mimeType?.startsWith("image/") == true) Icons.Default.Image else Icons.Default.Description, null) },
                trailingIcon = { Icon(Icons.Default.Close, "Remove") })
        }
        Row(verticalAlignment = Alignment.Bottom) {
            Box {
                IconButton(onClick = { pickMenu = true }, enabled = enabled) { Icon(Icons.Default.Add, "Attach") }
                DropdownMenu(expanded = pickMenu, onDismissRequest = { pickMenu = false }) {
                    DropdownMenuItem(text = { Text("Photo") }, leadingIcon = { Icon(Icons.Default.Image, null) }, onClick = { pickMenu = false; photoPicker.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)) })
                    DropdownMenuItem(text = { Text("Document") }, leadingIcon = { Icon(Icons.Default.Description, null) }, onClick = { pickMenu = false; filePicker.launch(arrayOf("application/pdf", "text/*", "application/json")) })
                }
            }
            OutlinedTextField(value = text, onValueChange = { text = it }, modifier = Modifier.weight(1f), placeholder = { Text(if (enabled) "Message (offline)" else "Load a model first") }, maxLines = 6, shape = RoundedCornerShape(22.dp))
            Spacer(Modifier.width(6.dp))
            if (generatingHere) FilledIconButton(onClick = { app.engine.stop() }, shape = CircleShape) { Icon(Icons.Default.Stop, "Stop") }
            else FilledIconButton(onClick = {
                val t = text.trim(); val items = pending
                text = ""; pending = emptyList()
                scope.launch { app.engine.send(conversationID, t, items) }
            }, enabled = canSend, shape = CircleShape) { Icon(Icons.Default.ArrowUpward, "Send") }
        }
    }
}
