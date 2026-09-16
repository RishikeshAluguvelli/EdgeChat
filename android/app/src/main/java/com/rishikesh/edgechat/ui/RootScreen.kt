package com.rishikesh.edgechat.ui

import androidx.activity.compose.BackHandler
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Memory
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.Card
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.ListItem
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.rememberModalBottomSheetState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.rishikesh.edgechat.AppModel
import com.rishikesh.edgechat.engine.EngineController
import com.rishikesh.edgechat.model.TextUtils

/** Chat list, or the selected chat; Models and Settings are bottom sheets. */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun RootScreen(app: AppModel) {
    val selected by app.selectedConversationID.collectAsState()
    val conversations by app.store.conversations.collectAsState()
    val showModels by app.showModels.collectAsState()
    val showSettings by app.showSettings.collectAsState()

    val current = selected?.let { id -> conversations.firstOrNull { it.id == id } }
    if (current != null) {
        BackHandler { app.select(null) }
        ChatScreen(app, current.id, onBack = { app.select(null) })
    } else {
        ChatListScreen(app)
    }

    if (showModels) ModalBottomSheet(onDismissRequest = { app.showModels.value = false }, sheetState = rememberModalBottomSheetState(skipPartiallyExpanded = true)) {
        ModelsScreen(app)
    }
    if (showSettings) ModalBottomSheet(onDismissRequest = { app.showSettings.value = false }, sheetState = rememberModalBottomSheetState(skipPartiallyExpanded = true)) {
        SettingsScreen(app)
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun ChatListScreen(app: AppModel) {
    val conversations by app.store.conversations.collectAsState()
    val state by app.engine.state.collectAsState()
    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("EdgeChat", style = MaterialTheme.typography.headlineMedium) },
                navigationIcon = { IconButton(onClick = { app.showSettings.value = true }) { Icon(Icons.Default.Settings, "Settings") } },
                actions = { IconButton(onClick = { app.newConversation() }) { Icon(Icons.Default.Add, "New chat") } },
            )
        },
    ) { padding ->
        LazyColumn(Modifier.fillMaxSize().padding(padding), contentPadding = androidx.compose.foundation.layout.PaddingValues(16.dp)) {
            item {
                Card(Modifier.fillMaxWidth().clickable { app.showModels.value = true }) {
                    Row(Modifier.padding(14.dp), verticalAlignment = Alignment.CenterVertically) {
                        Icon(Icons.Default.Memory, null, tint = MaterialTheme.colorScheme.primary)
                        Spacer(Modifier.width(12.dp))
                        Column(Modifier.weight(1f)) {
                            when (val s = state) {
                                is EngineController.State.Ready -> {
                                    Text(s.info.name, style = MaterialTheme.typography.titleMedium)
                                    Text("${TextUtils.formatContext(s.info.contextLength)} context · ${s.info.threads} threads", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                                }
                                is EngineController.State.Loading -> {
                                    Text("Loading ${s.name}…", style = MaterialTheme.typography.titleMedium)
                                    LinearProgressIndicator(progress = { s.progress }, modifier = Modifier.fillMaxWidth().padding(top = 6.dp))
                                }
                                is EngineController.State.Failed -> {
                                    Text("Model failed to load", style = MaterialTheme.typography.titleMedium, color = MaterialTheme.colorScheme.error)
                                    Text(s.message, style = MaterialTheme.typography.bodySmall)
                                }
                                EngineController.State.Idle -> {
                                    Text("No model loaded", style = MaterialTheme.typography.titleMedium)
                                    Text("Tap to download or pick a model", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                                }
                            }
                        }
                    }
                }
                Spacer(Modifier.padding(8.dp))
                Text("CHATS", style = MaterialTheme.typography.labelMedium, color = MaterialTheme.colorScheme.onSurfaceVariant, modifier = Modifier.padding(start = 4.dp, bottom = 4.dp))
            }
            if (conversations.isEmpty()) item { Text("No chats yet. Tap + to start one.", color = MaterialTheme.colorScheme.onSurfaceVariant, modifier = Modifier.padding(8.dp)) }
            items(conversations, key = { it.id }) { c ->
                ListItem(
                    headlineContent = { Text(c.title, maxLines = 1) },
                    supportingContent = { Text("${c.messages.size} messages · ${TextUtils.relativeTime(c.updatedAt)}", style = MaterialTheme.typography.bodySmall) },
                    trailingContent = { IconButton(onClick = { app.deleteConversation(c.id) }) { Icon(Icons.Default.Delete, "Delete chat") } },
                    modifier = Modifier.clickable { app.select(c.id) },
                )
            }
        }
    }
}
