package com.rishikesh.edgechat.ui

import android.app.DownloadManager
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.AssistChip
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.rishikesh.edgechat.AppModel
import com.rishikesh.edgechat.data.InstalledModel
import com.rishikesh.edgechat.model.ModelCatalog
import com.rishikesh.edgechat.model.ModelSpec
import com.rishikesh.edgechat.model.TextUtils
import com.rishikesh.edgechat.model.Tier
import kotlinx.coroutines.delay

@Composable
fun ModelsScreen(app: AppModel) {
    val installed by app.models.installed.collectAsState()
    val downloads by app.models.downloads.collectAsState()
    val settings by app.settings.collectAsState()
    val loaded = app.engine.loadedModel
    var confirmDelete by remember { mutableStateOf<InstalledModel?>(null) }
    val importPicker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri -> uri?.let { app.importModel(it) } }

    LaunchedEffect(downloads.isNotEmpty()) {
        while (downloads.isNotEmpty()) { delay(1000); app.models.refresh() }
    }

    LazyColumn(Modifier.fillMaxWidth().padding(horizontal = 16.dp), contentPadding = androidx.compose.foundation.layout.PaddingValues(bottom = 32.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        item {
            Text("Models", style = MaterialTheme.typography.headlineMedium, modifier = Modifier.padding(vertical = 8.dp))
            Card(Modifier.fillMaxWidth()) {
                Column(Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                    KV("Device memory", TextUtils.formatBytes(app.physicalMemory))
                    KV("Free storage", TextUtils.formatBytes(app.models.freeBytes()))
                    KV("Models on device", TextUtils.formatBytes(app.models.modelsOnDeviceBytes()))
                }
            }
            Text("Models run entirely on this device. Download once over Wi-Fi; after that no connection is needed. 4B models need an 8 GB phone; 6 GB phones should use the 2B/1.7B tier.",
                style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant, modifier = Modifier.padding(vertical = 6.dp))
        }
        for (tier in Tier.entries) {
            item { Text(tier.title.uppercase(), style = MaterialTheme.typography.labelMedium, color = MaterialTheme.colorScheme.onSurfaceVariant, modifier = Modifier.padding(top = 8.dp)) }
            items(ModelCatalog.models.filter { it.tier == tier }, key = { it.id }) { spec ->
                val inst = installed.firstOrNull { it.id == spec.id }
                val dl = downloads[spec.id]
                Card(Modifier.fillMaxWidth()) {
                    Column(Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Text(spec.name, style = MaterialTheme.typography.titleMedium, modifier = Modifier.weight(1f))
                            when {
                                inst != null && loaded?.id == inst.id -> AssistChip(onClick = {}, label = { Text("Loaded") }, leadingIcon = { Icon(Icons.Default.CheckCircle, null) })
                                inst != null -> Button(onClick = { app.loadModel(inst.id); app.showModels.value = false }) { Text("Load") }
                                dl != null && dl.isActive -> OutlinedButton(onClick = { app.models.cancelDownload(spec.id) }) { Text("Cancel") }
                                else -> OutlinedButton(onClick = { app.models.download(spec) }) { Text("Get") }
                            }
                            if (inst != null) IconButton(onClick = { confirmDelete = inst }) { Icon(Icons.Default.Delete, "Delete", tint = MaterialTheme.colorScheme.error) }
                        }
                        Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                            Tag(spec.parameters); Tag(spec.quantization)
                            if (spec.hasVision) Tag("Vision")
                            if (spec.id == ModelCatalog.recommendedID) Tag("Recommended")
                        }
                        Text(spec.summary, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
                        if (dl != null && dl.isActive) {
                            LinearProgressIndicator(progress = { dl.fraction }, modifier = Modifier.fillMaxWidth())
                            Text(if (dl.status == DownloadManager.STATUS_PAUSED) "Paused (waiting for network)" else "${TextUtils.formatBytes(dl.bytes)} of ${TextUtils.formatBytes(spec.totalBytes)}", style = MaterialTheme.typography.labelSmall)
                        } else if (dl != null && dl.status == DownloadManager.STATUS_FAILED) {
                            Text("Download failed (code ${dl.reason}). Tap Get to retry.", style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.error)
                        } else {
                            val fit = spec.fit(app.physicalMemory, settings.engine.contextLength)
                            Text("${TextUtils.formatBytes(spec.totalBytes)} · " + when (fit) {
                                ModelSpec.Fit.GOOD -> "Fits this device"
                                ModelSpec.Fit.TIGHT -> "Tight fit: use a smaller context"
                                ModelSpec.Fit.TOO_LARGE -> "Probably too large for this device"
                            }, style = MaterialTheme.typography.labelSmall,
                                color = if (fit == ModelSpec.Fit.TOO_LARGE) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.onSurfaceVariant)
                        }
                    }
                }
            }
        }
        val imported = installed.filter { it.spec == null }
        item {
            Text("IMPORTED", style = MaterialTheme.typography.labelMedium, color = MaterialTheme.colorScheme.onSurfaceVariant, modifier = Modifier.padding(top = 8.dp))
            Text("Any GGUF works. Open a .gguf from your file manager with EdgeChat, or pick one here.", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            Spacer(Modifier.height(6.dp))
            OutlinedButton(onClick = { importPicker.launch(arrayOf("*/*")) }) { Text("Import a GGUF file") }
        }
        items(imported, key = { it.id }) { m ->
            Card(Modifier.fillMaxWidth()) {
                Row(Modifier.padding(14.dp), verticalAlignment = Alignment.CenterVertically) {
                    Column(Modifier.weight(1f)) { Text(m.name, style = MaterialTheme.typography.titleMedium); Text(TextUtils.formatBytes(m.sizeBytes), style = MaterialTheme.typography.labelSmall) }
                    if (loaded?.id == m.id) Text("Loaded") else Button(onClick = { app.loadModel(m.id); app.showModels.value = false }) { Text("Load") }
                    IconButton(onClick = { confirmDelete = m }) { Icon(Icons.Default.Delete, "Delete", tint = MaterialTheme.colorScheme.error) }
                }
            }
        }
    }

    confirmDelete?.let { m ->
        AlertDialog(onDismissRequest = { confirmDelete = null }, title = { Text("Delete ${m.name}?") },
            text = { Text("Frees ${TextUtils.formatBytes(m.sizeBytes)}. You can download it again later.") },
            confirmButton = { TextButton(onClick = {
                confirmDelete = null
                if (loaded?.id == m.id) app.viewModelScopeLaunch { app.engine.unload() }
                app.models.delete(m)
            }) { Text("Delete", color = MaterialTheme.colorScheme.error) } },
            dismissButton = { TextButton(onClick = { confirmDelete = null }) { Text("Cancel") } })
    }
}

@Composable private fun KV(k: String, v: String) {
    Row { Text(k, modifier = Modifier.weight(1f)); Text(v, color = MaterialTheme.colorScheme.onSurfaceVariant) }
}

@Composable private fun Tag(t: String) {
    Text(t, style = MaterialTheme.typography.labelSmall, modifier = Modifier
        .padding(0.dp)
        .then(Modifier), color = MaterialTheme.colorScheme.primary)
    Spacer(Modifier.width(2.dp))
}
