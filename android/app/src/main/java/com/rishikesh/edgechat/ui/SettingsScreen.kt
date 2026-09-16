package com.rishikesh.edgechat.ui

import android.content.Intent
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.FilterChip
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Slider
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.core.content.FileProvider
import com.rishikesh.edgechat.AppModel
import com.rishikesh.edgechat.BuildConfig
import com.rishikesh.edgechat.data.DiagnosticsLog
import com.rishikesh.edgechat.engine.EngineController
import com.rishikesh.edgechat.model.AppSettings
import com.rishikesh.edgechat.model.TextUtils
import java.util.Locale

@Composable
fun SettingsScreen(app: AppModel) {
    val s by app.settings.collectAsState()
    val state by app.engine.state.collectAsState()
    val context = LocalContext.current
    val info = (state as? EngineController.State.Ready)?.info

    Column(Modifier.fillMaxWidth().verticalScroll(rememberScrollState()).padding(horizontal = 16.dp).padding(bottom = 32.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        Text("Settings", style = MaterialTheme.typography.headlineMedium, modifier = Modifier.padding(vertical = 8.dp))

        Section("Inference") {
            Text("Context length", style = MaterialTheme.typography.bodyLarge)
            Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                for (n in listOf(2048, 4096, 8192, 16384, 32768)) FilterChip(selected = s.engine.contextLength == n, onClick = { app.updateSettings { it.copy(engine = it.engine.copy(contextLength = n)) } }, label = { Text(TextUtils.formatContext(n)) })
            }
            ToggleRow("Flash attention", s.engine.flashAttention) { v -> app.updateSettings { it.copy(engine = it.engine.copy(flashAttention = v)) } }
            ToggleRow("8-bit KV cache", s.engine.kvCacheQ8) { v -> app.updateSettings { it.copy(engine = it.engine.copy(kvCacheQ8 = v)) } }
            ToggleRow("Skip thinking (Qwen3 hybrids)", s.engine.disableThinking) { v -> app.updateSettings { it.copy(engine = it.engine.copy(disableThinking = v)) } }
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text("Threads: ${if (s.engine.threads == 0) "auto" else s.engine.threads}${info?.let { " (${it.threads})" } ?: ""}", modifier = Modifier.weight(1f))
                OutlinedButton(onClick = { app.updateSettings { it.copy(engine = it.engine.copy(threads = maxOf(0, it.engine.threads - 1))) } }) { Text("−") }
                OutlinedButton(onClick = { app.updateSettings { it.copy(engine = it.engine.copy(threads = minOf(8, it.engine.threads + 1))) } }) { Text("+") }
            }
            info?.let { Text("KV cache at ${TextUtils.formatContext(s.engine.contextLength)}: ${TextUtils.formatBytes(it.kvCacheBytes(s.engine.contextLength))}", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant) }
            Text("Longer contexts let you attach bigger documents but use more memory (roughly 64 KB per token for a 4B model; 8-bit KV cache halves that). Changes apply when the model is reloaded.", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            Button(onClick = { app.viewModelScopeLaunch { app.engine.applyEngineSettingsIfNeeded() } }, enabled = info != null) { Text("Apply and reload model") }
        }

        Section("Long conversations") {
            ToggleRow("Summarize dropped history", s.summarizeDroppedTurns) { v -> app.updateSettings { it.copy(summarizeDroppedTurns = v) } }
            ToggleRow("Compact before the window fills", s.autoCompact) { v -> app.updateSettings { it.copy(autoCompact = v) } }
            ToggleRow("Recall older context (RAG)", s.memoryRetrieval) { v -> app.updateSettings { it.copy(memoryRetrieval = v) } }
            ToggleRow("Shift context mid-reply", s.engine.contextShift) { v -> app.updateSettings { it.copy(engine = it.engine.copy(contextShift = v)) } }
            ToggleRow("Cache snapshots per chat", s.kvSnapshots) { v -> app.updateSettings { it.copy(kvSnapshots = v) } }
            Text("When a chat outgrows the context window, the model writes a running summary of the oldest messages, and relevant older messages or document passages are retrieved into each new turn. Compacting early runs that summary right after a reply once the window is about three-quarters full.", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }

        Section("Generation") {
            SliderRow("Temperature", s.sampling.temperature, 0f..1.5f) { v -> app.updateSettings { it.copy(sampling = it.sampling.copy(temperature = v)) } }
            SliderRow("Top-p", s.sampling.topP, 0.1f..1f) { v -> app.updateSettings { it.copy(sampling = it.sampling.copy(topP = v)) } }
            SliderRow("Min-p", s.sampling.minP, 0f..0.3f) { v -> app.updateSettings { it.copy(sampling = it.sampling.copy(minP = v)) } }
            SliderRow("Repeat penalty", s.sampling.repeatPenalty, 1f..1.5f) { v -> app.updateSettings { it.copy(sampling = it.sampling.copy(repeatPenalty = v)) } }
            Text("Reply length", style = MaterialTheme.typography.bodyLarge)
            Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                for (n in listOf(0, 512, 1024, 2048, 4096)) FilterChip(selected = s.sampling.maxTokens == n, onClick = { app.updateSettings { it.copy(sampling = it.sampling.copy(maxTokens = n)) } }, label = { Text(if (n == 0) "Unlimited" else "$n") })
            }
            Text("With no limit, a reply ends when the model finishes. If it reaches the edge of the context window, the oldest messages are summarized away and the reply continues automatically; you can also tap Continue under a reply that stopped early.", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }

        Section("System prompt") {
            OutlinedTextField(value = s.systemPrompt, onValueChange = { v -> app.updateSettings { it.copy(systemPrompt = v) } }, modifier = Modifier.fillMaxWidth(), minLines = 4, textStyle = MaterialTheme.typography.bodySmall)
            OutlinedButton(onClick = { app.updateSettings { it.copy(systemPrompt = AppSettings.DEFAULT_SYSTEM_PROMPT) } }) { Text("Reset to default") }
        }

        Section("Attachments") {
            ToggleRow("OCR photos for text-only models", s.ocrForImages) { v -> app.updateSettings { it.copy(ocrForImages = v) } }
            Text("Max image size", style = MaterialTheme.typography.bodyLarge)
            Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                for (n in listOf(512, 768, 1024)) FilterChip(selected = s.maxImageDimension == n, onClick = { app.updateSettings { it.copy(maxImageDimension = n) } }, label = { Text("$n px") })
            }
        }

        Section("Display") {
            ToggleRow("Show reply stats", s.showStats) { v -> app.updateSettings { it.copy(showStats = v) } }
            ToggleRow("Load last model at launch", s.autoLoadLastModel) { v -> app.updateSettings { it.copy(autoLoadLastModel = v) } }
        }

        Section("About") {
            Text("EdgeChat ${BuildConfig.VERSION_NAME} · llama.cpp b10988 · everything runs on this device", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            OutlinedButton(onClick = {
                val file = DiagnosticsLog.logFile() ?: return@OutlinedButton
                val uri = FileProvider.getUriForFile(context, "${context.packageName}.files", file)
                context.startActivity(Intent.createChooser(Intent(Intent.ACTION_SEND).apply { type = "text/plain"; putExtra(Intent.EXTRA_STREAM, uri); addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION) }, "Share diagnostics log"))
            }) { Text("Share diagnostics log") }
        }
    }
}

@Composable private fun Section(title: String, content: @Composable () -> Unit) {
    Text(title.uppercase(), style = MaterialTheme.typography.labelMedium, color = MaterialTheme.colorScheme.onSurfaceVariant, modifier = Modifier.padding(top = 8.dp))
    Card(Modifier.fillMaxWidth()) { Column(Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) { content() } }
}

@Composable private fun ToggleRow(label: String, value: Boolean, onChange: (Boolean) -> Unit) {
    Row(verticalAlignment = Alignment.CenterVertically) { Text(label, modifier = Modifier.weight(1f)); Switch(checked = value, onCheckedChange = onChange) }
}

@Composable private fun SliderRow(label: String, value: Float, range: ClosedFloatingPointRange<Float>, onChange: (Float) -> Unit) {
    Column {
        Row { Text(label, modifier = Modifier.weight(1f)); Text(String.format(Locale.US, "%.2f", value), color = MaterialTheme.colorScheme.onSurfaceVariant) }
        Slider(value = value, onValueChange = onChange, valueRange = range)
    }
}
