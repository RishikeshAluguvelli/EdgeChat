package com.rishikesh.edgechat

import android.app.Application
import android.net.Uri
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.rishikesh.edgechat.data.AttachmentIngest
import com.rishikesh.edgechat.data.ConversationStore
import com.rishikesh.edgechat.data.DiagnosticsLog
import com.rishikesh.edgechat.data.ModelManager
import com.rishikesh.edgechat.engine.EngineController
import com.rishikesh.edgechat.model.AppSettings
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch

/** App-wide state: settings, conversations, models, engine. One instance for the activity. */
class AppModel(app: Application) : AndroidViewModel(app) {
    private val _settings = MutableStateFlow(AppSettings.load(app))
    val settings: StateFlow<AppSettings> = _settings
    val store = ConversationStore(app)
    val models = ModelManager(app)
    val engine = EngineController(app, store, viewModelScope) { _settings.value }

    private val _selectedConversationID = MutableStateFlow<String?>(null)
    val selectedConversationID: StateFlow<String?> = _selectedConversationID
    val showModels = MutableStateFlow(false)
    val showSettings = MutableStateFlow(false)

    val physicalMemory: Long = run {
        val am = app.getSystemService(android.content.Context.ACTIVITY_SERVICE) as android.app.ActivityManager
        android.app.ActivityManager.MemoryInfo().also { am.getMemoryInfo(it) }.totalMem
    }

    init {
        engine.engine.setLogger(DiagnosticsLog.logger)
        store.loadAll()
        viewModelScope.launch {
            val s = _settings.value
            if (s.autoLoadLastModel) models.activeInstalledModel(s.activeModelID)?.let { engine.load(it) }
            else if (models.installed.value.isEmpty()) showModels.value = true
            if (models.installed.value.isEmpty()) showModels.value = true
        }
    }

    fun updateSettings(transform: (AppSettings) -> AppSettings) {
        val next = transform(_settings.value)
        _settings.value = next
        next.save(getApplication())
    }

    fun select(id: String?) {
        val previous = _selectedConversationID.value
        _selectedConversationID.value = id
        if (previous != null && previous != id) viewModelScope.launch { engine.saveCurrentSnapshot() }
    }

    fun newConversation() { select(store.create().id) }

    fun deleteConversation(id: String) {
        engine.forgetConversation(id)
        store.delete(id)
        if (_selectedConversationID.value == id) _selectedConversationID.value = null
    }

    fun loadModel(id: String) {
        val m = models.installed.value.firstOrNull { it.id == id } ?: return
        viewModelScope.launch {
            engine.load(m)
            if (engine.isReady) updateSettings { it.copy(activeModelID = id) }
        }
    }

    fun importModel(uri: Uri) {
        viewModelScope.launch(kotlinx.coroutines.Dispatchers.IO) {
            val name = AttachmentIngest.displayName(getApplication(), uri)
            models.importModel(uri, name)
        }
    }

    fun onBackground() { viewModelScope.launch { engine.saveCurrentSnapshot() } }

    /** Debug automation: loads the first installed model, opens a new chat and sends each `|||`-separated prompt. */
    fun runAutomation(prompts: String, contextLength: Int) {
        viewModelScope.launch {
            if (contextLength > 0) updateSettings { it.copy(engine = it.engine.copy(contextLength = contextLength)) }
            val first = models.installed.value.firstOrNull() ?: return@launch
            if (!engine.isReady || engine.loadedModel?.id != first.id) engine.load(first)
            newConversation()
            val id = selectedConversationID.value ?: return@launch
            for (p in prompts.split("|||").map { it.trim() }.filter { it.isNotEmpty() }) engine.send(id, p, emptyList())
        }
    }

    /** Runs a suspending engine action from Compose callbacks. */
    fun viewModelScopeLaunch(block: suspend () -> Unit) { viewModelScope.launch { block() } }

    override fun onCleared() { engine.engine.close() }
}
