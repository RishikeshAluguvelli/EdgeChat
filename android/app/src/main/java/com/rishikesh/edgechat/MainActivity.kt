package com.rishikesh.edgechat

import android.content.Intent
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.viewModels
import com.rishikesh.edgechat.ui.EdgeChatTheme
import com.rishikesh.edgechat.ui.RootScreen

class MainActivity : ComponentActivity() {
    private val app: AppModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        handleIntent(intent)
        setContent { EdgeChatTheme { RootScreen(app) } }
    }

    override fun onNewIntent(intent: Intent) { super.onNewIntent(intent); handleIntent(intent) }

    /** "Open with EdgeChat" on a .gguf file imports it. Debug builds also accept `--es autoPrompt "a ||| b"`. */
    private fun handleIntent(intent: Intent?) {
        if (intent?.action == Intent.ACTION_VIEW) intent.data?.let { app.importModel(it); app.showModels.value = true }
        if (BuildConfig.DEBUG) intent?.getStringExtra("autoPrompt")?.let { app.runAutomation(it, intent.getIntExtra("autoContext", 0)) }
    }

    override fun onStop() { super.onStop(); app.onBackground() }
}
