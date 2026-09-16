package com.rishikesh.edgechat

import android.app.ActivityManager
import android.app.Application
import android.content.Context
import com.rishikesh.edgechat.data.DiagnosticsLog
import com.tom_roush.pdfbox.android.PDFBoxResourceLoader

class EdgeChatApplication : Application() {
    override fun onCreate() {
        super.onCreate()
        PDFBoxResourceLoader.init(applicationContext)
        val am = getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val mi = ActivityManager.MemoryInfo().also { am.getMemoryInfo(it) }
        DiagnosticsLog.install(this, BuildConfig.VERSION_NAME, mi.totalMem)
    }
}
