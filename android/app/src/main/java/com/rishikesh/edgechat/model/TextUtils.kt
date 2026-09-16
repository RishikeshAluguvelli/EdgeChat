package com.rishikesh.edgechat.model

import java.util.Locale

object TextUtils {
    /** Keeps the head and tail of an over-long text, dropping the middle. */
    fun truncateMiddle(text: String, maxCharacters: Int, headFraction: Double = 0.7): Pair<String, Boolean> {
        if (text.length <= maxCharacters || maxCharacters <= 64) return text to false
        val marker = "\n\n[… ${text.length - maxCharacters} characters omitted …]\n\n"
        val budget = maxOf(0, maxCharacters - marker.length)
        val head = (budget * headFraction).toInt()
        val tail = budget - head
        return (text.take(head) + marker + text.takeLast(tail)) to true
    }

    /** "512 tokens", "4k", "32k". */
    fun formatContext(tokens: Int): String = when {
        tokens >= 1024 && tokens % 1024 == 0 -> "${tokens / 1024}k"
        tokens >= 1024 -> String.format(Locale.US, "%.1fk", tokens / 1024.0)
        else -> "$tokens tokens"
    }

    fun formatBytes(bytes: Long): String = when {
        bytes >= 1_000_000_000L -> String.format(Locale.US, "%.2f GB", bytes / 1e9)
        bytes >= 1_000_000L -> String.format(Locale.US, "%.1f MB", bytes / 1e6)
        else -> String.format(Locale.US, "%d KB", bytes / 1000)
    }

    fun relativeTime(iso: String): String {
        val then = Iso.parse(iso).toEpochMilli()
        val diff = (System.currentTimeMillis() - then) / 1000
        return when {
            diff < 60 -> "just now"
            diff < 3600 -> "${diff / 60} min ago"
            diff < 86400 -> "${diff / 3600} h ago"
            else -> "${diff / 86400} d ago"
        }
    }
}
