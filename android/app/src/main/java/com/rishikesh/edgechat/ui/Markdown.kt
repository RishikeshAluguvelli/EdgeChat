package com.rishikesh.edgechat.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextDecoration
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

/** Lightweight block-level Markdown (paragraphs, headings, lists, quotes, fenced code, tables, rules); inline bold/italic/code/links. */
sealed class MdBlock {
    data class Paragraph(val text: String) : MdBlock()
    data class Heading(val level: Int, val text: String) : MdBlock()
    data class Bullets(val items: List<Pair<Int, String>>) : MdBlock()
    data class Numbered(val items: List<Pair<String, String>>) : MdBlock()
    data class Quote(val text: String) : MdBlock()
    data class Code(val language: String?, val code: String) : MdBlock()
    data class Table(val header: List<String>, val rows: List<List<String>>) : MdBlock()
    data object Rule : MdBlock()
}

object MarkdownParser {
    private val numbered = Regex("^(\\d+)[.)]\\s+")
    private val cache = object : LinkedHashMap<String, List<MdBlock>>(64, 0.75f, true) {
        override fun removeEldestEntry(eldest: MutableMap.MutableEntry<String, List<MdBlock>>?) = size > 200
    }

    @Synchronized
    fun parse(text: String): List<MdBlock> = cache[text] ?: parseUncached(text).also { if (text.length > 64) cache[text] = it }

    fun parseUncached(text: String): List<MdBlock> {
        val blocks = mutableListOf<MdBlock>()
        val paragraph = mutableListOf<String>()
        val bullets = mutableListOf<Pair<Int, String>>()
        val numberedItems = mutableListOf<Pair<String, String>>()
        val quote = mutableListOf<String>()
        val tableLines = mutableListOf<String>()
        var codeLang: String? = null
        val code = mutableListOf<String>()
        var inCode = false

        fun flushTable() {
            if (tableLines.isEmpty()) return
            val rows = tableLines.map(::splitRow).filter { it.isNotEmpty() }
            val isSep: (List<String>) -> Boolean = { r -> r.all { c -> c.all { it == '-' || it == ':' || it == ' ' } && c.contains('-') } }
            val body = rows.filterNot(isSep)
            if (body.isNotEmpty() && rows.any(isSep)) blocks += MdBlock.Table(body.first(), body.drop(1))
            else blocks += MdBlock.Paragraph(tableLines.joinToString(" "))
            tableLines.clear()
        }
        fun flush() {
            flushTable()
            if (paragraph.isNotEmpty()) { blocks += MdBlock.Paragraph(paragraph.joinToString(" ")); paragraph.clear() }
            if (bullets.isNotEmpty()) { blocks += MdBlock.Bullets(bullets.toList()); bullets.clear() }
            if (numberedItems.isNotEmpty()) { blocks += MdBlock.Numbered(numberedItems.toList()); numberedItems.clear() }
            if (quote.isNotEmpty()) { blocks += MdBlock.Quote(quote.joinToString(" ")); quote.clear() }
        }

        for (line in text.split('\n')) {
            val trimmed = line.trim()
            if (trimmed.startsWith("```")) {
                if (inCode) { blocks += MdBlock.Code(codeLang, code.joinToString("\n")); code.clear(); codeLang = null; inCode = false }
                else { flush(); codeLang = trimmed.drop(3).trim().ifEmpty { null }; inCode = true }
                continue
            }
            if (inCode) { code += line; continue }
            if (trimmed.isEmpty()) { flush(); continue }
            if (trimmed.startsWith("|") && trimmed.length > 1) {
                if (paragraph.isNotEmpty() || bullets.isNotEmpty() || numberedItems.isNotEmpty() || quote.isNotEmpty()) flush()
                tableLines += trimmed; continue
            }
            if (tableLines.isNotEmpty()) flushTable()
            if (trimmed == "---" || trimmed == "***") { flush(); blocks += MdBlock.Rule; continue }
            val h = headingLevel(trimmed)
            if (h != null) { flush(); blocks += MdBlock.Heading(h, trimmed.dropWhile { it == '#' }.trim()); continue }
            if (trimmed.startsWith("- ") || trimmed.startsWith("* ") || trimmed.startsWith("• ") || trimmed.startsWith("+ ")) {
                if (paragraph.isNotEmpty() || numberedItems.isNotEmpty() || quote.isNotEmpty()) flush()
                val indent: Int = line.takeWhile { it == ' ' || it == '\t' }.fold(0) { acc, ch -> acc + (if (ch == '\t') 4 else 1) }
                bullets += (minOf(3, indent / 2)) to trimmed.drop(2).trim(); continue
            }
            val m = numbered.find(trimmed)
            if (m != null) {
                if (paragraph.isNotEmpty() || bullets.isNotEmpty() || quote.isNotEmpty()) flush()
                numberedItems += (m.groupValues[1] + ".") to trimmed.substring(m.range.last + 1); continue
            }
            if (trimmed.startsWith("> ") || trimmed == ">") {
                if (paragraph.isNotEmpty() || bullets.isNotEmpty() || numberedItems.isNotEmpty()) flush()
                quote += trimmed.drop(1).trim(); continue
            }
            if (line.startsWith("  ") && bullets.isNotEmpty()) { val (lvl, t) = bullets.removeAt(bullets.lastIndex); bullets += lvl to "$t $trimmed"; continue }
            if (line.startsWith("  ") && numberedItems.isNotEmpty()) { val (l, t) = numberedItems.removeAt(numberedItems.lastIndex); numberedItems += l to "$t $trimmed"; continue }
            if (bullets.isNotEmpty() || numberedItems.isNotEmpty() || quote.isNotEmpty()) flush()
            paragraph += trimmed
        }
        if (inCode) blocks += MdBlock.Code(codeLang, code.joinToString("\n"))
        flush()
        return blocks
    }

    private fun headingLevel(s: String): Int? {
        val n = s.takeWhile { it == '#' }.length
        return if (n in 1..4 && s.getOrNull(n) == ' ') n else null
    }

    private fun splitRow(line: String): List<String> {
        val cells = mutableListOf<String>(); val cur = StringBuilder(); var inCode = false
        for (ch in line.drop(1)) {
            if (ch == '`') inCode = !inCode
            if (ch == '|' && !inCode) { cells += cur.toString().trim(); cur.clear() } else cur.append(ch)
        }
        val tail = cur.toString().trim()
        if (tail.isNotEmpty()) cells += tail
        return cells.map { it.replace("<br>", " ").replace("<br/>", " ") }
    }

    /** Inline **bold**, *italic*, `code`, [text](url) into an AnnotatedString. */
    fun inline(s: String, codeColor: androidx.compose.ui.graphics.Color, linkColor: androidx.compose.ui.graphics.Color): AnnotatedString = buildAnnotatedString {
        var i = 0
        val n = s.length
        while (i < n) {
            val c = s[i]
            when {
                c == '`' -> {
                    val end = s.indexOf('`', i + 1)
                    if (end > i) { withStyle(SpanStyle(fontFamily = FontFamily.Monospace, background = codeColor, fontSize = 13.sp)) { append(s.substring(i + 1, end)) }; i = end + 1; continue }
                }
                s.startsWith("**", i) -> {
                    val end = s.indexOf("**", i + 2)
                    if (end > i) { withStyle(SpanStyle(fontWeight = FontWeight.SemiBold)) { append(inline(s.substring(i + 2, end), codeColor, linkColor)) }; i = end + 2; continue }
                }
                c == '*' || (c == '_' && (i == 0 || !s[i - 1].isLetterOrDigit())) -> {
                    val end = s.indexOf(c, i + 1)
                    if (end > i + 1 && !s[i + 1].isWhitespace()) { withStyle(SpanStyle(fontStyle = FontStyle.Italic)) { append(inline(s.substring(i + 1, end), codeColor, linkColor)) }; i = end + 1; continue }
                }
                c == '[' -> {
                    val close = s.indexOf("](", i)
                    val end = if (close > 0) s.indexOf(')', close) else -1
                    if (close > i && end > close) {
                        val label = s.substring(i + 1, close); val url = s.substring(close + 2, end)
                        pushStringAnnotation("URL", url)
                        withStyle(SpanStyle(color = linkColor, textDecoration = TextDecoration.Underline)) { append(label) }
                        pop(); i = end + 1; continue
                    }
                }
            }
            append(c); i++
        }
    }
}

@Composable
fun MarkdownText(text: String, modifier: Modifier = Modifier) {
    val blocks = remember(text) { MarkdownParser.parse(text) }
    val codeBg = MaterialTheme.colorScheme.surfaceVariant
    val link = MaterialTheme.colorScheme.primary
    val body = MaterialTheme.typography.bodyLarge
    SelectionContainer {
        Column(modifier, verticalArrangement = Arrangement.spacedBy(8.dp)) {
            for (b in blocks) when (b) {
                is MdBlock.Paragraph -> Text(MarkdownParser.inline(b.text, codeBg, link), style = body)
                is MdBlock.Heading -> Text(MarkdownParser.inline(b.text, codeBg, link),
                    style = when (b.level) { 1 -> MaterialTheme.typography.titleLarge; 2 -> MaterialTheme.typography.titleMedium; else -> MaterialTheme.typography.titleSmall },
                    fontWeight = FontWeight.Bold)
                is MdBlock.Bullets -> Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                    for ((level, item) in b.items) Row(Modifier.padding(start = (level * 16).dp)) {
                        Text(if (level == 0) "•" else "◦", style = body, modifier = Modifier.width(18.dp))
                        Text(MarkdownParser.inline(item, codeBg, link), style = body, modifier = Modifier.weight(1f))
                    }
                }
                is MdBlock.Numbered -> Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                    for ((label, item) in b.items) Row {
                        Text(label, style = body, modifier = Modifier.width(28.dp))
                        Text(MarkdownParser.inline(item, codeBg, link), style = body, modifier = Modifier.weight(1f))
                    }
                }
                is MdBlock.Quote -> Row {
                    Column(Modifier.width(3.dp).background(MaterialTheme.colorScheme.outline, RoundedCornerShape(2.dp)).padding(vertical = 10.dp)) {}
                    Text(MarkdownParser.inline(b.text, codeBg, link), style = body, color = MaterialTheme.colorScheme.onSurfaceVariant, modifier = Modifier.padding(start = 8.dp))
                }
                is MdBlock.Code -> Column(Modifier.fillMaxWidth().background(codeBg, RoundedCornerShape(10.dp)).padding(10.dp)) {
                    b.language?.let { Text(it, style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.onSurfaceVariant) }
                    Row(Modifier.horizontalScroll(rememberScrollState())) {
                        Text(b.code, fontFamily = FontFamily.Monospace, fontSize = 13.sp, softWrap = false)
                    }
                }
                is MdBlock.Table -> MarkdownTable(b, codeBg, link)
                MdBlock.Rule -> HorizontalDivider()
            }
        }
    }
}

@Composable
private fun MarkdownTable(t: MdBlock.Table, codeBg: androidx.compose.ui.graphics.Color, link: androidx.compose.ui.graphics.Color) {
    val columns = maxOf(t.header.size, t.rows.maxOfOrNull { it.size } ?: 0)
    val widths = (0 until columns).map { i -> val longest = (listOf(t.header) + t.rows).maxOf { r -> r.getOrNull(i)?.length ?: 0 }; (longest * 6.5f).coerceIn(70f, 220f).dp }
    Column(Modifier.fillMaxWidth().background(codeBg, RoundedCornerShape(10.dp)).padding(10.dp).horizontalScroll(rememberScrollState())) {
        Row(horizontalArrangement = Arrangement.spacedBy(14.dp), verticalAlignment = Alignment.Top) {
            for (i in 0 until columns) Text(MarkdownParser.inline(t.header.getOrNull(i) ?: "", codeBg, link), style = MaterialTheme.typography.bodySmall, fontWeight = FontWeight.SemiBold, modifier = Modifier.width(widths[i]))
        }
        HorizontalDivider(Modifier.padding(vertical = 4.dp))
        for (r in t.rows) Row(Modifier.padding(vertical = 3.dp), horizontalArrangement = Arrangement.spacedBy(14.dp), verticalAlignment = Alignment.Top) {
            for (i in 0 until columns) Text(MarkdownParser.inline(r.getOrNull(i) ?: "", codeBg, link), style = MaterialTheme.typography.bodySmall, modifier = Modifier.width(widths[i]))
        }
    }
}
