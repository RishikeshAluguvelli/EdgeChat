import Foundation
import CoreGraphics
import ImageIO
import PDFKit
import Vision
import UniformTypeIdentifiers
#if canImport(UIKit)
import UIKit
#elseif canImport(AppKit)
import AppKit
#endif

public struct DocumentExtraction: Sendable {
    public let text: String
    public let pageCount: Int?
    public let truncated: Bool
    /// "pdf", "pdf+ocr", "text", "rtf", "html"
    public let method: String
}

public enum AttachmentError: Error, LocalizedError {
    case unreadable(String)
    case unsupportedType(String)
    case imageDecodeFailed
    case empty

    public var errorDescription: String? {
        switch self {
        case .unreadable(let n): return "Could not read \(n)."
        case .unsupportedType(let n): return "\(n) is not a supported file type. Use PDF, text, Markdown, code, CSV, JSON, RTF, HTML, or images."
        case .imageDecodeFailed: return "The image could not be decoded."
        case .empty: return "No text could be extracted from the file."
        }
    }
}

/// Turns user files into model input: text for documents, RGB bitmaps (plus OCR fallback) for images.
public enum AttachmentProcessor {

    public static let markdownType = UTType("net.daringfireball.markdown") ?? .plainText

    /// Types accepted by the document picker.
    public static var supportedDocumentTypes: [UTType] {
        [.pdf, .plainText, .utf8PlainText, .text, .sourceCode, .json, .commaSeparatedText, .tabSeparatedText,
         .xml, .html, .rtf, .yaml, .log, .propertyList, .swiftSource, .cSource, .cPlusPlusSource, .objectiveCSource,
         .pythonScript, .javaScript, .shellScript, markdownType]
    }

    public static var supportedImageTypes: [UTType] { [.image] }

    // MARK: Documents

    public static func extractText(from url: URL, maxCharacters: Int, ocrFallback: Bool = true) async throws -> DocumentExtraction {
        let name = url.lastPathComponent
        let type = UTType(filenameExtension: url.pathExtension) ?? .data

        if type.conforms(to: .pdf) {
            return try await extractPDF(url: url, maxCharacters: maxCharacters, ocrFallback: ocrFallback)
        }
        if type.conforms(to: .rtf) {
            guard let attributed = try? NSAttributedString(url: url, options: [.documentType: NSAttributedString.DocumentType.rtf], documentAttributes: nil) else {
                throw AttachmentError.unreadable(name)
            }
            return finish(attributed.string, pages: nil, maxCharacters: maxCharacters, method: "rtf")
        }
        if type.conforms(to: .html) {
            let raw = try readText(url: url)
            return finish(stripHTML(raw), pages: nil, maxCharacters: maxCharacters, method: "html")
        }
        if type.conforms(to: .text) || type.conforms(to: .sourceCode) || type.conforms(to: .json)
            || type.conforms(to: .xml) || type.conforms(to: .propertyList) || type == markdownType
            || ["md", "markdown", "txt", "csv", "tsv", "log", "yaml", "yml", "toml", "ini", "cfg", "env"].contains(url.pathExtension.lowercased()) {
            let raw = try readText(url: url)
            return finish(raw, pages: nil, maxCharacters: maxCharacters, method: "text")
        }
        // Last resort: if it decodes as UTF-8 text, treat it as text.
        if let data = try? Data(contentsOf: url), data.count < 5_000_000, let s = String(data: data, encoding: .utf8), !s.isEmpty {
            return finish(s, pages: nil, maxCharacters: maxCharacters, method: "text")
        }
        throw AttachmentError.unsupportedType(name)
    }

    private static func readText(url: URL) throws -> String {
        guard let data = try? Data(contentsOf: url) else { throw AttachmentError.unreadable(url.lastPathComponent) }
        if let s = String(data: data, encoding: .utf8) { return s }
        if let s = String(data: data, encoding: .utf16) { return s }
        if let s = String(data: data, encoding: .isoLatin1) { return s }
        return String(decoding: data, as: UTF8.self)
    }

    private static func finish(_ text: String, pages: Int?, maxCharacters: Int, method: String) -> DocumentExtraction {
        let cleaned = text.replacingOccurrences(of: "\r\n", with: "\n").trimmingCharacters(in: .whitespacesAndNewlines)
        let (t, truncated) = TextUtils.truncateMiddle(cleaned, maxCharacters: maxCharacters)
        return DocumentExtraction(text: t, pageCount: pages, truncated: truncated, method: method)
    }

    private static func extractPDF(url: URL, maxCharacters: Int, ocrFallback: Bool) async throws -> DocumentExtraction {
        guard let doc = PDFDocument(url: url) else { throw AttachmentError.unreadable(url.lastPathComponent) }
        var parts: [String] = []
        var total = 0
        var usedOCR = false
        var truncated = false
        for i in 0..<doc.pageCount {
            guard let page = doc.page(at: i) else { continue }
            var text = page.string?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            if text.count < 20, ocrFallback, let image = render(page: page, maxDimension: 1800) {
                if let ocr = try? await recognizeText(in: image), !ocr.isEmpty {
                    text = ocr
                    usedOCR = true
                }
            }
            let header = doc.pageCount > 1 ? "--- Page \(i + 1) of \(doc.pageCount) ---\n" : ""
            parts.append(header + text)
            total += text.count + header.count
            if total > maxCharacters * 2 { truncated = true; break } // stop reading far past the budget
        }
        let joined = parts.joined(separator: "\n\n")
        guard !joined.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { throw AttachmentError.empty }
        let (t, cut) = TextUtils.truncateMiddle(joined, maxCharacters: maxCharacters)
        return DocumentExtraction(text: t, pageCount: doc.pageCount, truncated: truncated || cut, method: usedOCR ? "pdf+ocr" : "pdf")
    }

    private static func render(page: PDFPage, maxDimension: CGFloat) -> CGImage? {
        let bounds = page.bounds(for: .mediaBox)
        guard bounds.width > 0, bounds.height > 0 else { return nil }
        let scale = min(maxDimension / bounds.width, maxDimension / bounds.height, 4)
        let w = Int(bounds.width * scale), h = Int(bounds.height * scale)
        guard let cg = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 0,
                                 space: CGColorSpaceCreateDeviceRGB(),
                                 bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { return nil }
        cg.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))
        cg.fill(CGRect(x: 0, y: 0, width: w, height: h))
        cg.scaleBy(x: scale, y: scale)
        cg.translateBy(x: -bounds.origin.x, y: -bounds.origin.y)
        page.draw(with: .mediaBox, to: cg)
        return cg.makeImage()
    }

    private static func stripHTML(_ html: String) -> String {
        var s = html
        for tag in ["script", "style"] {
            s = s.replacingOccurrences(of: "<\(tag)[\\s\\S]*?</\(tag)>", with: " ", options: [.regularExpression, .caseInsensitive])
        }
        s = s.replacingOccurrences(of: "<br\\s*/?>|</p>|</div>|</h[1-6]>|</li>|</tr>", with: "\n", options: [.regularExpression, .caseInsensitive])
        s = s.replacingOccurrences(of: "<[^>]+>", with: " ", options: .regularExpression)
        let entities = ["&nbsp;": " ", "&amp;": "&", "&lt;": "<", "&gt;": ">", "&quot;": "\"", "&#39;": "'"]
        for (k, v) in entities { s = s.replacingOccurrences(of: k, with: v) }
        s = s.replacingOccurrences(of: "[ \\t]+", with: " ", options: .regularExpression)
        s = s.replacingOccurrences(of: "\\n\\s*\\n+", with: "\n\n", options: .regularExpression)
        return s
    }

    // MARK: Images

    /// Decodes, orientation-corrects, and downscales an image into an RGB888 bitmap for the vision encoder.
    public static func makeImageInput(from data: Data, id: String, maxDimension: Int = 1024) throws -> (input: ImageInput, image: CGImage) {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil) else { throw AttachmentError.imageDecodeFailed }
        let options: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: maxDimension,
            kCGImageSourceShouldCacheImmediately: true,
        ]
        guard let cg = CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary) else {
            throw AttachmentError.imageDecodeFailed
        }
        let w = cg.width, h = cg.height
        var rgba = [UInt8](repeating: 0, count: w * h * 4)
        let ok: Bool = rgba.withUnsafeMutableBytes { raw in
            guard let ctx = CGContext(data: raw.baseAddress, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                                      space: CGColorSpaceCreateDeviceRGB(),
                                      bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue)
            else { return false }
            ctx.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))
            ctx.fill(CGRect(x: 0, y: 0, width: w, height: h))
            ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
            return true
        }
        guard ok else { throw AttachmentError.imageDecodeFailed }
        var rgb = [UInt8](repeating: 0, count: w * h * 3)
        for i in 0..<(w * h) {
            rgb[i * 3] = rgba[i * 4]
            rgb[i * 3 + 1] = rgba[i * 4 + 1]
            rgb[i * 3 + 2] = rgba[i * 4 + 2]
        }
        return (ImageInput(id: id, width: w, height: h, rgb: Data(rgb)), cg)
    }

    /// Re-encodes an image as JPEG for storage (bounded size).
    public static func jpegData(_ image: CGImage, quality: Double = 0.85) -> Data? {
        let data = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(data, UTType.jpeg.identifier as CFString, 1, nil) else { return nil }
        CGImageDestinationAddImage(dest, image, [kCGImageDestinationLossyCompressionQuality: quality] as CFDictionary)
        guard CGImageDestinationFinalize(dest) else { return nil }
        return data as Data
    }

    /// On-device OCR (Vision). Returns recognized lines joined by newlines.
    public static func recognizeText(in image: CGImage) async throws -> String {
        try await withCheckedThrowingContinuation { cont in
            DispatchQueue.global(qos: .userInitiated).async {
                let request = VNRecognizeTextRequest()
                request.recognitionLevel = .accurate
                request.usesLanguageCorrection = true
                do {
                    try VNImageRequestHandler(cgImage: image, options: [:]).perform([request])
                    let lines = (request.results ?? []).compactMap { $0.topCandidates(1).first?.string }
                    cont.resume(returning: lines.joined(separator: "\n"))
                } catch {
                    cont.resume(throwing: error)
                }
            }
        }
    }

    /// Coarse scene/object labels (Vision). Empty when unavailable (e.g. Simulator).
    public static func classify(_ image: CGImage, maxLabels: Int = 6) async -> [String] {
        await withCheckedContinuation { cont in
            DispatchQueue.global(qos: .userInitiated).async {
                let request = VNClassifyImageRequest()
                do {
                    try VNImageRequestHandler(cgImage: image, options: [:]).perform([request])
                    let labels = (request.results ?? [])
                        .filter { $0.confidence > 0.3 }
                        .sorted { $0.confidence > $1.confidence }
                        .prefix(maxLabels)
                        .map { $0.identifier.replacingOccurrences(of: "_", with: " ") }
                    cont.resume(returning: Array(labels))
                } catch {
                    cont.resume(returning: [])
                }
            }
        }
    }

    /// Text stand-in for an image when the loaded model has no vision encoder.
    public static func describeImageAsText(_ image: CGImage, fileName: String) async -> String {
        let ocr = (try? await recognizeText(in: image)) ?? ""
        let labels = await classify(image)
        var parts: [String] = []
        if !labels.isEmpty { parts.append("Detected content: \(labels.joined(separator: ", ")).") }
        if !ocr.isEmpty { parts.append("Text in image:\n\(ocr)") }
        if parts.isEmpty { parts.append("(no text or recognizable content detected)") }
        return "[Image \"\(fileName)\", \(image.width)x\(image.height)]\n" + parts.joined(separator: "\n")
    }
}
