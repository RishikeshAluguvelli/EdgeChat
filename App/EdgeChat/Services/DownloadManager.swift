import Foundation

/// Background URLSession downloads that survive the app being suspended. Keys look like "<modelID>|model".
final class DownloadManager: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    nonisolated(unsafe) static var backgroundCompletionHandler: (() -> Void)?

    var onProgress: (@Sendable (String, Int64, Int64) -> Void)?
    var onFinished: (@Sendable (String, Result<URL, Error>) -> Void)?
    var onPaused: (@Sendable (String) -> Void)?

    private lazy var session: URLSession = {
        let config = URLSessionConfiguration.background(withIdentifier: "com.rishikesh.edgechat.downloads")
        config.isDiscretionary = false
        config.sessionSendsLaunchEvents = true
        config.timeoutIntervalForRequest = 120
        return URLSession(configuration: config, delegate: self, delegateQueue: nil)
    }()

    private let lock = NSLock()
    private var tasks: [String: URLSessionDownloadTask] = [:]
    private var resumeData: [String: Data] = [:]
    private var pausing: Set<String> = []

    struct DownloadError: LocalizedError { let message: String; var errorDescription: String? { message } }

    /// Re-attaches to downloads still running from a previous launch.
    func reattach(completion: @escaping @Sendable ([String]) -> Void) {
        session.getAllTasks { tasks in
            var keys: [String] = []
            self.lock.withLock {
                for task in tasks {
                    guard let dl = task as? URLSessionDownloadTask, let key = Self.key(of: dl) else { continue }
                    self.tasks[key] = dl
                    keys.append(key)
                }
            }
            completion(keys)
        }
    }

    func start(key: String, url: URL, destination: URL) {
        lock.withLock {
            guard tasks[key] == nil else { return }
            let task: URLSessionDownloadTask
            if let data = resumeData.removeValue(forKey: key) {
                task = session.downloadTask(withResumeData: data)
            } else {
                var request = URLRequest(url: url)
                request.setValue("EdgeChat/1.0 (iOS; llama.cpp)", forHTTPHeaderField: "User-Agent")
                task = session.downloadTask(with: request)
            }
            task.taskDescription = key + "\n" + destination.path
            tasks[key] = task
            task.resume()
        }
    }

    func pause(key: String) {
        lock.withLock {
            guard let task = tasks[key] else { return }
            pausing.insert(key)
            task.cancel { [weak self] data in
                guard let self else { return }
                self.lock.withLock {
                    if let data { self.resumeData[key] = data }
                    self.tasks[key] = nil
                    self.pausing.remove(key)
                }
                self.onPaused?(key)
            }
        }
    }

    func cancel(key: String) {
        lock.withLock {
            resumeData[key] = nil
            pausing.insert(key) // suppress the failure callback
            tasks[key]?.cancel()
            tasks[key] = nil
        }
    }

    func isActive(key: String) -> Bool { lock.withLock { tasks[key] != nil } }

    private static func key(of task: URLSessionTask) -> String? {
        task.taskDescription?.split(separator: "\n", maxSplits: 1).first.map(String.init)
    }
    private static func destination(of task: URLSessionTask) -> URL? {
        guard let parts = task.taskDescription?.split(separator: "\n", maxSplits: 1), parts.count == 2 else { return nil }
        return URL(fileURLWithPath: String(parts[1]))
    }

    // MARK: URLSessionDownloadDelegate

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didWriteData bytesWritten: Int64,
                    totalBytesWritten: Int64, totalBytesExpectedToWrite: Int64) {
        guard let key = Self.key(of: downloadTask) else { return }
        onProgress?(key, totalBytesWritten, totalBytesExpectedToWrite)
    }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didFinishDownloadingTo location: URL) {
        guard let key = Self.key(of: downloadTask), let destination = Self.destination(of: downloadTask) else { return }
        let status = (downloadTask.response as? HTTPURLResponse)?.statusCode ?? 200
        guard (200..<300).contains(status) else {
            lock.withLock { tasks[key] = nil }
            onFinished?(key, .failure(DownloadError(message: status == 401 || status == 403
                ? "The model repository requires a Hugging Face login (HTTP \(status))." : "Download failed (HTTP \(status)).")))
            return
        }
        do {
            try? FileManager.default.removeItem(at: destination)
            try FileManager.default.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
            try FileManager.default.moveItem(at: location, to: destination)
            var values = URLResourceValues(); values.isExcludedFromBackup = true
            var dest = destination; try? dest.setResourceValues(values)
            lock.withLock { tasks[key] = nil }
            onFinished?(key, .success(destination))
        } catch {
            lock.withLock { tasks[key] = nil }
            onFinished?(key, .failure(error))
        }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        guard let error, let key = Self.key(of: task) else { return }
        let wasPausing = lock.withLock { pausing.remove(key) != nil }
        lock.withLock {
            tasks[key] = nil
            if let data = (error as NSError).userInfo[NSURLSessionDownloadTaskResumeData] as? Data { resumeData[key] = data }
        }
        if wasPausing { return }
        onFinished?(key, .failure(error))
    }

    func urlSessionDidFinishEvents(forBackgroundURLSession session: URLSession) {
        DispatchQueue.main.async {
            Self.backgroundCompletionHandler?()
            Self.backgroundCompletionHandler = nil
        }
    }
}
