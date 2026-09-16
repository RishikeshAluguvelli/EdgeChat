import SwiftUI
import EdgeChatCore

@main
struct EdgeChatApp: App {
    @UIApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
    @Environment(\.scenePhase) private var scenePhase
    @State private var app = AppModel()

    var body: some Scene {
        WindowGroup {
            RootView()
                .environment(app)
                .onOpenURL { url in app.models.importModel(from: url) }
                .task { await app.start() }
        }
        .onChange(of: scenePhase) { _, phase in
            if phase == .background { Task { await app.engine.saveCurrentSnapshot() } }
        }
    }
}

final class AppDelegate: NSObject, UIApplicationDelegate {
    /// Lets background downloads finish even if the app was suspended.
    func application(_ application: UIApplication, handleEventsForBackgroundURLSession identifier: String,
                     completionHandler: @escaping () -> Void) {
        DownloadManager.backgroundCompletionHandler = completionHandler
    }
}
