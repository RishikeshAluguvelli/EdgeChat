// swift-tools-version:5.9
import PackageDescription

// EdgeChatCore: the on-device inference engine (llama.cpp + mtmd), turn management,
// and attachment (document / image) processing. Shared by the iOS app and the macOS CLI.
let package = Package(
    name: "EdgeChat",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [
        .library(name: "EdgeChatCore", targets: ["EdgeChatCore"]),
        .executable(name: "edgechat-cli", targets: ["EdgeChatCLI"]),
    ],
    targets: [
        // Prebuilt llama.cpp (Metal + mtmd). Fetch with scripts/setup-llama.sh
        .binaryTarget(name: "llama", path: "Frameworks/llama.xcframework"),
        .target(
            name: "EdgeChatCore",
            dependencies: ["llama"],
            linkerSettings: [
                .linkedLibrary("c++"),
                .linkedFramework("Metal"),
                .linkedFramework("Accelerate"),
                .linkedFramework("Vision"),
                .linkedFramework("PDFKit"),
            ]
        ),
        .executableTarget(name: "EdgeChatCLI", dependencies: ["EdgeChatCore"]),
        .testTarget(name: "EdgeChatCoreTests", dependencies: ["EdgeChatCore"]),
    ]
)
