#!/usr/bin/env bash
# Fetches the prebuilt llama.cpp xcframework (pinned build) into Frameworks/llama.xcframework.
# The official release ships iOS-device + macOS slices. Pass --with-simulator to also build the
# iOS Simulator slice from source (same tag) and merge it in, so the app can run in the Simulator.
#
#   scripts/setup-llama.sh                  # device + macOS (fast, ~60 MB download)
#   scripts/setup-llama.sh --with-simulator # + builds simulator slice (needs cmake, ~10 min)
set -euo pipefail

LLAMA_BUILD="${LLAMA_BUILD:-b10988}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FW="$ROOT/Frameworks"
mkdir -p "$FW"

ZIP="$FW/llama-$LLAMA_BUILD-xcframework.zip"
if [[ ! -f "$ZIP" ]]; then
  echo "Downloading llama.cpp $LLAMA_BUILD xcframework..."
  curl -L --fail --progress-bar -o "$ZIP" \
    "https://github.com/ggml-org/llama.cpp/releases/download/$LLAMA_BUILD/llama-$LLAMA_BUILD-xcframework.zip"
fi

rm -rf "$FW/llama.xcframework" "$FW/_unzip"
unzip -q "$ZIP" -d "$FW/_unzip"
mv "$FW/_unzip/build-apple/llama.xcframework" "$FW/llama.xcframework"
rm -rf "$FW/_unzip"
echo "Installed $FW/llama.xcframework (slices: $(ls "$FW/llama.xcframework" | grep -v Info.plist | tr '\n' ' '))"

if [[ "${1:-}" == "--with-simulator" ]]; then
  command -v cmake >/dev/null || { echo "cmake is required: brew install cmake"; exit 1; }
  SRC="$FW/build-llama-src"
  if [[ ! -d "$SRC" ]]; then
    git clone --depth 1 --branch "$LLAMA_BUILD" https://github.com/ggml-org/llama.cpp.git "$SRC"
  fi
  echo "Building iOS Simulator slice from source ($LLAMA_BUILD)..."
  (cd "$SRC" && ./build-xcframework.sh ios-sim)
  SIM_FW="$(find "$SRC/build-apple/llama.xcframework" -maxdepth 1 -type d -name 'ios-*simulator' | head -1)/llama.framework"
  [[ -d "$SIM_FW" ]] || { echo "Simulator framework not found under $SRC/build-apple"; exit 1; }
  xcodebuild -create-xcframework \
    -framework "$FW/llama.xcframework/ios-arm64/llama.framework" \
    -framework "$FW/llama.xcframework/macos-arm64_x86_64/llama.framework" \
    -framework "$SIM_FW" \
    -output "$FW/llama-merged.xcframework"
  rm -rf "$FW/llama.xcframework"
  mv "$FW/llama-merged.xcframework" "$FW/llama.xcframework"
  echo "Merged simulator slice. Slices: $(ls "$FW/llama.xcframework" | grep -v Info.plist | tr '\n' ' ')"
fi
