#!/usr/bin/env bash
# Checks out llama.cpp (same release as the iOS xcframework) for the Android NDK build.
set -euo pipefail
TAG="${LLAMA_TAG:-b10988}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/android/third_party/llama.cpp"
if [[ -d "$DEST/.git" ]]; then
  echo "llama.cpp already present at $DEST"; exit 0
fi
mkdir -p "$(dirname "$DEST")"
git clone --depth 1 --branch "$TAG" https://github.com/ggml-org/llama.cpp.git "$DEST"
echo "llama.cpp $TAG ready at $DEST"
