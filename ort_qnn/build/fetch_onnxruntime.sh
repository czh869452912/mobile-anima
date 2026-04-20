#!/usr/bin/env bash
set -euo pipefail

TAG="${1:-v1.23.2}"
TARGET_DIR="${2:-ort_qnn/artifacts/ort/onnxruntime}"

if [ ! -d "$TARGET_DIR/.git" ]; then
  git clone --branch "$TAG" --depth 1 https://github.com/microsoft/onnxruntime.git "$TARGET_DIR"
fi
