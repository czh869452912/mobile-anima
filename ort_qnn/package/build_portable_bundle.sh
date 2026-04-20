#!/usr/bin/env bash
set -euo pipefail

: "${ORT_RELEASE_ROOT:?Set ORT_RELEASE_ROOT}"
: "${PORTABLE_OUT_DIR:?Set PORTABLE_OUT_DIR}"

rm -rf "$PORTABLE_OUT_DIR"
mkdir -p "$PORTABLE_OUT_DIR"
cp -a "$ORT_RELEASE_ROOT/onnxruntime" "$PORTABLE_OUT_DIR/"
