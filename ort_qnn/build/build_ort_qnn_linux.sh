#!/usr/bin/env bash
set -euo pipefail

: "${QAIRT_SDK_ROOT:?Set QAIRT_SDK_ROOT}"
: "${ORT_SRC_ROOT:?Set ORT_SRC_ROOT}"
: "${PYTHON310:?Set PYTHON310}"
: "${ORT_BUILD_DIR:?Set ORT_BUILD_DIR}"

cd "$ORT_SRC_ROOT"
"$PYTHON310" tools/ci_build/build.py \
  --config Release \
  --build_shared_lib \
  --build_wheel \
  --parallel \
  --use_qnn \
  --qnn_home "$QAIRT_SDK_ROOT" \
  --build_dir "$ORT_BUILD_DIR"
