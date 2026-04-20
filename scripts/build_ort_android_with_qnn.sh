#!/usr/bin/env bash
set -euo pipefail

: "${QNN_SDK_ROOT:?Set QNN_SDK_ROOT}"
: "${ANDROID_SDK_ROOT:?Set ANDROID_SDK_ROOT}"
: "${ANDROID_NDK_HOME:?Set ANDROID_NDK_HOME}"
: "${ORT_SRC_ROOT:?Set ORT_SRC_ROOT}"

cd "$ORT_SRC_ROOT"
./build.sh \
  --android \
  --android_abi arm64-v8a \
  --android_api 29 \
  --android_sdk_path "$ANDROID_SDK_ROOT" \
  --android_ndk_path "$ANDROID_NDK_HOME" \
  --build_shared_lib \
  --config Release \
  --use_qnn static_lib \
  --qnn_home "$QNN_SDK_ROOT" \
  --build_java
