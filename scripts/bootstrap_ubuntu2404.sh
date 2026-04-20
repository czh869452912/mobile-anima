#!/usr/bin/env bash
set -euo pipefail

QAIRT_PACKAGE="${QAIRT_PACKAGE:-qualcomm_ai_runtime_sdk}"
QAIRT_VERSION="${QAIRT_VERSION:-2.41.0.251128}"
QAIRT_ROOT="${QAIRT_ROOT:-/opt/qcom/aistack/qairt/${QAIRT_VERSION}}"
ANDROID_SDK_ROOT="${ANDROID_SDK_ROOT:-/opt/android-sdk}"
ANDROID_CMDLINE_TOOLS_URL="${ANDROID_CMDLINE_TOOLS_URL:-https://dl.google.com/android/repository/commandlinetools-linux-11076708_latest.zip}"

log() {
  printf '[bootstrap] %s\n' "$*"
}

require_ubuntu_2404() {
  if [[ ! -f /etc/os-release ]]; then
    log "Unable to detect OS; this script targets Ubuntu 24.04 only."
    exit 1
  fi

  . /etc/os-release
  if [[ "${ID:-}" != "ubuntu" || "${VERSION_ID:-}" != "24.04" ]]; then
    log "This script targets Ubuntu 24.04 only. Detected ${ID:-unknown} ${VERSION_ID:-unknown}."
    exit 1
  fi
}

install_public_prereqs() {
  log "Installing Ubuntu 24.04 build prerequisites, Android base packages, and Python 3.10 build dependencies"
  apt-get update
  apt-get install -y \
    build-essential clang clang++ cmake ninja-build git curl unzip zip pkg-config ca-certificates \
    openjdk-21-jdk-headless \
    libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev libffi-dev liblzma-dev libncursesw5-dev xz-utils tk-dev uuid-dev \
    libc++1 libc++abi1
}

prepare_android_base() {
  log "Preparing Android base environment"
  mkdir -p "${ANDROID_SDK_ROOT}/cmdline-tools"

  if [[ ! -x "${ANDROID_SDK_ROOT}/cmdline-tools/latest/bin/sdkmanager" ]]; then
    local tmp_zip
    tmp_zip="$(mktemp /tmp/android-cmdline-tools-XXXXXX.zip)"
    curl -L "${ANDROID_CMDLINE_TOOLS_URL}" -o "${tmp_zip}"
    rm -rf "${ANDROID_SDK_ROOT}/cmdline-tools/latest"
    mkdir -p "${ANDROID_SDK_ROOT}/cmdline-tools/latest"
    unzip -q "${tmp_zip}" -d "${ANDROID_SDK_ROOT}/cmdline-tools/latest.tmp"
    if [[ -d "${ANDROID_SDK_ROOT}/cmdline-tools/latest.tmp/cmdline-tools" ]]; then
      mv "${ANDROID_SDK_ROOT}/cmdline-tools/latest.tmp/cmdline-tools"/* "${ANDROID_SDK_ROOT}/cmdline-tools/latest/"
    else
      mv "${ANDROID_SDK_ROOT}/cmdline-tools/latest.tmp"/* "${ANDROID_SDK_ROOT}/cmdline-tools/latest/"
    fi
    rm -rf "${ANDROID_SDK_ROOT}/cmdline-tools/latest.tmp" "${tmp_zip}"
  fi

  export ANDROID_SDK_ROOT
  export PATH="${ANDROID_SDK_ROOT}/cmdline-tools/latest/bin:${ANDROID_SDK_ROOT}/platform-tools:${PATH}"
  yes | sdkmanager --licenses >/dev/null || true
  sdkmanager "platform-tools" "platforms;android-35" "build-tools;35.0.0"
  log "Android SDK root prepared at ${ANDROID_SDK_ROOT}"
}

report_python310_hint() {
  log "Python 3.10 runtime is required for QAIRT tooling."
  log "If /opt/python3.10.19 or /opt/qcom/qairt-py310 do not exist yet, build or recreate them before QAIRT-dependent steps."
}

handle_qpm_and_qairt() {
  if ! command -v qpm-cli >/dev/null 2>&1; then
    log "qpm-cli not detected. Public prerequisites are complete."
    log "Install qpm-cli, complete login, then rerun this script to finish QAIRT setup."
    exit 0
  fi

  if ! qpm-cli --check-login >/dev/null 2>&1; then
    log "qpm-cli is installed but not logged in. Complete login, then rerun this script."
    exit 0
  fi

  log "qpm-cli detected and logged in; continuing QAIRT setup"
  qpm-cli --license-activate "${QAIRT_PACKAGE}" || true
  qpm-cli --install "${QAIRT_PACKAGE}" --version "${QAIRT_VERSION}"

  if [[ ! -d "${QAIRT_ROOT}" ]]; then
    log "QAIRT install command finished but ${QAIRT_ROOT} was not created. Check entitlement/version and rerun."
    exit 1
  fi

  log "QAIRT installed at ${QAIRT_ROOT}"
}

main() {
  require_ubuntu_2404
  install_public_prereqs
  prepare_android_base
  report_python310_hint
  handle_qpm_and_qairt
}

main "$@"
