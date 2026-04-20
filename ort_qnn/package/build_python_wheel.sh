#!/usr/bin/env bash
set -euo pipefail

: "${ORT_RELEASE_ROOT:?Set ORT_RELEASE_ROOT}"
: "${WHEEL_STAGE_DIR:?Set WHEEL_STAGE_DIR}"
: "${WHEEL_OUT_DIR:?Set WHEEL_OUT_DIR}"
: "${PYTHON310:?Set PYTHON310}"

rm -rf "$WHEEL_STAGE_DIR"
mkdir -p "$WHEEL_STAGE_DIR/src" "$WHEEL_OUT_DIR"
cp -a "$ORT_RELEASE_ROOT/onnxruntime" "$WHEEL_STAGE_DIR/src/"
"$PYTHON310" - <<'PY' "$WHEEL_STAGE_DIR/src/onnxruntime/capi"
from pathlib import Path
import shutil
import sys
import sysconfig

capi = Path(sys.argv[1])
plain = capi / 'onnxruntime_pybind11_state.so'
suffix = sysconfig.get_config_var('EXT_SUFFIX') or '.so'
abi = capi / f'onnxruntime_pybind11_state{suffix}'
if plain.exists() and abi.name != plain.name:
    shutil.copy2(plain, abi)
(capi / 'build_and_package_info.py').write_text("package_name = 'onnxruntime-qnn-local'\n__version__ = '1.23.2'\n")
PY
cat > "$WHEEL_STAGE_DIR/pyproject.toml" <<'TOML'
[build-system]
requires = ["setuptools>=69", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "onnxruntime-qnn-local"
version = "1.23.2"
requires-python = ">=3.10,<3.11"
TOML
cat > "$WHEEL_STAGE_DIR/setup.py" <<'PY'
from setuptools import find_packages, setup

setup(
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    package_data={"onnxruntime": ["capi/*", "capi/training/*"]},
)
PY
"$PYTHON310" -m pip install --upgrade build wheel
"$PYTHON310" -m build --wheel --outdir "$WHEEL_OUT_DIR" "$WHEEL_STAGE_DIR"
