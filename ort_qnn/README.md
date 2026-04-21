# Desktop ORT + QNN EP Validation

This subproject validates the last missing desktop layer in the smoke ladder:

1. acquire or build ORT with QNN EP
2. verify `QNNExecutionProvider` visibility
3. create a `QNN EP` session for `toy.onnx`
4. execute `toy.onnx`
5. repeat for `mini_block.onnx`

## First Manual Run Order

1. Check for a usable Linux prebuilt `ORT + QNN EP`
2. If not available, fetch ORT source
3. Build Linux desktop ORT Python wheel with `QNN EP`
4. Confirm `QNNExecutionProvider` is visible
5. Run `toy.onnx` session creation
6. Run `toy.onnx` execution
7. Run `mini_block.onnx` session creation
8. Run `mini_block.onnx` execution
9. Update `ort_qnn/docs/ort-qnn-matrix.md`

## Reusable Outputs

This subproject now supports two reuse paths:

- **Portable bundle** — a copied runtime tree that is imported directly via `PYTHONPATH`
- **Local wheel** — a Python-installable package produced from the validated build tree

## Build Portable Bundle

```bash
export ORT_RELEASE_ROOT=/project/mobile_anima_npu/ort_qnn/artifacts/ort/build/linux_qnn/Release
export PORTABLE_OUT_DIR=/project/mobile_anima_npu/ort_qnn/artifacts/package/portable
bash ort_qnn/package/build_portable_bundle.sh
```

Verify the bundle:

```bash
PYTHONPATH=/project/mobile_anima_npu/ort_qnn/artifacts/package/portable:/project/mobile_anima_npu \
LD_LIBRARY_PATH=/project/mobile_anima_npu/ort_qnn/artifacts/package/portable/onnxruntime/capi:/opt/python3.10.19/lib:/opt/qcom/aistack/qairt/2.41.0.251128/lib/x86_64-linux-clang \
/opt/qcom/qairt-py310/bin/python - <<'PY'
import onnxruntime as ort
print(ort.get_available_providers())
PY
```

Then validate `toy` and `mini_block` with the existing runtime helpers.

## Build Local Wheel

```bash
export ORT_RELEASE_ROOT=/project/mobile_anima_npu/ort_qnn/artifacts/ort/build/linux_qnn/Release
export WHEEL_STAGE_DIR=/project/mobile_anima_npu/ort_qnn/artifacts/package/wheel_staging
export WHEEL_OUT_DIR=/project/mobile_anima_npu/ort_qnn/artifacts/package/wheels
export PYTHON310=/opt/qcom/qairt-py310/bin/python
bash ort_qnn/package/build_python_wheel.sh
```

Install and verify:

```bash
WHEEL=$(find ort_qnn/artifacts/package/wheels -name 'onnxruntime_qnn_local-*.whl' | head -n 1)
/opt/qcom/qairt-py310/bin/python -m pip install --force-reinstall "$WHEEL"

PYTHONPATH=/project/mobile_anima_npu \
LD_LIBRARY_PATH=/opt/python3.10.19/lib:/opt/qcom/qairt-py310/lib/python3.10/site-packages/onnxruntime/capi:/opt/qcom/aistack/qairt/2.41.0.251128/lib/x86_64-linux-clang \
/opt/qcom/qairt-py310/bin/python - <<'PY'
import onnxruntime as ort
print(ort.get_available_providers())
PY
```

## Expected Failure Classes

When packaging fails, classify it as one of:

- install failure
- provider unavailable after install
- session creation failure
- execute failure

Do not leave wheel problems as a vague “it failed”.

## Bootstrap a Stronger Host

For a fresh Ubuntu 24.04 machine:

```bash
sudo bash scripts/bootstrap_ubuntu2404.sh
```

If `qpm-cli` is not detected, the script completes public prerequisites, tells the user what is missing, and asks them to rerun it after manual `qpm-cli` installation and login.
