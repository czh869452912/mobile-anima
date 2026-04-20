# Desktop ORT + QNN Real Build Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans or superpowers:subagent-driven-development to execute this plan with explicit checkpoints.

**Goal:** Produce a Python-callable Linux `ONNX Runtime` build with `QNNExecutionProvider`, then validate provider visibility, `toy.onnx`, and `mini_block.onnx` on this host.

**Scope:** This plan covers the real host execution path only. It does not add Android work, model changes, or performance tuning.

**Existing Preconditions:**

- QAIRT exists at `/opt/qcom/aistack/qairt/2.41.0.251128`
- QAIRT Python 3.10 venv exists at `/opt/qcom/qairt-py310`
- `smoke/artifacts/onnx/toy.onnx` and `smoke/artifacts/onnx/mini_block.onnx` already exist
- `ort_qnn/` helper scripts and tests are already green

## Task 1: Prepare host prerequisites and short prebuilt probe

**Files:**
- No repo file changes expected

- [ ] Confirm missing build prerequisites (`ninja`, compiler tools, Python headers if needed)
- [ ] Install `ninja-build` on the host if it is still missing
- [ ] Perform one short Linux prebuilt probe for a Python-callable `ORT + QNN EP`
- [ ] If no usable prebuilt exists, record that source build is the active path

Run:

```bash
command -v ninja || true
/opt/qcom/qairt-py310/bin/python -m pip index versions onnxruntime-qnn || true
apt-get update
apt-get install -y ninja-build
```

Expected:

- `ninja` is available afterwards
- no usable Linux `onnxruntime-qnn` prebuilt is identified, or the result is clearly non-actionable

## Task 2: Fetch ONNX Runtime source and launch QNN-enabled Linux build

**Files:**
- Build outputs only under `ort_qnn/artifacts/ort/`
- Logs under `ort_qnn/artifacts/logs/`

- [ ] Fetch pinned ORT source under `ort_qnn/artifacts/ort/onnxruntime`
- [ ] Export `QAIRT_SDK_ROOT`, `ORT_SRC_ROOT`, `PYTHON310`, and `ORT_BUILD_DIR`
- [ ] Run `ort_qnn/build/build_ort_qnn_linux.sh` and save the full log
- [ ] Identify the produced wheel or Python package location

Run:

```bash
bash ort_qnn/build/fetch_onnxruntime.sh v1.23.2 ort_qnn/artifacts/ort/onnxruntime
export QAIRT_SDK_ROOT=/opt/qcom/aistack/qairt/2.41.0.251128
export ORT_SRC_ROOT=/project/mobile_anima_npu/ort_qnn/artifacts/ort/onnxruntime
export PYTHON310=/opt/qcom/qairt-py310/bin/python
export ORT_BUILD_DIR=/project/mobile_anima_npu/ort_qnn/artifacts/ort/build/linux_qnn
bash ort_qnn/build/build_ort_qnn_linux.sh 2>&1 | tee ort_qnn/artifacts/logs/build_ort_qnn_linux.log
```

Expected:

- ORT source exists locally
- build completes or fails with a concrete error log
- a wheel appears under the build tree if the build succeeds

## Task 3: Install the built wheel and validate provider visibility

**Files:**
- Runtime logs under `ort_qnn/artifacts/logs/`

- [ ] Install the built wheel into `/opt/qcom/qairt-py310`
- [ ] Confirm `import onnxruntime as ort` works in that environment
- [ ] Confirm `QNNExecutionProvider` appears in `ort.get_available_providers()`

Run:

```bash
WHEEL=$(find ort_qnn/artifacts/ort/build/linux_qnn -name 'onnxruntime-*.whl' | head -n 1)
/opt/qcom/qairt-py310/bin/python -m pip install --force-reinstall "$WHEEL"
/opt/qcom/qairt-py310/bin/python - <<'PY' 2>&1 | tee ort_qnn/artifacts/logs/provider_visibility.log
import onnxruntime as ort
print("providers=", ort.get_available_providers())
PY
```

Expected:

- wheel installs cleanly
- Python import works
- `QNNExecutionProvider` is visible, or the failure is explicit and logged

## Task 4: Run `toy` then `mini_block` through Python ORT + QNN EP

**Files:**
- Runtime logs under `ort_qnn/artifacts/logs/`
- Profiles under `ort_qnn/artifacts/profiles/`

- [ ] Run `toy.onnx` first
- [ ] Stop immediately if provider visibility or session creation fails
- [ ] If `toy` passes, run `mini_block.onnx`
- [ ] Preserve logs for both runs

Run:

```bash
/opt/qcom/qairt-py310/bin/python ort_qnn/runtime/run_toy_ort_qnn.py \
  --model smoke/artifacts/onnx/toy.onnx \
  --backend /opt/qcom/aistack/qairt/2.41.0.251128/lib/x86_64-linux-clang/libQnnCpu.so \
  --profile ort_qnn/artifacts/profiles/toy_ort_qnn.csv \
  2>&1 | tee ort_qnn/artifacts/logs/toy_ort_qnn.log

/opt/qcom/qairt-py310/bin/python ort_qnn/runtime/run_mini_block_ort_qnn.py \
  --model smoke/artifacts/onnx/mini_block.onnx \
  --backend /opt/qcom/aistack/qairt/2.41.0.251128/lib/x86_64-linux-clang/libQnnCpu.so \
  --profile ort_qnn/artifacts/profiles/mini_block_ort_qnn.csv \
  2>&1 | tee ort_qnn/artifacts/logs/mini_block_ort_qnn.log
```

Expected:

- `toy` reaches either success or a clearly classified stage failure
- `mini_block` runs only if `toy` proves the path is alive

## Task 5: Record outcomes and run verification

**Files:**
- Modify: `ort_qnn/docs/ort-qnn-matrix.md`
- Modify: `README.md` if the host notes or milestone status changed materially

- [ ] Update the matrix with acquisition path, ORT build identity, provider visibility, session result, execute result, and failing stage
- [ ] Update top-level status notes only if the build materially changed repo state
- [ ] Run the `ort_qnn` test suite again
- [ ] Check `git status --short`

Run:

```bash
python3 -m pytest ort_qnn/tests -q
git status --short
```

Expected:

- matrix reflects the actual host result
- tests remain green
- uncommitted changes are limited to intentional docs updates
