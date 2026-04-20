# Mobile Anima NPU

Experimental workspace for bringing an `Anima` denoiser pipeline onto Qualcomm NPU-capable targets.

The repo is intentionally split into small validation ladders so toolchain risk is proven in the cheapest order before touching the real model.

## Current Status

- `android/`: Android proof-of-concept shell and local host-side checks are in place.
- `python/`: shared Python-side utilities and tests for model/export support are in place.
- `smoke/`: toy and mini-block graphs have already passed CPU `ONNX Runtime`, `qnn-onnx-converter`, `qnn-model-lib-generator`, `qnn-context-binary-generator`, and `qnn-net-run` on the host CPU backend.
- `ort_qnn/`: a Linux source build of `ONNX Runtime 1.23.2` with `QNNExecutionProvider` is now importable from the build tree, and both `toy.onnx` and `mini_block.onnx` pass `provider -> session -> execute` with profiling output.

## Repository Layout

- `android/` — Android app shell for the eventual on-device integration path.
- `python/` — Python package, helpers, and tests shared by local tooling.
- `smoke/` — cheapest-order QNN smoke tests for `toy` and `mini_block` graphs.
- `ort_qnn/` — desktop `ONNX Runtime + QNN Execution Provider` acquisition, runtime helpers, and validation docs.
- `docs/superpowers/` — design specs and implementation plans captured during execution.
- `scripts/` — one-off host/build helper scripts.

## Validation Ladder

1. Prove ONNX export and CPU runtime correctness on minimal graphs.
2. Prove QAIRT/QNN conversion, model-lib generation, context generation, and host runtime execution.
3. Prove desktop Python-callable `ONNX Runtime + QNNExecutionProvider` session and execute paths.
4. Only after the first three stages are green, move to the real `Anima` denoiser.

## Key Docs

- `smoke/README.md` — smoke test purpose and first-run order.
- `smoke/docs/smoke-matrix.md` — current smoke validation evidence.
- `ort_qnn/README.md` — desktop ORT + QNN validation flow.
- `ort_qnn/docs/ort-qnn-matrix.md` — provider/session/execute result matrix.
- `docs/superpowers/specs/2026-04-20-anima-npu-poc-design.md:1`
- `docs/superpowers/specs/2026-04-20-qnn-smoke-test-mini-block-design.md:1`
- `docs/superpowers/specs/2026-04-20-desktop-ort-qnn-validation-design.md:1`

## Host Notes

- QAIRT currently lives under `/opt/qcom/aistack/qairt/2.41.0.251128` on this machine.
- The QAIRT Python tooling depends on Python 3.10; the local helper environment is under `/opt/qcom/qairt-py310`.
- `ninja` is installed and the desktop ORT/QNN source build lives under `ort_qnn/artifacts/ort/build/linux_qnn/Release`.
- Direct `PYTHONPATH` import from that build tree already exposes `QNNExecutionProvider`.

## Next Milestone

Use the now-validated desktop ORT/QNN path to decide the next model step: either package the build into a reusable wheel/install flow, or start the real `Anima` denoiser integration and operator-gap analysis.
