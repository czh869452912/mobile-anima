# Mobile Anima NPU

Experimental workspace for bringing an `Anima` denoiser pipeline onto Qualcomm NPU-capable targets.

The repo is intentionally split into small validation ladders so toolchain risk is proven in the cheapest order before touching the real model.

## Current Status

- `android/`: Android proof-of-concept shell and local host-side checks are in place.
- `python/`: shared Python-side utilities and tests for model/export support are in place.
- `smoke/`: toy and mini-block graphs have already passed CPU `ONNX Runtime`, `qnn-onnx-converter`, `qnn-model-lib-generator`, `qnn-context-binary-generator`, and `qnn-net-run` on the host CPU backend.
- `ort_qnn/`: a Linux source build of `ONNX Runtime 1.23.2` with `QNNExecutionProvider` is now importable from the build tree, and both `toy.onnx` and `mini_block.onnx` pass `provider -> session -> execute` with profiling output.
- `ort_qnn/package/`: packaging helpers now produce both a portable runtime bundle and a local wheel path for the validated desktop `ORT + QNN` build.
- `scripts/bootstrap_ubuntu2404.sh`: prepares a stronger Ubuntu 24.04 host for desktop ORT/QNN work and Android base tooling, then optionally continues into QAIRT setup if `qpm-cli` is already installed and logged in.

## Repository Layout

- `android/` — Android app shell for the eventual on-device integration path.
- `python/` — Python package, helpers, and tests shared by local tooling.
- `smoke/` — cheapest-order QNN smoke tests for `toy` and `mini_block` graphs.
- `ort_qnn/` — desktop `ONNX Runtime + QNN Execution Provider` acquisition, runtime helpers, and validation docs.
- `docs/superpowers/` — design specs and implementation plans captured during execution.
- `scripts/` — one-off host/build helper scripts.

## Stronger Host Setup

For a new Ubuntu 24.04 machine, start with:

```bash
sudo bash scripts/bootstrap_ubuntu2404.sh
```

The script intentionally works in two phases:

- if `qpm-cli` is not installed yet, it completes public prerequisites, explains the missing Qualcomm step, and exits cleanly
- after you install/login `qpm-cli`, rerun the same script and it continues into `QAIRT` setup

## Packaging Outputs

The validated desktop ORT/QNN build can now be reused in two forms:

- **Portable bundle** — lives under `ort_qnn/artifacts/package/portable` and is the most robust reuse path
- **Local wheel path** — lives under `ort_qnn/artifacts/package/wheels` and supports `pip install` into a Python 3.10 environment

Recommended order on a fresh machine:

1. run `scripts/bootstrap_ubuntu2404.sh`
2. use the portable bundle first to confirm provider visibility and `toy` / `mini_block`
3. use the wheel path when you want a cleaner Python install workflow

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
- `docs/superpowers/specs/2026-04-20-ort-qnn-packaging-and-bootstrap-design.md:1`
- `docs/superpowers/specs/2026-04-20-anima-npu-poc-design.md:1`
- `docs/superpowers/specs/2026-04-20-qnn-smoke-test-mini-block-design.md:1`
- `docs/superpowers/specs/2026-04-20-desktop-ort-qnn-validation-design.md:1`

## Host Notes

- QAIRT currently lives under `/opt/qcom/aistack/qairt/2.41.0.251128` on this machine.
- The QAIRT Python tooling depends on Python 3.10; the local helper environment is under `/opt/qcom/qairt-py310`.
- `ninja` is installed and the desktop ORT/QNN source build lives under `ort_qnn/artifacts/ort/build/linux_qnn/Release`.
- Direct `PYTHONPATH` import from that build tree already exposes `QNNExecutionProvider`.
- A portable bundle and a local wheel path can now be built from that validated release tree.

## Next Milestone

Use the now-validated and now-packageable desktop ORT/QNN path on a stronger host, then move into real `Anima` denoiser export and operator-gap analysis.
