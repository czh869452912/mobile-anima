# ORT QNN Packaging and Bootstrap Design

## Goal

Finish the desktop `ORT + QNN` validation track with reusable packaging and reproducible environment setup so the workflow can be moved to a stronger Ubuntu 24.04 host with minimal manual reconstruction.

This stage has three deliverables:

1. a portable desktop `ORT + QNN` runtime bundle
2. a Python-installable `wheel/package` path for the same validated build
3. a host bootstrap script for Ubuntu 24.04 that prepares desktop and Android prerequisites, and conditionally continues into `QAIRT/QNN` setup when `qpm-cli` is already available

## Why This Stage Exists

The project has already proven the core technical risk on the current host:

- raw `QAIRT/QNN` compile and host runtime smoke tests work
- desktop source-built `ONNX Runtime 1.23.2` exposes `QNNExecutionProvider`
- both `toy.onnx` and `mini_block.onnx` pass `provider -> session -> execute`

The remaining gap is not whether `ORT + QNN` works. The remaining gap is whether that validated state can be:

- packaged for reuse
- documented cleanly
- recreated on a stronger machine without depending on chat history or ad hoc terminal notes

## Scope

This stage covers:

- packaging the validated desktop Linux `ORT + QNN` build as a portable runtime bundle
- attempting a Python-installable package or wheel from the same validated build
- adding a Ubuntu 24.04 bootstrap script for shared prerequisites
- detecting `qpm-cli` and continuing `QAIRT` installation only when it is already installed and usable
- documenting the packaging and bootstrap workflow in the repository README files

This stage does not cover:

- real `Anima` denoiser export
- Android runtime integration beyond base environment preparation
- performance tuning
- model quantization strategy work
- Qualcomm entitlement or account troubleshooting beyond explicit detection and guidance

## User Decisions Captured

- Deliver both a portable runtime bundle and a wheel/package path
- Target Ubuntu 24.04 first; broader Linux support can come later
- `qpm-cli` bootstrap behavior should be two-phase:
  - if not detected, explain how to install it and stop cleanly
  - if detected later, rerunning the same script should continue and fill in the remaining setup
- Android base environment should be included in bootstrap, but not full Android runtime validation

## Recommended Strategy

Use a layered packaging strategy.

### Layer 1: Portable Runtime Bundle

Package the already-proven desktop build into a directory that can be copied to a stronger host and used directly via `PYTHONPATH` and `LD_LIBRARY_PATH`.

### Layer 2: Wheel/Package Path

Build on top of the portable-bundle knowledge to attempt a Python-installable package. The first goal is practical installability, not polished distribution-grade packaging.

### Layer 3: Host Bootstrap

Provide a single Ubuntu 24.04 bootstrap entrypoint that prepares all public prerequisites first, then optionally completes `QAIRT` setup when `qpm-cli` is already installed and logged in.

This ordering keeps the lowest-risk path first and avoids blocking the entire stage on wheel packaging details.

## Architecture

### Directory Structure

- `ort_qnn/package/`
  - `build_portable_bundle.sh`
  - `build_python_wheel.sh`
  - `package_manifest.py`
  - optional `verify_package.py`
- `scripts/`
  - `bootstrap_ubuntu2404.sh`
- documentation
  - `README.md`
  - `ort_qnn/README.md`
  - optional `docs/manual/bootstrap-ubuntu2404.md`

### Responsibility Split

- `ort_qnn/package/` is responsible only for packaging and package verification
- `ort_qnn/runtime/` remains responsible only for provider/session/execute verification logic
- `scripts/bootstrap_ubuntu2404.sh` is responsible only for host preparation, dependency detection, and staged QAIRT installation
- repository docs explain how to use the outputs and how to recover from expected missing pieces such as absent `qpm-cli`

## Portable Runtime Bundle Design

The portable bundle should package the minimum validated runtime necessary to reproduce the desktop success path.

### Portable Bundle Contents

- the `onnxruntime` Python package tree from the validated build output
- `onnxruntime/capi/` shared libraries required for desktop QNN execution
- `libonnxruntime.so`
- `libonnxruntime_providers_qnn.so`
- required `QNN/QAIRT` runtime `.so` files needed by the desktop provider path
- a tiny manifest or metadata file describing:
  - ORT version
  - QAIRT version
  - source build location
  - expected environment variables

### Portable Bundle Success Definition

From a fresh shell on a prepared Ubuntu 24.04 host, the bundle must be sufficient to:

- import `onnxruntime`
- report `QNNExecutionProvider` in `get_available_providers()`
- run `toy.onnx`
- run `mini_block.onnx`

If the portable bundle cannot satisfy those conditions, stop and repair the packaging boundary before attempting to finalize a wheel.

## Wheel/Package Design

The wheel/package effort is intended to provide a cleaner installation path without redefining success for the whole stage.

### Wheel/Package Goal

Produce a Python-installable package artifact, ideally a wheel, that allows the validated desktop build to be installed into a Python 3.10 environment.

### Wheel/Package Minimum Success

- package installs in a clean Python 3.10 environment
- `import onnxruntime as ort` works
- `ort.get_available_providers()` includes `QNNExecutionProvider`
- `toy.onnx` and `mini_block.onnx` still pass

### Wheel/Package Fallback Rule

If wheel installation works poorly or provider visibility breaks after installation, the stage still succeeds as long as:

- the portable bundle is working
- the wheel failure is explicitly classified as one of:
  - install failure
  - provider missing after install
  - session creation failure
  - execute failure

This stage does not permit a vague “wheel did not work” outcome.

## Bootstrap Script Design

### Target Host

The bootstrap script will explicitly target Ubuntu 24.04.

### Public Prerequisites Covered

The script should prepare the host for both desktop ORT/QNN work and Android base tooling by installing or configuring:

- compiler and build tools
- Python 3.10 build/runtime prerequisites
- common archive and network utilities
- Android base prerequisites such as JDK, command-line SDK tooling, and required environment variables
- repository-local directory scaffolding if needed

### QAIRT/QPM Two-Phase Behavior

The script must detect `qpm-cli` and branch clearly:

#### Branch A: `qpm-cli` not detected

- do not fail with a cryptic error
- print a clear message explaining that public prerequisites are complete
- explain that the user must install `qpm-cli`, log in, and rerun the same script

#### Branch B: `qpm-cli` detected but not usable

Examples:

- not logged in
- missing entitlement
- install command fails

The script should stop cleanly and print what was detected, plus the next manual action required.

#### Branch C: `qpm-cli` detected and usable

- continue with `license-activate` and `QAIRT` install steps
- validate that the expected `QAIRT` directory exists afterward
- stop with a precise error if the install did not materialize the expected files

### Bootstrap Success Definition

A single script should be rerunnable.

- first run may stop after public prerequisites if `qpm-cli` is absent
- second run, after manual `qpm-cli` installation and login, should continue from detection and complete the remaining QAIRT setup

## Documentation Design

### Root README

The top-level `README.md` should become the entrypoint for a new host and cover:

- what the repo currently proves
- how to bootstrap a new Ubuntu 24.04 machine
- how to continue if `qpm-cli` is not yet installed
- where the portable bundle and wheel outputs live
- what to run next after bootstrap

### `ort_qnn/README.md`

The `ort_qnn` README should cover:

- how to build the portable runtime bundle
- how to build the wheel/package path
- how to verify provider visibility
- how to run `toy` and `mini_block`
- how to interpret expected failure classes

### Optional Manual Doc

If the root and subproject README become too crowded, move detailed Ubuntu bootstrap steps into `docs/manual/bootstrap-ubuntu2404.md` and keep both README files short and navigable.

## Execution Flow

### Phase 1: Package Portable Runtime Bundle

- derive the package contents from the validated source-build tree
- generate a portable directory output
- verify provider visibility, `toy`, and `mini_block` from that packaged directory

#### Stop-Loss 1

If the portable bundle cannot reproduce the already-proven results, fix the bundle before touching wheel logic.

### Phase 2: Build Wheel/Package Path

- reuse the same packaged runtime knowledge
- produce a Python-installable artifact if possible
- validate install, provider visibility, and `toy`/`mini_block`

#### Stop-Loss 2

If the wheel regresses provider visibility or execution, keep the working portable bundle as the main artifact and record the wheel failure precisely.

### Phase 3: Add Ubuntu 24.04 Bootstrap Script

- install public prerequisites
- detect `qpm-cli`
- stop with clear guidance if `qpm-cli` is absent
- continue into QAIRT installation when `qpm-cli` is already installed and usable

#### Stop-Loss 3

Do not attempt to automate Qualcomm entitlement or account resolution. Detect and explain only.

### Phase 4: Update README Files

- make the new-host flow discoverable from the root README
- make packaging and validation discoverable from `ort_qnn/README.md`

## Acceptance Criteria

### Portable Bundle Acceptance

A portable directory exists, such as `ort_qnn/artifacts/package/portable/`, and after following its instructions on a prepared host it can:

- import `onnxruntime`
- show `QNNExecutionProvider`
- execute `toy.onnx`
- execute `mini_block.onnx`

### Wheel/Package Acceptance

An installable package artifact exists, such as `ort_qnn/artifacts/package/wheels/onnxruntime-*.whl`, and in a clean Python 3.10 environment:

- it installs successfully
- `QNNExecutionProvider` is visible
- `toy` and `mini_block` run successfully

If it fails, the failure is explicitly classified and documented.

### Bootstrap Acceptance

On Ubuntu 24.04:

- public prerequisites install successfully
- Android base prerequisites are prepared successfully
- rerunning the same script after manual `qpm-cli` installation continues the remaining setup
- the script detects and reports missing login or missing entitlement clearly

### Documentation Acceptance

A new user can follow repository docs without chat history and determine:

- how to prepare a new host
- how to fill in missing `qpm-cli` setup
- how to use the portable bundle
- how to use the wheel/package path
- how to verify provider visibility and model execution

## File Plan

Expected files for this stage:

- create `ort_qnn/package/build_portable_bundle.sh`
- create `ort_qnn/package/build_python_wheel.sh`
- create `ort_qnn/package/package_manifest.py`
- optionally create `ort_qnn/package/verify_package.py`
- create `scripts/bootstrap_ubuntu2404.sh`
- modify `README.md`
- modify `ort_qnn/README.md`
- optionally create `docs/manual/bootstrap-ubuntu2404.md`
- optionally add tests such as:
  - `ort_qnn/tests/test_package_manifest.py`
  - `ort_qnn/tests/test_bootstrap_script_layout.py`

## Success Definition

Best-case completion means:

- the portable bundle works
- the wheel/package path works
- the bootstrap script works on Ubuntu 24.04
- repository docs are updated

Acceptable fallback completion means:

- the portable bundle works
- the bootstrap script works
- the wheel/package path is either working or blocked with a precise documented failure

In both cases, the repo is ready to be moved onto a stronger host without reconstructing environment knowledge from memory.
