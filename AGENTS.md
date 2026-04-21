# AGENTS.md

This file is the canonical repository-level instruction source for agentic tools working in this project.

If another tool-specific file such as `CLAUDE.md` exists, this file takes precedence as the maintained source. Compatibility files should point here instead of duplicating the full content.

## Project Context

`Mobile Anima NPU` is an experimental project for bringing `Anima` inference onto Qualcomm Snapdragon Android devices.

The current engineering strategy is to reduce risk in layers:

1. validate minimal `QNN/QAIRT` smoke graphs
2. validate desktop `ORT + QNNExecutionProvider`
3. validate Android `denoiser-first` runtime execution
4. only then push into real `Anima` export, operator-gap analysis, and device-proven evidence

## Primary Reading Order

Before making non-trivial changes, read in this order:

1. `README.md`
2. `docs/project/index.md`
3. the most relevant file under `docs/project/`
4. the relevant module doc such as `smoke/README.md`, `ort_qnn/README.md`, or `docs/manual/android-validation.md`
5. `docs/superpowers/specs/` and `docs/superpowers/plans/` only when historical design context is needed

## Development Rules

- Follow the existing validation ladder; do not skip directly to high-cost work if a cheaper proof layer exists.
- Prefer precise, limited changes over broad restructuring.
- Do not claim a milestone is complete without fresh verification evidence.
- Do not widen scope silently; call out when a request implies a larger architecture shift.
- Treat Android `denoiser-first` as the current mainline unless project docs explicitly say otherwise.

## Validation Expectations

- Prefer fresh, local verification before claiming code or docs are correct.
- If the sandbox prevents a normal verification path, say so explicitly and use the closest trustworthy fallback.
- Keep failure modes explicit; avoid masking ambiguity with "best effort" logic unless the user asks for it.

## Documentation Rules

- Update `docs/project/` when project-level goals, architecture, status, or roadmap change materially.
- Keep `README.md` as the human entrypoint, not a rolling engineering diary.
- Keep `docs/project/` human-facing and structured.
- Keep `docs/superpowers/` as historical design and implementation artifacts.
- Avoid duplicating the same project narrative across multiple docs; prefer one primary home and link to it.

## Instruction Precedence

Repository instruction files guide normal work in this repo, but they do not override higher-priority runtime instructions such as system, developer, harness, or direct user requests.
