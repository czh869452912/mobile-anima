# Agent Instructions Linking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add root-level `AGENTS.md` and `CLAUDE.md` so the repository has one canonical agent-instructions file and one lightweight compatibility shim.

**Architecture:** `AGENTS.md` is the only maintained source of repo-level agent guidance. `CLAUDE.md` is intentionally tiny and only points readers to `AGENTS.md`, making the one-source-of-truth model explicit and avoiding duplicated maintenance.

**Tech Stack:** Markdown documentation, root-level repository guidance files, grep-based verification commands

---

### File Structure

- Create: `AGENTS.md` — canonical repository-level agent instructions
- Create: `CLAUDE.md` — lightweight compatibility shim that defers to `AGENTS.md`

### Task 1: Add the canonical `AGENTS.md`

**Files:**
- Create: `AGENTS.md`

- [ ] **Step 1: Write the failing verification command**

Run:

```bash
test -f AGENTS.md
```

Expected: exit code `1` because the file does not exist yet.

- [ ] **Step 2: Write the canonical `AGENTS.md`**

```bash
cat > AGENTS.md <<'EOF'
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
- Keep failure modes explicit; avoid masking ambiguity with “best effort” logic unless the user asks for it.

## Documentation Rules

- Update `docs/project/` when project-level goals, architecture, status, or roadmap change materially.
- Keep `README.md` as the human entrypoint, not a rolling engineering diary.
- Keep `docs/project/` human-facing and structured.
- Keep `docs/superpowers/` as historical design and implementation artifacts.
- Avoid duplicating the same project narrative across multiple docs; prefer one primary home and link to it.

## Instruction Precedence

Repository instruction files guide normal work in this repo, but they do not override higher-priority runtime instructions such as system, developer, harness, or direct user requests.
EOF
```

- [ ] **Step 3: Verify the new `AGENTS.md` content**

Run:

```bash
grep -n 'canonical repository-level instruction source\|README.md\|docs/project/index.md\|docs/superpowers/' AGENTS.md
```

Expected:

```text
AGENTS.md:<line>: This file is the canonical repository-level instruction source...
AGENTS.md:<line>: 1. `README.md`
AGENTS.md:<line>: 2. `docs/project/index.md`
AGENTS.md:<line>: ... `docs/superpowers/` ...
```

### Task 2: Add the lightweight `CLAUDE.md` compatibility shim

**Files:**
- Create: `CLAUDE.md`

- [ ] **Step 1: Write the failing verification command**

Run:

```bash
test -f CLAUDE.md
```

Expected: exit code `1` because the file does not exist yet.

- [ ] **Step 2: Write the lightweight `CLAUDE.md`**

```bash
cat > CLAUDE.md <<'EOF'
# CLAUDE.md

This repository's canonical agent instructions live in `AGENTS.md`.

Read and follow `AGENTS.md` first.

This file is a compatibility shim only and should not be maintained independently from `AGENTS.md`.
EOF
```

- [ ] **Step 3: Verify that `CLAUDE.md` defers to `AGENTS.md`**

Run:

```bash
grep -n 'AGENTS.md\|compatibility shim\|should not be maintained independently' CLAUDE.md
```

Expected:

```text
CLAUDE.md:<line>: This repository's canonical agent instructions live in `AGENTS.md`.
CLAUDE.md:<line>: This file is a compatibility shim only...
CLAUDE.md:<line>: ... should not be maintained independently ...
```

### Task 3: Verify the one-source-of-truth relationship and commit once

**Files:**
- Test: `AGENTS.md`
- Test: `CLAUDE.md`

- [ ] **Step 1: Run the relationship verification commands**

Run:

```bash
grep -n 'canonical repository-level instruction source' AGENTS.md
grep -n 'Read and follow `AGENTS.md` first' CLAUDE.md
grep -n 'should not be maintained independently' CLAUDE.md
```

Expected:

```text
AGENTS.md:<line>: This file is the canonical repository-level instruction source...
CLAUDE.md:<line>: Read and follow `AGENTS.md` first.
CLAUDE.md:<line>: ... should not be maintained independently ...
```

- [ ] **Step 2: Commit**

```bash
git add AGENTS.md CLAUDE.md
git commit -m "docs: add linked agent instruction files"
```
