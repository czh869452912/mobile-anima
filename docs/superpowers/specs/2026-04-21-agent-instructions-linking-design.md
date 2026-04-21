# Agent Instructions Linking Design

**Date:** 2026-04-21

## Goal

Add repository-level `AGENTS.md` and `CLAUDE.md` files so agent-facing project instructions are easy to discover across tools, while ensuring there is only one canonical source to maintain.

The chosen design is:

- `AGENTS.md` is the single source of truth
- `CLAUDE.md` is a short compatibility shim that points readers to `AGENTS.md`
- the two files are intentionally linked rather than duplicated

## Why This Stage Exists

The repository now has enough structure, constraints, and development conventions that agent-facing instructions should live in the repo itself rather than only in session-time prompts.

Adding a repo-level instruction file helps with:

- consistent behavior across sessions
- faster onboarding for agentic tools
- less repetition in future conversations
- clearer expectations for validation, documentation, and scope control

At the same time, maintaining the same content in both `AGENTS.md` and `CLAUDE.md` would create unnecessary drift risk.

## User Decisions Captured

- `AGENTS.md` should be the canonical maintained file
- `CLAUDE.md` should exist for tool compatibility but should not become a second maintained copy
- the two files should explicitly point to each other in a way that makes the maintenance model obvious
- the content should follow best engineering-practice structure rather than a loose note dump

## Scope

This stage covers:

- creating a root-level `AGENTS.md`
- creating a root-level `CLAUDE.md`
- defining a one-source-of-truth policy between them
- defining the content structure of the canonical agent guidance

This stage does not cover:

- rewriting project human-facing docs such as `README.md`
- per-subdirectory agent instructions
- adding tool-specific generated files or sync scripts
- duplicating full instructions into multiple files

## Alternatives Considered

### Option 1: Duplicate full instructions in both files

**Recommendation:** No

**Pros**

- every tool can read a full instruction file directly
- no dependency on following a pointer file

**Cons**

- guarantees drift over time
- creates unnecessary maintenance overhead
- violates the user's desire to maintain only one file

### Option 2: `AGENTS.md` as canonical, `CLAUDE.md` as a short pointer

**Recommendation:** Yes

**Pros**

- one file to maintain
- explicit and easy to explain
- compatible with agent ecosystems that look for either filename
- minimal overhead

**Cons**

- depends on readers respecting the pointer in `CLAUDE.md`

### Option 3: `AGENTS.md` canonical plus generated `CLAUDE.md`

**Recommendation:** Not now

**Pros**

- can preserve one source of truth while producing multi-tool outputs

**Cons**

- adds tooling and process cost to a problem that does not yet justify automation
- introduces another thing to debug and maintain

## Recommended Structure

### 1. Root `AGENTS.md`

This is the canonical project-level instruction file.

It should be organized into a small set of stable sections:

- **Project Context**
  - what the repo is for
  - current top-level goal
  - current main development strategy
- **Primary Reading Order**
  - where an agent should look first for formal project docs
  - where module docs live
- **Development Rules**
  - follow existing structure and validation ladder
  - do not overclaim progress
  - do not widen scope without explicit discussion
- **Validation Expectations**
  - prefer fresh evidence before claiming status
  - note sandbox-specific limitations when relevant
- **Documentation Rules**
  - update formal project docs when architecture/status changes materially
  - keep `docs/project/` human-facing and `docs/superpowers/` historical
- **Instruction Precedence Note**
  - repo instructions guide normal work but must respect higher-priority session/tool instructions

The tone should be concise, repo-specific, and durable.

### 2. Root `CLAUDE.md`

This should remain intentionally short.

It should contain:

- a one-line statement that the repository's canonical agent instructions live in `AGENTS.md`
- a directive to read and follow `AGENTS.md`
- a statement that `CLAUDE.md` is a compatibility shim and should not be maintained independently

This keeps the maintenance model unambiguous.

## Content Boundaries for `AGENTS.md`

`AGENTS.md` should include durable guidance only.

Good content:

- project identity and current target
- stable doc entrypoints
- validation discipline
- scope guardrails
- documentation update expectations

Bad content:

- ephemeral one-off task notes
- rolling progress bullets that will go stale quickly
- duplicated copies of detailed plans already stored in `docs/superpowers/plans/`
- long historical narratives better suited to human docs

## Linking Policy

The two files should link like this:

- `CLAUDE.md` → points to `AGENTS.md` as canonical
- `AGENTS.md` may mention that `CLAUDE.md` exists only as a compatibility alias

The direction of authority must be one-way:

- authority lives in `AGENTS.md`
- `CLAUDE.md` defers to it

## Best-Practice Writing Guidelines

The canonical `AGENTS.md` should follow these practices:

- keep sections short and scannable
- prefer explicit rules over vague style advice
- tie instructions to the actual repo structure
- avoid repeating what already belongs in `README.md`
- avoid agent-platform-specific implementation details unless necessary
- make the maintenance model explicit in the first screenful of text

## Acceptance Criteria

This stage is complete when:

- the repository contains a root `AGENTS.md`
- the repository contains a root `CLAUDE.md`
- `AGENTS.md` clearly acts as the only maintained source
- `CLAUDE.md` clearly points to `AGENTS.md` instead of duplicating it
- the `AGENTS.md` structure is concise, durable, and repo-specific

## Stop-Loss Rules

### Stop-Loss 1: Do not duplicate the full content

If `CLAUDE.md` starts to mirror the full text of `AGENTS.md`, the design has failed.

### Stop-Loss 2: Do not overload the file with transient status

If the canonical agent file starts reading like a progress log, move that information back into project docs.

### Stop-Loss 3: Do not make the guidance tool-fragile

If instructions only make sense for a single agent runtime, they are too narrow for a repository-level file.

## Next Step After This Design

After approval, implement the two files directly in the repo:

- add canonical `AGENTS.md`
- add lightweight pointer `CLAUDE.md`
- keep the content minimal, durable, and linked
