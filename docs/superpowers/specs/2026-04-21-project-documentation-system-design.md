# Project Documentation System Design

**Date:** 2026-04-21

## Goal

Replace the current progress-update-style top-level documentation with a more durable project documentation system that can guide development, onboard collaborators, and track project progress in a structured way.

The new system should make `README.md` the stable entrypoint for humans, while introducing a formal `docs/project/` documentation area for project-level requirements, architecture, roadmap, status, and validation guidance.

## Why This Stage Exists

The repository has grown from a lightweight experimental workspace into a multi-stage engineering effort with:

- host-side QNN smoke validations
- desktop `ORT + QNN` validation
- Android `ORT + QNN` runtime integration
- a growing collection of specs and plans under `docs/superpowers/`

The current documentation state no longer matches that maturity level.

### Current Problems

1. `README.md` behaves more like a rolling progress note than a durable project entrypoint.
2. project-level knowledge is scattered across:
   - root `README.md`
   - module READMEs under `smoke/` and `ort_qnn/`
   - manual runbooks under `docs/manual/`
   - agent-generated specs and plans under `docs/superpowers/`
3. `docs/superpowers/` contains valuable design content, but it is not the right primary reading path for humans trying to understand the current project.
4. there is no single formal project-level status view that tracks progress by module with evidence and next steps.

The result is that a new collaborator can discover a lot of information, but not through one stable, deliberate path.

## User Decisions Captured

- The new documentation should serve multiple audiences, but primarily optimize for a new collaborator joining the effort.
- The repo should gain a formal project-level documentation area under `docs/project/`.
- Progress tracking should use a module-based status view with:
  - status
  - evidence links
  - next step
- Stable design content from `docs/superpowers/specs/` should be absorbed into the new formal docs where appropriate.
- `docs/superpowers/` should remain as original agent work artifacts, but should no longer be the main reading entrypoint.
- Formal docs should be Chinese-first, while keeping key technical terms and filenames in English.
- The rewrite should be a medium-size restructuring, not a full destructive documentation reset.

## Scope

This stage covers:

- rewriting the root `README.md`
- creating a formal `docs/project/` documentation structure
- defining the role of each documentation layer in the repo
- introducing a module-oriented project status document
- aligning the most important existing module docs to the new top-level structure
- absorbing stable content from `docs/superpowers/specs/` into the formal docs where appropriate

This stage does not cover:

- deleting historical specs or plans from `docs/superpowers/`
- rewriting every documentation file in the repository
- changing runtime code behavior
- adding ADRs or a full decision-record system
- translating every module document into bilingual form

## Recommended Approach

Use a layered documentation system with a clear separation between:

1. **Entry documentation** — root `README.md`
2. **Formal project documentation** — `docs/project/`
3. **Module/operator documentation** — `smoke/README.md`, `ort_qnn/README.md`, and related manual docs
4. **Agent work artifacts** — `docs/superpowers/`

This is the best balance because it preserves existing working documents while creating a human-first structure for ongoing development.

## Alternatives Considered

### Option 1: Rewrite only `README.md`

**Recommendation:** No

**Pros**

- very cheap
- minimal file churn

**Cons**

- does not solve scattered project knowledge
- still leaves no formal project-level status system
- keeps the project dependent on `docs/superpowers/` for important context

### Option 2: Build a formal `docs/project/` layer and align key docs

**Recommendation:** Yes

**Pros**

- introduces a durable information architecture
- preserves current module docs and work artifacts
- makes onboarding and progress tracking significantly better
- avoids unnecessary destructive churn

**Cons**

- requires a coordinated rewrite rather than a single-file edit
- some duplication must be cleaned up carefully during migration

### Option 3: Full documentation reset and consolidation

**Recommendation:** Not now

**Pros**

- could produce the cleanest final state

**Cons**

- too much churn for the current stage
- high risk of breaking references and losing useful historical context
- slower than the value it creates right now

## Documentation Architecture

The new documentation should use the following structure.

### 1. Root `README.md`

`README.md` becomes the stable human entrypoint.

It should answer, in order:

- what this project is trying to achieve
- what the current target and non-goals are
- what has already been validated
- where to find the formal project docs
- where to find module-specific docs
- what the next major milestone is

It should no longer act as the main home for detailed progress notes.

### 2. Formal Project Docs: `docs/project/`

This becomes the primary structured documentation area for humans.

Recommended files:

- `docs/project/index.md`
  - documentation map and reading order
- `docs/project/overview.md`
  - project mission, target platform, guiding constraints
- `docs/project/requirements.md`
  - current product/technical requirements and scope boundaries
- `docs/project/architecture.md`
  - system architecture and major runtime split decisions
- `docs/project/roadmap.md`
  - milestone-oriented plan for moving from current state to the target
- `docs/project/status.md`
  - module-based progress tracker with status, evidence, and next step
- `docs/project/validation.md`
  - validation ladder and what each stage proves

This directory is the main place a collaborator should read after `README.md`.

### 3. Module/Operator Docs

These remain where they are, but should be reframed as lower-level operational documents.

Examples:

- `smoke/README.md`
- `ort_qnn/README.md`
- `docs/manual/android-validation.md`

Their role is to explain how to operate or validate a sub-area, not to carry the whole project narrative.

### 4. Agent Artifact Docs

`docs/superpowers/` remains in the repo, but should be documented as:

- working history
- detailed design artifacts
- implementation plans used during development

It should not remain the primary navigation path for understanding the current official project state.

## Status Tracking Model

`docs/project/status.md` should track progress by module, not just by chronological milestone.

Recommended columns:

- **Module**
- **Goal**
- **Current Status**
- **Evidence**
- **Next Step**

Recommended status vocabulary:

- `Not Started`
- `In Progress`
- `Blocked`
- `Validated`

The status page should make it easy to answer:

- which parts are already proven
- which parts are partially built but not yet validated
- which part is the current bottleneck

## Content Migration Rules

Stable information currently living in `docs/superpowers/specs/` should be absorbed into `docs/project/` where it has become part of the durable project understanding.

Examples of stable content likely to migrate upward:

- overall project goal and architectural direction
- Android denoiser-first strategy
- mixed-backend scope decisions
- validation ladder ordering
- packaging/bootstrap role in the broader project

The originals in `docs/superpowers/specs/` should remain intact, but the formal docs should become the primary source for current project understanding.

## Linking Policy

The repo should follow a clear linking hierarchy.

### `README.md` links to:

- `docs/project/index.md`
- key module docs (`smoke/README.md`, `ort_qnn/README.md`)
- key manual docs when necessary

### `docs/project/index.md` links to:

- all formal project docs
- key module docs
- optionally a short note that `docs/superpowers/` contains historical design/implementation artifacts

### Module docs link back to:

- `docs/project/index.md`
- the most relevant project-level architecture or validation page

This creates two-way navigation instead of isolated documents.

## Language Policy

Formal project docs should be written in Chinese first, while preserving:

- file paths
- code identifiers
- library names
- platform/runtime names
- critical technical terms such as `ONNX Runtime`, `QNNExecutionProvider`, `QAIRT`, `HTP`, `denoiser`

This keeps the docs readable for the primary user while remaining precise for engineering work.

## Implementation Sequence

The documentation rewrite should happen in this order:

1. rewrite root `README.md`
2. create `docs/project/index.md`
3. create the formal project docs under `docs/project/`
4. align `smoke/README.md`, `ort_qnn/README.md`, and `docs/manual/android-validation.md` with the new navigation model
5. reduce duplicated project narrative from lower-level docs where appropriate

This order ensures that the new entrypoint exists before module docs are updated to point to it.

## Acceptance Criteria

This stage is complete when:

- `README.md` reads like a durable project entrypoint instead of a status dump
- `docs/project/` exists and provides a coherent human-first documentation path
- a collaborator can understand the project through `README.md` → `docs/project/index.md` → module docs
- `docs/project/status.md` tracks project progress by module with status, evidence, and next step
- stable content from `docs/superpowers/specs/` is reflected in the formal project docs
- key module docs are aligned with and linked into the new structure

## Stop-Loss Rules

### Stop-Loss 1: Don’t delete historical design artifacts

Do not “clean up” by removing `docs/superpowers/` content that is still useful as development history.

### Stop-Loss 2: Don’t overload `README.md`

If `README.md` starts becoming a second full documentation tree, move content back into `docs/project/`.

### Stop-Loss 3: Don’t duplicate narrative unnecessarily

If the same project-level explanation appears in multiple places, choose one primary home and link to it.

## Next Step After This Design

After this design is approved, the next step is to write the implementation plan for the documentation rewrite, then update the docs in a focused sequence beginning with `README.md` and `docs/project/`.
