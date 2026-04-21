# Android Denoiser Strict Input Binding Design

**Date:** 2026-04-21

## Goal

Tighten the Android denoiser execute path so input binding is driven by strict metadata heuristics instead of mostly relying on exact input names.

This stage does not add new submodels or broader runtime scope. It refines the correctness boundary for the existing one-shot denoiser execute path so the app fails safely when the exported model contract is ambiguous.

## Why This Stage Exists

The repository now already proves that Android can:

- create an `ORT + QNN` denoiser session
- inspect session metadata
- validate the four runtime raw input files
- execute one denoiser pass and persist `denoiser_output.raw`

However, the current binding logic is still too name-driven:

- `latent` is matched by a name containing `latent`
- `timestep` is matched by a name containing `time`
- `cond` and `uncond` are matched by exact names

That is cheaper than hardcoding shapes, but it does not yet satisfy the stricter metadata-first design direction we agreed on. A denoiser export with slightly different names but equivalent metadata could fail unnecessarily, while a badly named export could still be accepted if the names happen to match superficially.

The next lowest-cost improvement is therefore not a new runtime feature. It is to make input binding more trustworthy by using shape, type, and constrained naming rules together.

## Scope

This stage covers:

- replacing the current mostly name-based denoiser input binding with strict metadata heuristics
- using tensor rank, static shape, and type to narrow candidate sets before names are used
- preserving explicit failure when metadata is ambiguous
- keeping the current four runtime input files:
  - `latent.raw`
  - `timestep.raw`
  - `cond.raw`
  - `uncond.raw`
- adding targeted JVM tests for ambiguous and successful strict-binding cases

This stage does not cover:

- new input file formats
- sidecar metadata files
- Android device automation
- full prompt-to-image generation
- additional tensor element types beyond the currently supported set
- broader multi-output or multi-branch denoiser contracts

## User Decisions Captured

- Continue refining the Android denoiser-first path instead of expanding to device automation first
- Use the most conservative binding policy available
- Prefer explicit failure over risky inference
- Require metadata plus sufficiently clear names before `cond` and `uncond` are accepted

## Approaches Considered

### Option 1: Keep name-based matching only

**Recommendation:** No

**Pros**

- lowest implementation cost
- simple to understand

**Cons**

- still too brittle to name drift
- does not use the metadata already available from the session
- weak protection against ambiguous contracts

### Option 2: Strict metadata-first heuristics with name confirmation

**Recommendation:** Yes

**Pros**

- uses runtime metadata as the primary source of truth
- keeps the binding rules auditable and deterministic
- rejects ambiguous exports before they reach execution
- improves correctness without adding new artifacts or tooling

**Cons**

- will reject some exports that might be recoverable with looser heuristics
- requires slightly more complex binding logic and tests

### Option 3: Aggressive automatic inference from metadata alone

**Recommendation:** No

**Pros**

- accepts more models automatically
- less sensitive to naming differences

**Cons**

- too risky for the current denoiser-first validation stage
- makes wrong bindings harder to diagnose
- conflicts with the chosen “strict first” policy

## Recommended Strategy

Use strict metadata-first binding with explicit name confirmation only where needed.

The runtime should treat metadata as the primary filter and names as the final disambiguation layer.

The rule is simple:

- if metadata makes a candidate unique, bind it
- if metadata narrows the set but names are still needed, require strong and explicit names
- if ambiguity remains at any step, fail with `INPUT_MAPPING_FAILED`

The runtime must not:

- bind by file order
- bind by “remaining unmatched tensors”
- silently swap `cond` and `uncond`
- weaken its standards to keep execution going

## Binding Rules

### `timestep`

`Timestep` is the most constrained input.

A candidate is valid only if:

- it is the unique scalar-like or single-element input
- its type is supported for timestep in the current stage
  - `INT64`
  - `FLOAT`
- its shape is fully static

If more than one such candidate exists, fail.

If none exists, fail.

### `latent`

`Latent` is the main denoiser tensor.

A candidate is valid only if:

- it is the unique 4D tensor candidate
- its type is `FLOAT`
- its shape is fully static

If more than one valid 4D `FLOAT` candidate exists, fail.

If none exists, fail.

### `cond` and `uncond`

`Cond` and `uncond` are handled as a pair, not independently.

A candidate pair is valid only if:

- there are exactly two remaining candidates after `latent` and `timestep` are bound
- both have the same supported type
- both have the same fully static shape
- both are plausible conditioning tensors by exclusion

Even after that, the pair may only be bound if names are sufficiently explicit.

Accepted examples:

- `cond` and `uncond`
- `positive_cond` and `negative_uncond`
- equivalent names where one side clearly denotes conditioned input and the other clearly denotes unconditioned input

Rejected examples:

- `context_a` and `context_b`
- `embedding_0` and `embedding_1`
- two unnamed or semantically opaque tensors with matching shapes

If the pair is structurally plausible but naming is not explicit enough, fail with `INPUT_MAPPING_FAILED`.

## Output Behavior

This stage does not change the current output policy.

The runtime still expects exactly one clearly usable primary output tensor for the current denoiser path. If multiple outputs are returned and there is no unambiguous primary output, execution must fail explicitly.

## Failure Model

This stage does not add new failure codes. It sharpens when existing failures are used.

Primary failure used by this stage:

- `INPUT_MAPPING_FAILED`

Supporting failures that still apply from the previous stage:

- `DYNAMIC_SHAPE_UNSUPPORTED`
- `UNSUPPORTED_TENSOR_TYPE`
- `INPUT_FILE_SIZE_MISMATCH`

The key change is that `INPUT_MAPPING_FAILED` now means:

- no unique timestep candidate
- no unique latent candidate
- no valid `cond/uncond` pair
- `cond/uncond` pair structurally matches but naming is not explicit enough

## API Impact

The public runtime surface does not need a large redesign for this stage.

Expected code changes are focused in:

- `DenoiserInputBinding.kt`
- related tests for strict candidate selection

The ORT bridge and session factory should remain mostly unchanged because this stage improves interpretation of metadata, not metadata collection itself.

## Testing Strategy

Keep this stage fully JVM-test-driven.

### Required tests

- unique scalar-like timestep binds successfully
- multiple scalar-like timestep candidates fail
- unique 4D float latent binds successfully
- multiple 4D float latent candidates fail
- two matching conditioning tensors with explicit names bind successfully
- two matching conditioning tensors with opaque names fail
- dynamic-shape candidates still fail before binding completes
- unsupported element types still fail before binding completes

### Non-goal for testing

- no device tests are required to prove the heuristic policy itself

The correctness target here is binding logic, not device integration.

## Acceptance Criteria

This stage is complete when:

- `bindDenoiserInputs(...)` no longer depends primarily on exact name matches
- `timestep` and `latent` are selected through strict metadata heuristics
- `cond/uncond` are only accepted when both metadata and names are sufficiently clear
- ambiguous pairs such as `context_a/context_b` are rejected
- the updated binding logic is covered by focused JVM tests
- the full Android runtime test set remains green

## Stop-Loss Rules

### Stop-Loss 1: Ambiguous timestep

If more than one scalar-like supported input exists, fail instead of guessing.

### Stop-Loss 2: Ambiguous latent

If more than one 4D float candidate exists, fail instead of preferring the first one.

### Stop-Loss 3: Ambiguous conditioning pair

If the remaining pair is structurally plausible but naming is not explicit enough, fail instead of swapping or guessing.

## Next Stage After This One

Once strict metadata-first binding is in place, the next lowest-cost step is either tightening real device validation or expanding the runtime contract only if the exported denoiser requires broader type or output handling.
