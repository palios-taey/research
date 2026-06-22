# llama.cpp #4218 grammar engine audit package

This is a public audit and validation package for the in-place grammar engine repair for llama.cpp issue #4218.

Public branch:

```text
https://github.com/palios-taey/llama.cpp/tree/codex/4218-rootcause-earley
```

Current audited commit:

```text
3e4aa92bed972d60cbf9a02795d40bed10a60338
grammar: reject EOG while UTF-8 is incomplete
```

## Status

- Observed: the branch replaces concrete continuation-stack enumeration with chart items and compacts sealed origins through normalized resume summaries.
- Observed: the branch keeps focused regression tests for stack compaction, clone-vs-commit parity, token terminals, `TOKEN_NOT`, split UTF-8, nullable UTF-8 EOG handling, and completion reallocation stress.
- Observed: the latest validation run passed the focused compactor harness, the expanded differential harness, six grammar/GBNF ctests, an ASAN/UBSAN focused run, and a full server-enabled build.
- Inferred: the current design addresses the #4218 exponential stack mechanism without the first-pass prototype's semantic blind spots.
- Unknown: upstream review outcome and language equivalence outside the tested grammar corpus remain open.

## What Changed

The original #4218 failure mode was exponential continuation-stack growth on ambiguous recursive grammars. The current branch moves the recognizer to chart-style items keyed by rule, dot, and origin, then compacts sealed origins that have equivalent resume behavior. This preserves the need for semantically distinct input positions while avoiding unbounded retention for reconvergent states such as `a*`.

The current package is not the older opt-in graph-structured-stack prototype. Historical prototype materials remain in `artifacts/` and `patches/` only as background and should not be treated as the current merge candidate.

## Package Contents

- `AUDIT_TRAIL.md`: five independent automated review findings and the fixes they drove.
- `VALIDATION.md`: fresh validation commands, public-safe outputs, and limitations.
- `harnesses/current/test-grammar-compactor.cpp`: current focused compactor and regression harness source.
- `harnesses/current/test-grammar-differential.cpp`: current expanded clone-vs-commit differential harness source.
- `harnesses/README.md`: how to run the current harnesses from a llama.cpp checkout.
- `artifacts/`: historical root-cause measurements and first-pass prototype background.
- `patches/`: historical prototype patch retained for audit context only.

## Current Validation Snapshot

Focused compactor, short run:

```text
astar n=1000 origins=2 sealed=1 current_items=5 stored_items=5 resume_entries=1
balanced-prefix n=64 origins=65 sealed=64 current_items=4 stored_items=67 resume_entries=63
token-not-empty-candidate excluded=0 allowed=1
nullable-utf8-eog partial=0 completed=1 wrong=0
completion-reallocation-stress callers=8191 advanced=8191 final_items=16384
```

Focused compactor, long run:

```text
astar n=100000 origins=2 sealed=1 current_items=5 stored_items=5 resume_entries=1
balanced-prefix n=512 origins=513 sealed=512 current_items=4 stored_items=515 resume_entries=511
token-not-empty-candidate excluded=0 allowed=1
nullable-utf8-eog partial=0 completed=1 wrong=0
completion-reallocation-stress callers=8191 advanced=8191 final_items=16384
long_elapsed=0:18.96 maxrss_kb=8252
```

Differential harness:

```text
grammar-differential decisions=2328 fnv64=b77904254b842fea
```

Grammar/GBNF ctest gate:

```text
100% tests passed, 0 tests failed out of 6
```

ASAN/UBSAN focused compactor run:

```text
asan_stderr_bytes=0
```

Full server-enabled build:

```text
completed to 100%, including llama-server and llama-app
```

See `VALIDATION.md` for commands, trace-equivalence notes, and limitations.

## How To Review

1. Check out the branch above at commit `3e4aa92bed972d60cbf9a02795d40bed10a60338`.
2. Build with tests enabled.
3. Run the current harnesses in `tests/` from the branch, or compare against the copied sources under `harnesses/current/`.
4. Read `AUDIT_TRAIL.md` before reviewing the patch; it lists the concrete defects independent reviews found and how the branch was changed.
5. Treat `artifacts/` and `patches/` as historical context, not as the active fix.

For exact build and test commands, use `VALIDATION.md`.

## Limits

The validation here is adversarial differential testing plus focused regression coverage. It is not a formal proof. The strongest evidence is that the final branch preserves the expanded precompactor decision trace for the tested corpus while adding targeted tests for review-found blind spots.
