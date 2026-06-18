# First-pass prototype audit status

Prototype patch retained for audit and reproduction: `patches/llamacpp-4218-gss-recognizer.patch`.

Commit under test: `26a78fc34afe7e5b93af58ee8b88c268df9569b9`.

The first-pass implementation added an opt-in graph-structured-stack recognizer path selected by `LLAMA_GRAMMAR_GSS=1`. It is not the final fix.

## Audit status

- Observed: the prototype collapsed the measured exponential stack growth on the synthetic and realistic fixtures.
- Observed: a later audit found `CHAR_ALT` multi-range under-acceptance.
- Observed: the same audit found raw-byte token under-acceptance.
- Observed: the same audit found clone-related undefined behavior risk.
- Inferred: the root-cause fix should be rebuilt in place around existing grammar semantics, using stack sharing only at safe reconvergence points.
- Unknown: the final mergeable patch is not included in this artifact.

## Performance measurements

| Case | Baseline max stacks | Prototype max stacks | Baseline accept time | Prototype accept time |
|---|---:|---:|---:|---:|
| Synthetic `a^16 b^16` | 196,608 | 8 | 42,900.235 ms | 0.034 ms |
| Recursive workflow, depth 13 | 81,920 | 10 | 57,554.288 ms | 0.371 ms |

These measurements support the mechanism: shared reconvergent parse state prevents the stack explosion. They are not correctness certification for the prototype patch.

## Reject-path benchmark

`llama_grammar_apply_impl()` over 128k candidate tokens, 30 measured iterations:

| Path | Median | Mean | p95 |
|---|---:|---:|---:|
| Baseline recognizer | 4,654.760 us | not reported here | not reported here |
| First-pass prototype | 3,811.255 us | not reported here | not reported here |

The faster reject-path result is important because a stack-sharing recognizer must not merely improve acceptance of pathological inputs while slowing ordinary candidate filtering.

## Compatibility note

The public stack accessor in the prototype was served by a shim. It exposed active terminal frontier information compactly, but it was not a full materialization of all historical concrete continuations. A final in-place fix needs explicit compatibility decisions and tests around stack access, reset, and clone behavior.
