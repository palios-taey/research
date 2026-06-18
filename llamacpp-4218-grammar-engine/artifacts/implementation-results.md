# Implementation results

Candidate implementation patch: `patches/llamacpp-4218-gss-recognizer.patch`.

Commit under test: `26a78fc34afe7e5b93af58ee8b88c268df9569b9`.

The implementation adds an opt-in GSS recognizer path selected by `LLAMA_GRAMMAR_GSS=1`. It preserves the existing path for default execution and for in-process differential testing.

## Performance measurements

| Case | Baseline max stacks | GSS max stacks | Baseline accept time | GSS accept time |
|---|---:|---:|---:|---:|
| Synthetic `a^16 b^16` | 196,608 | 8 | 42,900.235 ms | 0.034 ms |
| Recursive workflow, depth 13 | 81,920 | 10 | 57,554.288 ms | 0.371 ms |

## Reject-path benchmark

`llama_grammar_apply_impl()` over 128k candidate tokens, 30 measured iterations:

| Path | Median | Mean | p95 |
|---|---:|---:|---:|
| Baseline recognizer | 4,654.760 us | not reported here | not reported here |
| GSS recognizer | 3,811.255 us | not reported here | not reported here |

The faster reject-path result is important because a stack-sharing recognizer must not merely improve acceptance of pathological inputs while slowing ordinary candidate filtering.

## Compatibility note

The public stack accessor is served by a shim on the GSS path. It exposes active terminal frontier information compactly. It is not a full materialization of all historical concrete continuations; code that depends on direct mutation of returned stack vectors needs targeted compatibility testing.
