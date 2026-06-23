# Validation — binary as oracle

All results below were produced on a built binary on real hardware with a real
248k-vocabulary GGUF (Qwen3.5-35B-A3B, Q8_0). Synthetic unit tests were used for
regression coverage but were **not** treated as proof; a clean run of the real
workload on the real engine is the bar. Where a claim was contested by a
reviewer, a probe was built and run, and the binary's output is what is recorded.

Fix commit is against upstream `master` (`0eb874d3...`); production change is in
`src/llama-grammar.cpp` (~30 lines, deletes both `throw` sites), plus regression
tests in `tests/test-grammar-integration.cpp`.

## Counterexample battery (all PASS / refuted on the validated fix)

| Case  | What it probes                                              | Result |
|-------|-------------------------------------------------------------|--------|
| CX-1  | `root ::= "a" !<[7]>` + `accept_str("ab")` (token-rule)     | no throw; grammar stays at a valid position |
| CX-2  | `[^"]*` + `[C2,h,e,l,l,o]` (invalid continuation, resync)   | `e/l/l/o` accepted; decoder resyncs; no deadlock |
| CX-3  | `[^U+0100]` + split `C4 81` (valid multi-byte across calls) | U+0101 accepted (matches upstream) |
| CX-4  | `<[7]>` + `accept_token(8, "")` (empty piece, wrong id)     | no throw; no-op |
| CX-5  | mid-string rejection + trailing partial (torn transaction)  | no false accept; stacks + partial both roll back together |
| CX-6  | once-only warning under concurrency                         | ThreadSanitizer clean (8 threads × 10,000 calls, 0 races) |

## No-regression: differential against upstream master

A differential harness ran a corpus of valid inputs (literals, repetition, a JSON
object, token-terminal and inverse-token-terminal grammars, split U+0101,
contiguous split UTF-8) through both the fixed engine and clean `master`, hashing
the resulting grammar state after each step.

```
fix    FNV64 = a2b3694207dca6c1
master FNV64 = a2b3694207dca6c1   (identical)
```

On valid inputs the fix is behaviorally identical to upstream — it only differs
on the non-conformant forced-token path, where `master` crashes.

## Original repro

The original server-terminating sequence (a byte-fallback partial lead followed
by non-conformant control tokens, real tokenizer ids) on `[^"]*`:

- `master`: aborts with "Unexpected empty grammar stack".
- fix: no crash, no spurious "bridged" completion across the rejected bytes, and a
  following valid token resyncs and completes the grammar.

## Performance

Real server generation, 111 predicted tokens, real GGUF:

```
master : 58.509 ms/token
fix    : 58.857 ms/token   (+0.60%, within single-run noise)
```

Grammar accept is a vanishing fraction of per-token time; the change only guards
an existing per-accept transition. The once-only warning fired **0 times** during
valid generation.

## Test / sanitizer gates

- Release grammar ctests (`grammar | gbnf | json-schema`): 4/4 pass.
- ASAN/UBSAN grammar ctests: 4/4 pass.
- Full server build: success.

## Limitations (Unknown)

- Language equivalence outside the tested grammar corpus is not proven (it is
  argued from per-codepoint transition equivalence + the differential match).
- The fix is scoped to the two `accept_*` `runtime_error` sites. A separate
  forced-EOG `GGML_ABORT` (forcing END on an unsatisfiable grammar) is a distinct
  failure mode and intentionally out of scope.
- Upstream review/merge outcome is open.
