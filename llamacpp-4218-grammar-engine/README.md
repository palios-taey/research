# llama.cpp #4218 grammar-engine stack sharing prototype

This artifact packages a measured reproduction, a candidate GLL/GSS-style grammar recognizer patch, and validation harnesses for llama.cpp issue #4218.

## Problem

The current grammar engine can enumerate exponentially many continuation stacks for ambiguous recursive grammars. A minimal family is:

```gbnf
root ::= nest
nest ::= "a" nest "b" | "a" nest "c" |
```

On input `a^n b^n`, the engine reaches `3 * 2^n` active stacks. At `n = 16`, the measured baseline reached `196,608` stacks. A representative baseline validation run spent `42,900.235 ms` in token acceptance for that case; a separate initial repro run on the same grammar measured `32,799.244 ms`.

The same failure mode appears in realistic JSON-schema-derived grammars. A recursive workflow schema with a late `strategy` discriminator reached `81,920` active stacks at depth 13 and spent `57,554.288 ms` in token acceptance. Controls that moved the discriminator earlier or removed the recursive union stayed under 10 stacks.

## Approach

The patch augments the existing grammar engine with an opt-in recognizer that memoizes reconverged parse continuations instead of keeping every equivalent continuation stack as a separate concrete vector. Conceptually, it is a GLL/GSS-style graph-structured stack over the existing parsed grammar, not a replacement grammar parser or grammar syntax change.

The implementation remains opt-in through `LLAMA_GRAMMAR_GSS=1`. The default path is left available for differential testing and compatibility checks.

## Results

| Case | Baseline max stacks | GSS max stacks | Baseline accept time | GSS accept time |
|---|---:|---:|---:|---:|
| Synthetic `a^n b^n`, n=16 | 196,608 | 8 | 42,900.235 ms | 0.034 ms |
| Recursive workflow schema, depth 13 | 81,920 | 10 | 57,554.288 ms | 0.371 ms |

Reject-path benchmarking with 128k token candidates also improved in the measured case: median `llama_grammar_apply_impl()` time was `4,654.760 us` on the baseline path and `3,811.255 us` on the GSS path.

## Validation

Differential validation compared old and GSS accept sets at generation states using the real `llama_grammar_apply_impl()` implementation in both modes.

| Corpus | Grammars built by both | States | Token decisions | Divergences |
|---|---:|---:|---:|---:|
| Existing-test-style plus generated corpus | 238 | 16,804 | 138,018,304 | 0 |
| Independent adversarial falsification corpus | 2,275 | 30,298 | 1,522,686,586 | 0 |
| Combined | 2,513 | 47,102 | 1,660,704,890 | 0 |

The independent corpus specifically targeted failure modes expected to break naive stack merging: contingent-pop behavior, mutually recursive cycles, nullable epsilon cycles, nullable repetition bodies, high reconvergence ambiguity, and long-range dependencies.

## Caveats

This is adversarial differential testing, not a formal proof of language equivalence. The implementation should remain behind the opt-in flag until broader review and integration testing are complete.

The `get_stacks()` compatibility shim exports the active terminal frontier compactly. It is sufficient for existing reset-to-initial behavior and the validation harnesses here, but it does not reconstruct every historical concrete continuation vector from the shared representation. Downstream code that mutates returned stack vectors directly needs explicit compatibility review.

## How to reproduce

1. Check out llama.cpp and apply `patches/llamacpp-4218-gss-recognizer.patch`.
2. Configure and build with tests enabled.
3. From the patched llama.cpp checkout, run the repro harness on the baseline path and on the opt-in GSS path:

```bash
python3 path/to/llamacpp-4218-grammar-engine/harnesses/run_llamacpp_4218_validation.py build --llama-dir . --out-dir 4218-baseline
LLAMA_GRAMMAR_GSS=1 python3 path/to/llamacpp-4218-grammar-engine/harnesses/run_llamacpp_4218_validation.py build --llama-dir . --out-dir 4218-gss
```

4. Build and run `harnesses/equivalence_harness.cpp` with a vocab-only GGUF to compare old and GSS accept sets in-process.
5. Build and run `harnesses/independent_falsify.cpp` with a vocab-only GGUF for the independent adversarial corpus.

See `harnesses/README.md` for concrete compiler commands.
