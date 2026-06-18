# llama.cpp #4218 root-cause foundation

This artifact is a public research foundation for llama.cpp issue #4218. It documents the measured exponential blowup, the root mechanism, validation harnesses, and the audit status of a first-pass prototype.

It does not present the included prototype patch as the final fix.

## Current status

- Observed: the baseline grammar engine can reach `3 * 2^n` active continuation stacks on a minimal ambiguous recursive grammar. At `n = 16`, measured baseline runs reached `196,608` stacks.
- Observed: a first-pass opt-in graph-structured-stack prototype collapsed the measured pathological stack counts on the synthetic and realistic fixtures.
- Observed: a later code audit found correctness defects in that first-pass prototype: `CHAR_ALT` multi-range under-acceptance, raw-byte token under-acceptance, and clone-related undefined behavior.
- Inferred: the root-cause target is still stack sharing, but the implementation should be rebuilt in place around the existing grammar engine semantics rather than shipped as the first-pass augment shape.
- Unknown: the final in-place root-cause patch is not included here. The differential fuzzing results below are useful evidence, not a proof.

## Problem

The current grammar engine can enumerate exponentially many continuation stacks for ambiguous recursive grammars. A minimal family is:

```gbnf
root ::= nest
nest ::= "a" nest "b" | "a" nest "c" |
```

On input `a^n b^n`, the engine reaches `3 * 2^n` active stacks. At `n = 16`, the measured baseline reached `196,608` stacks. A representative baseline validation run spent `42,900.235 ms` in token acceptance for that case; a separate initial repro run on the same grammar measured `32,799.244 ms`.

The same failure mode appears in realistic JSON-schema-derived grammars. A recursive workflow schema with a late `strategy` discriminator reached `81,920` active stacks at depth 13 and spent `57,554.288 ms` in token acceptance. Controls that moved the discriminator earlier or removed the recursive union stayed under 10 stacks.

## Mechanism

The mechanism is reconvergence without sharing. The engine keeps concrete continuation-stack vectors, so parse threads that temporarily differ and later reach the same grammar state remain duplicated. The root-cause repair target is to merge threads that reconverge at the same `(nonterminal, input position)` while preserving existing token, byte, `CHAR_ALT`, partial-UTF-8, clone, and stack-access semantics.

Graph-structured stack or equivalent chart-style sharing is the right mechanism class. The implementation detail still matters: sharing must be integrated without narrowing the accepted language.

## First-pass prototype result

| Case | Baseline max stacks | Prototype max stacks | Baseline accept time | Prototype accept time |
|---|---:|---:|---:|---:|
| Synthetic `a^n b^n`, n=16 | 196,608 | 8 | 42,900.235 ms | 0.034 ms |
| Recursive workflow schema, depth 13 | 81,920 | 10 | 57,554.288 ms | 0.371 ms |

These numbers show that merging reconverged parse threads addresses the performance mechanism. They do not certify the first-pass patch as correct or mergeable.

Reject-path benchmarking with 128k token candidates also improved in the measured case: median `llama_grammar_apply_impl()` time was `4,654.760 us` on the baseline path and `3,811.255 us` on the first-pass prototype path.

## Validation

Differential validation compared old and prototype accept sets at generation states using the real `llama_grammar_apply_impl()` implementation in both modes.

| Corpus | Grammars built by both | States | Token decisions | Divergences |
|---|---:|---:|---:|---:|
| Existing-test-style plus generated corpus | 238 | 16,804 | 138,018,304 | 0 |
| Independent adversarial falsification corpus | 2,275 | 30,298 | 1,522,686,586 | 0 |
| Combined | 2,513 | 47,102 | 1,660,704,890 | 0 |

The independent corpus specifically targeted failure modes expected to break naive stack merging: contingent-pop behavior, mutually recursive cycles, nullable epsilon cycles, nullable repetition bodies, high reconvergence ambiguity, and long-range dependencies.

## Validation blind spots

This is adversarial differential testing, not a formal proof of language equivalence. It missed silent under-acceptance defects later found by code audit:

- `CHAR_ALT` multi-range handling accepted too little in the first-pass prototype.
- Raw-byte token handling accepted too little in the first-pass prototype.
- Clone behavior had undefined-behavior risk in the first-pass prototype.

The `get_stacks()` compatibility shim in the prototype exported the active terminal frontier compactly. It was sufficient for the validation harnesses here, but it did not reconstruct every historical concrete continuation vector from the shared representation. That is another reason the prototype is audit material, not the final fix.

## Root-cause target

The target for a publishable fix is an in-place grammar-engine change that:

- Preserves current language acceptance for all token forms and grammar element types.
- Merges parse threads only when they reconverge at equivalent `(nonterminal, input position)` states.
- Avoids concrete continuation-vector explosion in recursive ambiguous grammars.
- Keeps clone, reset, stack access, and partial UTF-8 behavior within existing API expectations.
- Uses differential fuzzing plus focused code review and targeted unit tests for the audited blind spots.

## How to reproduce

1. Check out llama.cpp.
2. Run the baseline growth harness to reproduce the problem.

```bash
python3 path/to/llamacpp-4218-grammar-engine/harnesses/run_llamacpp_4218_validation.py build --llama-dir . --out-dir 4218-baseline
```

3. Optional historical prototype check: apply `patches/llamacpp-4218-gss-recognizer.patch` only if you want to reproduce the first-pass stack-sharing measurements. Do not treat that patch as the final fix.

```bash
LLAMA_GRAMMAR_GSS=1 python3 path/to/llamacpp-4218-grammar-engine/harnesses/run_llamacpp_4218_validation.py build --llama-dir . --out-dir 4218-gss
```

4. Build and run `harnesses/equivalence_harness.cpp` with a vocab-only GGUF to compare baseline and prototype accept sets in-process.
5. Build and run `harnesses/independent_falsify.cpp` with a vocab-only GGUF for the independent adversarial corpus.

See `harnesses/README.md` for concrete compiler commands.
