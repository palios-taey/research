# Structured-output soundness — differential testing the constrained-decoding stack

Constrained decoding (xgrammar, outlines, llguidance — the backends behind vLLM, SGLang, and guidance) compiles a JSON Schema into a decoding grammar so a model can only emit conforming output. A **soundness bug** is when that grammar accepts output the schema forbids: the "constrained" decoder silently emits schema-invalid JSON. These are easy to miss, because the engine compiles without error and the output looks plausible.

This is a method for finding them, plus two worked examples from xgrammar that became upstream fixes.

## The method

Differential testing against a ground-truth validator:

1. Build a corpus of JSON Schemas, weighted toward the features that are hard to compile into a grammar.
2. For each schema, generate candidate JSON strings — valid and invalid.
3. Ground truth: a real JSON Schema validator (`jsonschema`, Draft 2020-12) decides accept/reject.
4. For each engine, compile the schema and test whether the matcher accepts the candidate.
5. Flag divergences. A **false-accept** (engine accepts a schema-invalid string) is the high-severity case — the constraint is unsound.
6. Minimize each divergence to the smallest schema + string, and confirm it against the spec.

`difftest_harness.py` is the runner. It drives the engines through their CPU matcher APIs, so the whole thing is hardware-agnostic and reproducible anywhere.

## Worked example — xgrammar `multipleOf` (upstream PR)

The first numeric class surfaced a real one. xgrammar silently dropped the `multipleOf` keyword during schema conversion, so the generated grammar never constrained on it:

```
{"type": "number", "multipleOf": 2}   ->   matcher accepts "3" and "5.0"
```

Ground truth rejects both. Root cause: `multipleOf` was handed to a no-op warning path and there was no field for it in the integer/number spec, so generation ignored it.

The fix makes the integer case sound and fails closed everywhere else. A `type:integer` with a positive integral `multipleOf` generates a modulo-N digit-divisibility DFA, so only true multiples parse (zero, negatives, and no-leading-zeros handled); a small finite range enumerates the valid multiples over the intersected bounds. Cases that can't be made sound with the current generator — `type:number` `multipleOf`, non-integral or out-of-range values, `multipleOf` with an unbounded or large range — raise a clear error instead of producing an unsound grammar.

`xgrammar-multipleof.diff` is the change. Upstream PR: https://github.com/mlc-ai/xgrammar/pull/670 (open).

The fix went through binary-as-oracle validation (a sweep checking grammar acceptance against numeric divisibility across a range, matched to `jsonschema`) and a multi-reviewer adversarial audit that caught a second false-accept — a non-intersected-bounds case where `{minimum:10, exclusiveMinimum:3, maximum:20, multipleOf:5}` wrongly accepted `5` — before submission.

## Worked example — xgrammar `oneOf`

The composition class surfaced a second one. `oneOf` was compiled as `anyOf`, so the grammar accepted values matching more than one branch — `oneOf` means exactly one, and that semantic was lost:

```
{"oneOf": [{"type": "integer"}, {"type": "number"}]}   ->   matcher accepts "1"
```

Ground truth rejects `1`, because it matches both branches. Exactly-one is only sound when the branches are mutually exclusive, so the fix proves that before compiling. A conservative prover checks pairwise disjointness by primitive type-set (integer overlaps number), by exact `const`/`enum` value, or by a strict object discriminator. When every pair proves disjoint the `oneOf` compiles as the existing union, which is then exact-one; everything else fails closed with a clear error rather than emit an unsound grammar. The numeric comparison stays conservative — two numbers prove distinct only when both are exact integers, so a precision-lossy case like a large integer against its float spelling fails closed instead of wrongly unioning.

`xgrammar-oneof.diff` is the change. It went through the same binary-as-oracle validation and a five-reviewer adversarial audit; that audit caught a real precision false-accept (a value at the `2^53` double boundary) that was fixed conservatively before sign-off.

## Status (cannot-lie)

- xgrammar `multipleOf`: PR **submitted, open, not merged** as of this writing — https://github.com/mlc-ai/xgrammar/pull/670
- xgrammar `oneOf`: fix **built, validated, and audited; upstream PR prepared, not yet submitted** as of this writing.
- The method generalizes to harder schema classes (patternProperties, `$ref` recursion, `format`) and the other engines (outlines, llguidance); those sweeps are ongoing.

## Register

- **Observed**: the false-accepts, the fix, and the validation results were reproduced on built engines with a real ground-truth validator.
- **Inferred**: a soundness bug in xgrammar/llguidance propagates to the inference servers that use them.
- **Unknown**: upstream review/merge outcome.
