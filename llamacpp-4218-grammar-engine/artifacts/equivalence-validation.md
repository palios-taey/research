# Differential equivalence validation

The equivalence harness initializes the baseline recognizer and the first-pass prototype in one process and compares accept sets from `llama_grammar_apply_impl()` at each tested generation state.

Observed result for this harness: PASS, 0 divergences.

| Metric | Count |
|---|---:|
| Grammars generated or loaded | 256 |
| Grammars built by both recognizers | 238 |
| Grammars rejected by both recognizers | 18 |
| Sequence pairs | 3,453 |
| Compared generation states | 16,804 |
| Token decisions compared | 138,018,304 |
| Full-vocabulary states | 468 |
| Sampled-vocabulary states | 16,336 |
| Divergences | 0 |

Corpus contents:

- Existing-test-style grammars for integer schemas, expressions, token delimiters, complex token sections, ellipsis, exact repetition, and nullable repetition.
- Hand adversarial grammars for ambiguous wrappers, nullable-body repetition, highly ambiguous recursion, deep nesting, nullable helpers, hidden left recursion, and epsilon cycles.
- The recursive workflow grammar from the realistic fixture.
- 240 generated non-left-recursive GBNF grammars with random walks.

Parser-rejected hidden-left-recursion and epsilon-cycle cases were counted as common build rejections, not runtime accept-set evidence.

## Known blind spots

This harness did not prove the first-pass prototype correct. A later code audit found defects that this differential corpus missed:

- `CHAR_ALT` multi-range under-acceptance.
- Raw-byte token under-acceptance.
- Clone-related undefined behavior.

The lesson is methodological: high-volume differential fuzzing is necessary but insufficient. The final root-cause fix needs focused tests for grammar-element semantics and API lifetime behavior, plus code review against those invariants.
