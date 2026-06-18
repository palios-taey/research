# Differential equivalence validation

The equivalence harness initializes both recognizers in one process and compares accept sets from `llama_grammar_apply_impl()` at each tested generation state.

Result: PASS, 0 divergences.

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
