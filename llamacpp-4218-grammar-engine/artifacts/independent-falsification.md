# Independent adversarial falsification

A separate harness generated taxonomy-driven grammars designed to falsify naive stack-sharing equivalence. It used full-vocabulary accept-set comparisons at every walked state.

Observed result for this harness: PASS, 0 divergences.

| Metric | Count |
|---|---:|
| Taxonomy grammars generated | 2,520 |
| Built by both recognizers | 2,275 |
| Rejected upfront by both recognizers | 245 |
| Build mismatches | 0 |
| Token sequence walks | 9,100 |
| Compared generation states | 30,298 |
| Full-vocabulary token decisions compared | 1,522,686,586 |
| Divergences | 0 |

| Class | Grammars | Built both | Decisions | Divergences |
|---|---:|---:|---:|---:|
| contingent-pop | 420 | 420 | 262,190,769 | 0 |
| deep-mutual-indirect-recursion | 420 | 315 | 240,278,717 | 0 |
| highly-ambiguous-reconvergence | 420 | 420 | 257,416,354 | 0 |
| long-range-dependency | 420 | 420 | 332,600,826 | 0 |
| nullable-bodies-in-repetition | 420 | 420 | 249,827,547 | 0 |
| nullable-epsilon-cycles | 420 | 280 | 180,372,373 | 0 |

This corpus was designed separately from the first equivalence harness. Its purpose was to find counterexamples in the areas most likely to hide stack-sharing bugs. It did not find one, but it is still adversarial testing rather than a formal proof.

## Known blind spots

The independent corpus missed audit-caught silent under-acceptance in the first-pass prototype:

- `CHAR_ALT` multi-range handling.
- Raw-byte token handling.
- Clone lifetime behavior.

The final root-cause fix should keep this harness, but it also needs focused semantic tests for those audited gaps.
