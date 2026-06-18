# Design synthesis

The existing implementation stores concrete continuation stacks. Exact-vector deduplication only helps when two continuations are byte-for-byte identical at the same point. In the failure cases, many continuations differ temporarily and then reconverge at the same grammar slot and input position.

The proven mechanism is to merge parse threads that reconverge at the same `(nonterminal, input position)` rather than duplicating each concrete continuation vector. A graph-structured stack or equivalent chart-style representation is the right mechanism class.

The current repair target is an in-place root-cause fix:

- Share continuations by grammar slot and input position instead of duplicating every full stack vector.
- Keep the current grammar parser and grammar syntax.
- Preserve existing `CHAR_ALT`, raw-byte token, partial-UTF-8, clone, reset, and stack-access semantics.
- Avoid an augment layer that narrows acceptance or changes API behavior outside the stack-sharing objective.

Rejected narrower fixes:

- Interning suffix vectors alone does not prevent the exponential number of live concrete stack heads.
- A special-case rewrite for one ambiguous grammar family does not cover realistic recursive-schema cases.
- Eager branch pruning is unsafe without proving identical future language acceptance.
- The first-pass opt-in augment prototype demonstrated the performance mechanism but is not the final fix because audit found silent under-acceptance and clone-safety defects.

Required correctness gates:

- Contingent-pop behavior must match the old recognizer.
- Nullable bodies and epsilon cycles must not introduce infinite worklists.
- Hidden left-recursive grammars rejected by the shared parser must remain rejected.
- `llama_grammar_apply_impl()` accept sets must match the old recognizer across full-vocabulary states where feasible.
- Focused tests must cover `CHAR_ALT` multi-range acceptance, raw-byte token acceptance, clone behavior, and any public stack-access expectations.
