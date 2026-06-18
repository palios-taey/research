# Design synthesis

The existing implementation stores concrete continuation stacks. Exact-vector deduplication only helps when two continuations are byte-for-byte identical at the same point. In the failure cases, many continuations differ temporarily and then reconverge at the same grammar slot and input position.

The selected design is to add a graph-structured stack recognizer for grammar matching:

- Share continuations by grammar slot and input position instead of duplicating every full stack vector.
- Keep the current grammar parser and grammar syntax.
- Keep the current path available for compatibility and differential testing.
- Gate the new recognizer behind `LLAMA_GRAMMAR_GSS=1` while it is under review.

Rejected narrower fixes:

- Interning suffix vectors alone does not prevent the exponential number of live concrete stack heads.
- A special-case rewrite for one ambiguous grammar family does not cover realistic recursive-schema cases.
- Eager branch pruning is unsafe without proving identical future language acceptance.

Required correctness gates:

- Contingent-pop behavior must match the old recognizer.
- Nullable bodies and epsilon cycles must not introduce infinite worklists.
- Hidden left-recursive grammars rejected by the shared parser must remain rejected.
- `llama_grammar_apply_impl()` accept sets must match the old recognizer across full-vocabulary states where feasible.
- The `get_stacks()` shim must be documented as a compatibility frontier, not a full reconstruction of historical concrete stack vectors.
