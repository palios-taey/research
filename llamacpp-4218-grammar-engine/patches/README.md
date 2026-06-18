# Patch status

`llamacpp-4218-gss-recognizer.patch` is retained as first-pass prototype and audit material.

It is useful for reproducing the measured stack-sharing effect, but it is not the final fix and should not be applied as a merge proposal. A later audit found:

- `CHAR_ALT` multi-range under-acceptance.
- Raw-byte token under-acceptance.
- Clone-related undefined behavior risk.

The active repair target is an in-place root-cause fix that merges reconverged parse threads at equivalent `(nonterminal, input position)` states while preserving existing grammar-engine semantics.
