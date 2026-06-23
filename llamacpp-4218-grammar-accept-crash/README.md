# llama.cpp #4218 — grammar accept-path crash: transactional commit fix

A focused correctness/availability fix for a server-terminating crash in
llama.cpp's GBNF grammar sampler, reached on the token-commit path
(`llama_grammar_accept_token` / `llama_grammar_accept_str`).

This package documents the bug, the validated fix, the multi-iteration
adversarial validation process that produced it (three dead iterations before
the validated one), and the public-safe evidence.

> Scope note. This is the **focused crash fix** on the existing stack-based
> grammar engine. It is distinct from — and supersedes, as the practical
> contribution — the exploratory chart/recognizer **rewrite** documented in
> `../llamacpp-4218-grammar-engine/`. The rewrite addressed the exponential
> blow-up that #4218's title asks about but is not mergeable in practice
> (a second grammar engine, common-case perf regression, and `llguidance`
> already exists upstream for that need). The crash documented here is a
> separate, real, reachable defect on the shipping engine.

## The bug (Observed)

Grammar sampling has two code paths that must agree on whether a token is
acceptable:

- **filtering** (`llama_grammar_apply_impl`) — masks the logits so only
  grammar-conformant tokens can be sampled, and
- **commit** (`llama_grammar_accept_token` / `_str`) — advances the grammar
  state once a token is chosen.

On the *forced-accept* paths — lazy-grammar trigger replay, and special/added
control tokens emitted by chat templates and tool-calling machinery — a token
can reach **commit** without having passed **filtering**. When such a token is
not grammar-conformant, the commit path mutates the grammar stacks to empty and
then throws:

```
Unexpected empty grammar stack after accepting piece: ...
```

The throw is uncaught (it propagates through `common_sampler_accept` to
`std::terminate`), so a single non-conformant forced token **terminates the
whole server** — not just the one request. This is reachable across many models
and grammars (Gemma `<unused*>` tokens, DeepSeek/Qwen3-Coder tool-call
delimiters, GLM-4.5-Air, plain `json.gbnf`), which is why it recurs.

Maps to open issues **#23677** and **#14413** (and the earlier #19353 / #21017).

## The fix (Observed)

Make the commit **transactional**: run the real transition into a *local* copy
of the stacks, and commit the new grammar stacks **and** the UTF-8 decoder state
together **iff** the result is non-empty. If the forced token would empty the
stacks, the call is a full no-op (both registers stay at their pre-call values —
the non-conformant token is ignored, generation continues under the unchanged
grammar) with a once-only diagnostic warning naming the offending token.

See `fix.diff` (production change is ~30 lines in `src/llama-grammar.cpp`; it
**deletes both `throw` sites**). Key properties, all validated on a real binary
(see `VALIDATION.md`):

- **No server crash** on a forced non-conformant token (the #23677 / #14413 DoS).
- **Byte-identical to upstream `master` on all valid inputs** (differential FNV64
  match) — this is not a behavior change for conformant generation; it only
  replaces "crash" with "ignore + continue" on the non-conformant forced path.
- The transactional inner-loop shape is **already upstream** in `master`'s
  `accept_token`; this completes the same pattern in `accept_str` and converts the
  terminal `empty → throw` into `empty → no-op`.
- Negligible perf delta (within run-to-run noise); the once-only warning is
  thread-safe and does not fire on valid generation.

## Why no-op-and-continue (not error-the-request)

The crashing tokens are control/special tokens emitted by the template and
tool-calling machinery — not user content. On the generation hot path there is
no clean request boundary at which to fail; surfacing an exception mid-stream is
exactly the denial-of-service. Ignoring the non-conformant forced token and
letting generation re-converge under the unchanged grammar keeps the server and
the request alive; the once-only warning (carrying the token id) preserves
observability so a genuine upstream template/sampler issue is still diagnosable.

## Contents

- `fix.diff` — the production diff against upstream `master` (grammar source only;
  the full branch additionally carries regression tests).
- `ITERATIONS.md` — the three dead iterations and what each one's binary
  counterexample taught, before the validated shape.
- `VALIDATION.md` — the binary-as-oracle methodology, the counterexample battery,
  the differential / perf / sanitizer results, and limitations.

## Register

- **Observed**: the crash, the fix shape, and every result in `VALIDATION.md`
  were reproduced on a built binary on real hardware with a real 248k-vocab GGUF.
- **Inferred**: the transactional commit preserves upstream semantics on all
  valid inputs (supported by the differential FNV64 match + source equivalence of
  the per-codepoint transition).
- **Unknown**: upstream review/merge outcome; language equivalence outside the
  tested grammar corpus.
