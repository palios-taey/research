# Iterations — three dead ends before the validated fix

The fix went through three discarded candidates. Each was killed by a concrete
counterexample executed on a built binary (the binary, not review consensus, was
the arbiter). Five independent model reviewers cross-checked each candidate
under a default-refute mandate; where reviewers disagreed, the binary decided.
This record is kept because the dead ends are the interesting part — each one
looked correct until a counterexample ran.

## v1 — validate via the candidate-filter oracle (DEAD)

Idea: before committing, re-validate the token through the same
`llama_grammar_reject_candidates` the logit filter uses; return without mutating
if rejected.

Killed by four binary counterexamples:

- **CX-1 / CX-4** — for token-rule grammars (`<[id]>` / `!<[id]>`), the filter
  oracle accepts a piece the real mutator cannot perform, so the crash is *not*
  prevented (still throws).
- **CX-2** — early-returning on reject skipped the UTF-8 decoder-state update that
  upstream always performs, **stranding `partial_utf8` and deadlocking** every
  subsequent token (worse than the original bug).
- **CX-3** — the partial-character check rejected a valid split multi-byte
  codepoint that upstream accepts (a regression).

Root cause of the dead end: the candidate filter is **not** an exact transition
oracle for the commit path. Using it as a preflight was the wrong shape.

## v2 — transactional stacks, but commit `partial_utf8` unconditionally (DEAD)

Idea: run the real transition into a local `stacks_new`; commit stacks only if
non-empty; **always** commit `partial_utf8`.

Cleared CX-1..4. Killed by a new counterexample, found independently by three
reviewers and confirmed on the binary:

- **CX-5** — on a multi-codepoint piece that the grammar rejects mid-string while
  a trailing partial byte remains, the stacks roll back but `partial_utf8`
  advances to the whole-piece residue → a torn transaction → a later continuation
  byte **falsely completes** a codepoint that was never contiguously emitted.

Root cause: committing the two state registers (`stacks`, `partial_utf8`) on
different conditions is a split transaction.

## v3 — commit both registers together (CORE FIX; one residual)

Idea: commit `stacks` **and** `partial_utf8` under the *same* non-empty
condition — both, or neither.

Cleared CX-1..5 on the binary, preserved the CX-2 resync, and matched upstream
`master` byte-for-byte (FNV64) on the valid corpus. The grammar logic was
unanimously cleared by all five reviewers. One residual, non-grammar item
remained:

- **CX-6** — the once-only warning used a plain `static bool`, a data race across
  concurrent server slots (reproduced under ThreadSanitizer).

## v3.1 — atomic the once-only flag + actionable warning (VALIDATED)

- `static std::atomic<bool>` with `exchange(true, relaxed)` — TSAN-clean.
- The warning now carries the offending piece + token id (the diagnostic the
  original throw message had), so graceful degradation stays observable.

No other change from v3. This is the validated shape in `fix.diff`. Full results
in `VALIDATION.md`.
