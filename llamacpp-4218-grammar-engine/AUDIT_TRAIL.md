# Audit trail

This file records five independent automated review findings that drove concrete branch changes. Reviewer labels are generic by design; no private session names or host details are required to validate the findings.

## RE-AUDIT UPDATE (2026-06-22): Finding 1 is REOPENED — not fully closed

A clean re-audit on this commit found that Finding 1's fresh-column repair is **incomplete**. The repair triggers only when a token yields zero complete code points (`src/llama-grammar.cpp:2118`, gated on `code_points.size() == 1`). A token piece containing **one or more complete code points followed by a trailing incomplete UTF-8 sequence** bypasses the repair and can preserve a stale `TOKEN` frontier across the token boundary — an over-accept ("branch laundering" across tokens).

Important honesty note: this defect is **not** caught by byte-identical comparison to upstream master, because master has the same gap, and the differential corpus lacked a mixed complete-plus-partial token piece. So the earlier "byte-identical to master" evidence does not establish closure of this case.

Required fix: generalize the fresh-frontier handling to **every** token whose decoded partial remainder is nonzero (`n_remain != 0`), not only the zero-complete-codepoint case, and apply the same partial-frontier filtering to `llama_grammar_accept_str()`. **Status: fix in progress.** This engine is **NOT submission-ready** until Finding 1 is fully closed and re-validated against a mixed-piece corpus. Findings 2–5 and the compactor/clone/cache/EOG checks were confirmed closed in the re-audit.

Current audited branch tip:

```text
3e4aa92bed972d60cbf9a02795d40bed10a60338
```

## Review Findings And Fixes

| Review | Finding | Fix commit | Regression evidence |
|---|---|---|---|
| Review A | Partial UTF-8 token acceptance needed a fresh-column frontier so split code points could complete correctly without stale state. | `923451992f76d712af4f576afd4ac0886f86d954` | Split UTF-8 cases in grammar integration and differential coverage. |
| Review B | The chart engine still retained too many sealed origins for `a*`; recognizer state needed safe sealed-origin compaction rather than only chart-item replacement. | `a5ec9c683f470b2c5d18f392104ddc0f1f78540d` | `astar n=100000 origins=2 sealed=1 current_items=5 stored_items=5 resume_entries=1`. |
| Review C | Completion could append into the same vector being iterated, creating iterator invalidation risk under high fan-in completion. | `ba94749d40a154ebc69546fbdb77d51108f04227` | `completion-reallocation-stress callers=8191 advanced=8191 final_items=16384`. |
| Review D | Pure token candidates without code points could be accepted from a non-empty chart instead of requiring the token element to match, affecting `TOKEN_NOT` behavior. | `ba94749d40a154ebc69546fbdb77d51108f04227` | `token-not-empty-candidate excluded=0 allowed=1`. |
| Review E | End-of-generation checks should explicitly reject while `partial_utf8.n_remain > 0`, even when the grammar has a nullable alternative. | `3e4aa92bed972d60cbf9a02795d40bed10a60338` | `nullable-utf8-eog partial=0 completed=1 wrong=0`. |

## Additional Audit Checks

- Observed: the current branch contains no tracked references to the old public stack-access shim names `get_stacks`, `stacks_cur`, or `stacks_org`.
- Observed: candidate filtering and committed acceptance both use the same EOG predicate after commit `3e4aa92bed972d60cbf9a02795d40bed10a60338`.
- Observed: the expanded differential trace is byte-identical between the precompactor baseline and the current branch for the copied harness corpus.
- Inferred: the review trail reduced the major known risks from the older prototype: UTF-8 frontier mismatch, unbounded retained origin state, completion lifetime hazards, token-terminal under/over-acceptance, and nullable UTF-8 EOG leakage.
- Unknown: independent upstream maintainer review may still find additional edge cases.
