# Artifact manifest

Current public package for llama.cpp #4218 grammar engine validation.

- `README.md`: public overview, branch pointer, current status, and validation snapshot.
- `AUDIT_TRAIL.md`: five independent automated review findings and corresponding fixes.
- `VALIDATION.md`: commands, observed outputs, trace-equivalence result, and limitations.
- `harnesses/README.md`: current harness build/run instructions.
- `harnesses/current/test-grammar-compactor.cpp`: current focused compactor and regression harness source.
- `harnesses/current/test-grammar-differential.cpp`: current expanded differential harness source.
- `artifacts/measured-repro.md`: historical minimal exponential reproduction.
- `artifacts/realistic-schema-case.md`: historical recursive JSON-schema-derived reproduction.
- `artifacts/design-synthesis.md`: historical root mechanism and repair target.
- `artifacts/implementation-results.md`: historical first-pass prototype measurements and audit defects.
- `artifacts/equivalence-validation.md`: historical first-pass prototype differential validation.
- `artifacts/independent-falsification.md`: historical independent adversarial corpus.
- `harnesses/equivalence_harness.cpp`: historical first-pass prototype equivalence harness.
- `harnesses/independent_falsify.cpp`: historical independent falsification harness.
- `harnesses/reject_apply_bench.cpp`: historical reject-path benchmark harness.
- `harnesses/run_llamacpp_4218_validation.py`: historical growth validation helper.
- `harnesses/helper_sources/llamacpp_4218_probe.cpp`: historical probe source.
- `harnesses/fixtures/synthetic_ambiguous_wrappers.gbnf`: historical fixture.
- `patches/README.md`: historical prototype patch warning.
- `patches/llamacpp-4218-gss-recognizer.patch`: historical prototype diff, not the current fix.

The active fix is the public branch linked from `README.md`, not the historical prototype patch.
