# Artifact manifest

- `README.md`: public summary, headline results, caveats, and reproduction steps.
- `artifacts/measured-repro.md`: minimal exponential reproduction.
- `artifacts/realistic-schema-case.md`: realistic recursive JSON-schema-derived case and controls.
- `artifacts/design-synthesis.md`: sanitized design rationale for the GLL/GSS-style recognizer.
- `artifacts/implementation-results.md`: build, performance, and compatibility notes.
- `artifacts/equivalence-validation.md`: differential validation corpus and counts.
- `artifacts/independent-falsification.md`: independently designed adversarial falsification corpus and counts.
- `harnesses/`: validation, benchmark, equivalence, and falsification source files.
- `patches/llamacpp-4218-gss-recognizer.patch`: cleaned patch diff for the candidate implementation.

The patch diff intentionally excludes commit metadata and contains only file diffs.
