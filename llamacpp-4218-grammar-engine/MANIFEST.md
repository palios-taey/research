# Artifact manifest

- `README.md`: public root-cause foundation, three-register status, validation blind spots, and reproduction steps.
- `artifacts/measured-repro.md`: minimal exponential reproduction.
- `artifacts/realistic-schema-case.md`: realistic recursive JSON-schema-derived case and controls.
- `artifacts/design-synthesis.md`: root mechanism and in-place repair target.
- `artifacts/implementation-results.md`: first-pass prototype measurements and audit defects.
- `artifacts/equivalence-validation.md`: differential validation corpus, counts, and blind spots.
- `artifacts/independent-falsification.md`: independently designed adversarial falsification corpus, counts, and blind spots.
- `harnesses/`: validation, benchmark, equivalence, and falsification source files.
- `patches/README.md`: patch status warning.
- `patches/llamacpp-4218-gss-recognizer.patch`: cleaned first-pass prototype diff retained for audit and reproduction.

The patch diff intentionally excludes commit metadata and contains only file diffs. It is not presented as the final fix.
