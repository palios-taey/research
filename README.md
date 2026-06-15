# palios-taey/research

Public research artifacts from the PALIOS-TAEY project. Each subdirectory is a
self-contained piece of work with its own README. Use this top-level index to
find specific artifacts.

## Currently public

### `research/audit-harness-moe/` — Paired-Capability-Control Audit Harness for MoE Fine-Tunes

**163-probe behavioral audit harness for Mixture-of-Experts fine-tunes.**
Distinguishes genuine behavioral lift from style mimicry / deception-shaped
surface effects using a composite promotion gate (per-category behavioral
probes + orthogonal capability controls).

Headline empirical result (Config A2 case study on Qwen3.5-35B-A3B):

- Full-surface DPO: nominally passed target categories but **regressed
  infrastructure reasoning from 4/4 → 1/4** — a deception-shaped failure mode
  invisible to single-axis behavioral evaluation.
- Keystone-Only DPO (LoRA on 6 keystone layers, 159 experts frozen):
  **84.7% audit pass rate** (+1.9pp over SFT baseline), regression restored.

Contents:

- `TAEY_AUDIT_V2.json` — canonical machine-readable probe set (163 probes,
  19 categories, paired controls).
- `audit_pipeline.py` — composite promotion gate + scoring logic.
- `soma_proxy.py` — auditor-service implementation (LLM-as-judge via Anthropic API).
- `frozen_experts_v4_1_polysemantic.json` — FQN-level expert-freeze mask.
- `launch_religion_dpo_v2.sh` — production training launcher for the
  Config A2 recipe.
- `DESIGN.md` — paired-control taxonomy and category documentation.

See [`research/audit-harness-moe/README.md`](research/audit-harness-moe/README.md)
for full usage.

### `research/training-stack/` — Production Training Stack (2026)

**Real and verifiable engineering portfolio.** Production training recipes
(Qwen3.5-35B-A3B MoE + Qwen3.5-9B Dense, FSDP on a 4-node DGX Spark GB10 cluster)
with the actual scripts that ran, the actual configs, the audit-harness verdicts
for each shipped checkpoint, and a `METRICS_PROVENANCE.md` mapping every headline
number to a specific file in the repository.

Headline measured results:

- Config A2 keystone-attention LoRA DPO refinement: **84.7% (138/163)** on the
  163-probe constitutional audit, **+1.9pp** over the 82.8% SFT baseline; all
  8 infra-control categories held (4/4 restored from DPO v1's 1/4 regression).
  Verdict in `audit_results/religion_dpo_v2/audit_v2/SUMMARY.md`; per-probe
  responses in `results.txt`; the launcher that produced it in
  `recipes/launch_religion_dpo_v2.sh`.
- 4-Spark NCCL fabric verified at **10.23–12.57 GB/s** on the `reduce_scatter`
  synth probe at the failing 218M-numel size; ConnectX-7 28.45.4028 + NCCL 2.28.9.
- Phase 3 Recovery SFT (single-Spark, cross-validated wedge-fix proof): identical
  train_loss on two independent runs after the offline conversation chunker
  resolved a 4-Spark FSDP wedge root-caused to corpus memory pressure.

Contents (the actual artifacts, not just write-ups):

- `recipes/` — 13 launcher scripts (the actual `.sh` that ran on the cluster).
- `trainers/` — 4 Python files (Phase 3 trainer, Phase 2 CPT trainer, offline
  chunker, bake script).
- `configs/` — 10 config files (159-expert freeze mask, Phase 2 expert config
  44KB, FSDP YAMLs, DeepSpeed JSON variants, accelerate config).
- `audit_results/` — 11 per-checkpoint audit verdicts (~13 MB of real
  measured-output: per-category pass/fail, per-probe model responses, failure
  exemplars).
- `proof_of_run/` — 4-Spark fabric verification results.
- `METRICS_PROVENANCE.md` — every headline number → its proof file.
- `README.md`, `TECHNICAL_APPENDIX.md`, `REPRODUCE.md` — documentation that
  links to the above.

The retrieval stack (ISMA) lives in a separate repository at
[`palios-taey/isma-core`](https://github.com/palios-taey/isma-core); this
subdirectory is training only.

See [`research/training-stack/README.md`](research/training-stack/README.md)
for the full portfolio.

### `research/ml-stack-fuzzing/` — Fuzzing the ML Inference Stack + Release-Significance Triage

**A reusable methodology for finding real memory-safety / denial-of-service
bugs in the native untrusted-input path of LLM serving** — model-file loaders,
tokenizers, and the grammar / structured-output compilers in front of
constrained decoding (vLLM / SGLang-class engines).

The contribution is one discipline most fuzzing write-ups skip:

- **A sanitizer crash is not production impact.** An ASAN build aborts on the
  first out-of-bounds byte; a release build usually absorbs that read and keeps
  running. The included triage tool re-runs every unique crash on a
  **no-sanitizer release build** and escalates only the `RELEASE-SEGV` subset —
  the crashes a real deployment would also hit — instead of the much larger raw
  sanitizer-crash count.

Contents:

- `release_significance_triage.py` — target-agnostic triage: dedup →
  ASAN-classify → group → **release-significance filter** → escalate
  (CI-friendly, exits non-zero on a production-significant finding).
- `harness/fuzz_target_skeleton.cc` — multiplexed libFuzzer target skeleton
  (every untrusted entry point behind one harness; catches expected validation
  failures so the corpus deepens).
- `harness/build.sh` — sanitizer build template with the non-obvious flags
  annotated (no-LTO, debug-asserts-off, release-shaped inlining).

Worked example: applied to a native structured-output / grammar-compilation
library on the inference path; surfaced a denial-of-service finding currently
in **GitHub coordinated disclosure (in triage, not yet public)** — affected
library, entry points, and reproducing inputs are intentionally withheld until
the advisory publishes, at which point the section is updated with the
verifiable advisory link.

See [`research/ml-stack-fuzzing/README.md`](research/ml-stack-fuzzing/README.md)
for the full methodology.

## License

Apache 2.0 — see `LICENSE`. Individual subdirectories may carry additional
license headers; the repository default is Apache-2.0 unless stated otherwise.

The PALIOS-TAEY constitutional / governance documents live in a separate
repository at [palios-taey/governance](https://github.com/palios-taey/governance)
under the Sacred Trust License v1.0; that license does **not** apply to code
artifacts in this repository.

## Provenance

Artifacts here are ported from internal working repos (`infra-soul`,
`embedding-server`, `the-conductor`, etc.) once they reach a state where the
underlying methodology is defensible as a standalone contribution and the
empirical claims can be cited as Observed rather than Inferred. Initial port
of `audit-harness-moe`: 2026-05-12.
