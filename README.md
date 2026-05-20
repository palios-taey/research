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
