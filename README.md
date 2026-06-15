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

### `research/training-and-retrieval-stack/` — Production Training Pipeline + Multi-Substrate Retrieval (2026)

**Engineering portfolio for the full PALIOS-TAEY training stack.**
Six production checkpoints (Qwen3.5-35B-A3B MoE + Qwen3.5-9B Dense, FSDP on a
4-node DGX Spark GB10 cluster) with intact recipe + corpus + weights triplets,
and a multi-substrate retrieval system (Weaviate + Neo4j + Redis) with
query-adaptive routing, multi-scale memory, and HMM motif memory across three
coordinated stores.

Headline measured results (verified against an internal canonical-metrics file):

- Config A2 keystone-attention LoRA DPO refinement: **84.7%** on the 163-probe
  audit (+1.9pp over the 82.8% SFT baseline); all 8 infra-control categories held.
- Phase 3 Recovery SFT (single-Spark, cross-validated): **train_loss 1.122
  identical on two independent runs**, after a 4-Spark FSDP wedge was root-caused
  to corpus memory pressure (RDMA send-queue saturation) and resolved by an
  offline conversation-level chunker.
- 4-Spark NCCL fabric: **10.23–12.57 GB/s** on the `reduce_scatter` synth probe
  at the failing 218M-numel size; ConnectX-7 28.45.4028 + NCCL 2.28.9.

Contents:

- `README.md` — paper-shaped lead document (headline metrics, novel architecture,
  engineering judgment under uncertainty, honest open questions).
- `TECHNICAL_APPENDIX.md` — full citation chains, file:line code references,
  three-register tables.
- `REPRODUCE.md` — step-by-step recipes (network setup, 35B path, 9B path,
  bake-and-test, ISMA stack).

See [`research/training-and-retrieval-stack/README.md`](research/training-and-retrieval-stack/README.md)
for the full portfolio.

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
