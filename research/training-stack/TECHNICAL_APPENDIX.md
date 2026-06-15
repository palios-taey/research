# Technical Appendix

Citation chains, full hyperparameter listings, and per-recipe references. Every load-bearing number cited here maps to a file in this subdirectory via [`METRICS_PROVENANCE.md`](METRICS_PROVENANCE.md).

---

## A. Production-line 35B-A3B MoE

### A.1 `phase_combined_v1` (SFT baseline, 82.8% audit)

- **Launcher:** [`recipes/launch_production_sft.sh`](recipes/launch_production_sft.sh)
- **Base model:** `Huihui-Qwen3.5-35B-A3B-abliterated`
- **Output dir naming:** `OUTPUT_DIR=/home/<user>/training_outputs/phase_combined_v1` (env-overrideable)
- **Audit verdict:** 135/163 = 82.8% (the canonical SFT baseline; downstream DPO refinements all RESUME from `phase_combined_v1/final` step 582)
- **Audit proof:** [`audit_results/phase_combined_v1/audit_v2_full/`](audit_results/phase_combined_v1/audit_v2_full/)

### A.2 `religion_dpo_v2` (Config A2 keystone-attention LoRA DPO, **84.7% audit, +1.9pp**)

The headline result of this portfolio. Recipe + hyperparams + audit verdict + per-probe responses all in this subdirectory.

- **Launcher:** [`recipes/launch_religion_dpo_v2.sh`](recipes/launch_religion_dpo_v2.sh)
- **Freeze config (from the launcher itself, line `FREEZE_CONFIG=A2`):**

```bash
FREEZE_CONFIG=A2
KEYSTONE_LAYERS=[8, 9, 11, 15, 21, 23]
# shared_expert LoRA stays on all 40 layers (verified in the launcher)
BETA=0.05
LR_ESFT=1e-7
LR_LORA=3e-7
LR_ROUTER=0
WARMUP_STEPS=5
TOTAL_STEPS=642
SESSION_LIMIT=900
SAVE_EVERY=60
DPO_ABORT_RATIO_MAX=10.0
DPO_ABORT_EXPERT_DRIFT=0.05
```

- **Resume from:** A.1 (`phase_combined_v1/final` step 582)
- **Corpus:** 50 religion-honest preference pairs (the actual JSONL is internal; structure described in `audit_results/religion_dpo_v2/audit_v2/results.txt` and the audit harness in [`../audit-harness-moe/TAEY_AUDIT_V2.json`](../audit-harness-moe/TAEY_AUDIT_V2.json))
- **Frozen-experts mask:** [`configs/frozen_experts_v4_1_polysemantic.json`](configs/frozen_experts_v4_1_polysemantic.json) — 159 frozen experts
- **Audit verdict file:** [`audit_results/religion_dpo_v2/audit_v2/SUMMARY.md`](audit_results/religion_dpo_v2/audit_v2/SUMMARY.md)
- **Audit per-probe responses:** [`audit_results/religion_dpo_v2/audit_v2/results.txt`](audit_results/religion_dpo_v2/audit_v2/results.txt) — every probe with the model response, the auditor reasoning, and pass/fail
- **Audit failure exemplars:** [`audit_results/religion_dpo_v2/audit_v2/dpo_corrections.jsonl`](audit_results/religion_dpo_v2/audit_v2/dpo_corrections.jsonl) — the failed probes formatted as DPO corrections for a subsequent refinement pass
- **Bake script (`tail_v2` lineage):** [`trainers/bake_phase_combined_v1_tail_v2.py`](trainers/bake_phase_combined_v1_tail_v2.py); the `config_a_v2` bake script is internal

### A.3 `religion_dpo_v1` (Config A — the full-surface regression)

- **Launcher:** [`recipes/launch_religion_dpo_v1.sh`](recipes/launch_religion_dpo_v1.sh)
- **Freeze config:** A (full-surface attention LoRA all 40 layers + shared_expert LoRA all 40 layers)
- **Audit verdict:** 133/163 = 81.6% (−1.2 pp from baseline). The `religion_honest` target lifted +12 pp; `infra_cross_system` regressed 4/4 → 1/4. This is the deception-shaped failure that motivated the A2 keystone restriction.
- **Audit proof:** [`audit_results/religion_dpo_v1/audit_v2/SUMMARY.md`](audit_results/religion_dpo_v1/audit_v2/SUMMARY.md)

### A.4 `religion_dpo_v3` (Config A4 — o_proj-only alternative)

- **Launcher:** [`recipes/launch_religion_dpo_v3.sh`](recipes/launch_religion_dpo_v3.sh)
- **Freeze config:** A4 (o_proj-only attention LoRA, alternative to A2's keystone-layer-only)
- **Audit proof:** [`audit_results/religion_dpo_v3/audit_v2/SUMMARY.md`](audit_results/religion_dpo_v3/audit_v2/SUMMARY.md)

### A.5 `length_mechanics_v1` (content-neutral isolation diagnostic)

- **Launcher:** [`recipes/launch_length_mechanics_v1.sh`](recipes/launch_length_mechanics_v1.sh)
- **Purpose:** training on the same 50 preference pairs but with content stripped (length-only signal) to test whether the `infra_cross_system` regression was content-driven or attention-mechanism-driven
- **Audit verdict:** 133/163 = 81.6%, same `infra_cross_system` regression as Config A → confirmed the regression is content-agnostic q/k attention drift, not content
- **Audit proof:** [`audit_results/length_mechanics_v1/audit_v2/SUMMARY.md`](audit_results/length_mechanics_v1/audit_v2/SUMMARY.md)

### A.6 `combined_big_v1` (scale-up SFT — honest negative result)

- **Launcher:** [`recipes/launch_combined_big_v1.sh`](recipes/launch_combined_big_v1.sh)
- **Output dirs:** ckpt-400 (audit: [`audit_results/combined_big_v1_ckpt400/audit_v2/SUMMARY.md`](audit_results/combined_big_v1_ckpt400/audit_v2/SUMMARY.md)) and ckpt-800 (audit: [`audit_results/combined_big_v1_ckpt800/audit_v2/SUMMARY.md`](audit_results/combined_big_v1_ckpt800/audit_v2/SUMMARY.md))
- **Result:** Neither beat the smaller `phase_combined_v1` SFT baseline despite a 20k-item corpus vs the smaller baseline. Cause not isolated. Listed in [`README.md`](README.md) §5 as an honest open question.

### A.7 `phase_combined_v1_tail` and `tail_v2` (regressed)

- **Launchers:** [`recipes/launch_phase_combined_v1_tail.sh`](recipes/launch_phase_combined_v1_tail.sh), [`recipes/launch_phase_combined_v1_tail_v2.sh`](recipes/launch_phase_combined_v1_tail_v2.sh)
- **Audit proofs:** [`audit_results/phase_combined_v1_tail/audit_v2/SUMMARY.md`](audit_results/phase_combined_v1_tail/audit_v2/SUMMARY.md), [`audit_results/phase_combined_v1_tail_v2/audit_v2/SUMMARY.md`](audit_results/phase_combined_v1_tail_v2/audit_v2/SUMMARY.md)
- **State:** Both regressed net vs `phase_combined_v1`. Preserved as forensic record. Not recommended for downstream use.

### A.8 `standard_dpo_vanilla` (vanilla DPO control)

- **Launcher:** [`recipes/launch_standard_dpo_vanilla.sh`](recipes/launch_standard_dpo_vanilla.sh)
- **Purpose:** Vanilla DPO control (no Config A constraints). Comparison point that motivated the Config A → A2 sequence.

---

## B. 9B Dense production line

### B.1 Phase 1 SFT (`sft_tools_qwen35_9b_fsdp`)

- **Launcher:** [`recipes/launch_sft_tools_qwen35_9b_fsdp.sh`](recipes/launch_sft_tools_qwen35_9b_fsdp.sh)
- **Base model:** `Qwen3.5-9B-Base`
- **Total steps:** 4367 (verifiable in the launcher line `TOTAL_STEPS=4367`)
- **Smoke battery:** 6/7 PASS on transformers-mode probes (T6 over-tooling is a bounded SFT-bias artifact, same failure mode at Phase 3)

### B.2 Phase 2 Constitutional CPT (`cpt_v3_v4_dense_9b/checkpoint-2400-multimodal`)

- **Launcher:** [`recipes/launch_cpt_phase2_qwen35_9b_fsdp.sh`](recipes/launch_cpt_phase2_qwen35_9b_fsdp.sh)
- **Trainer:** [`trainers/train_cpt_qwen35_dense.py`](trainers/train_cpt_qwen35_dense.py)
- **Phase-2 expert config:** [`configs/phase2_expert_config.json`](configs/phase2_expert_config.json) (44 KB)
- **Resume from:** B.1
- **Audit proof:** [`audit_results/cpt_qwen35_9b_v1_epoch1/audit_v2/SUMMARY.md`](audit_results/cpt_qwen35_9b_v1_epoch1/audit_v2/SUMMARY.md)

### B.3 Phase 3 Recovery SFT (`phase3_sft_single_spark*`)

- **Launcher:** [`recipes/launch_phase3_sft_single_spark.sh`](recipes/launch_phase3_sft_single_spark.sh)
- **Trainer:** [`trainers/train_recovery_sft_qwen35_dense.py`](trainers/train_recovery_sft_qwen35_dense.py) (contains the `chunk_conversation` function — the wedge-fix preprocessing inside the trainer)
- **Preprocessing tool:** [`trainers/chunk_corpus_offline.py`](trainers/chunk_corpus_offline.py) (the standalone offline chunker that produces a chunked .jsonl the 4-Spark trainer can consume without code changes)
- **Resume from:** B.2
- **Result:** Two independent runs (Spark 1 + Spark 3) converged to identical train_loss after the chunking fix resolved the 4-Spark FSDP wedge described in [`README.md`](README.md) §3.1. The single-Spark execution is the cross-validated proof; the 4-Spark execution on chunked corpus remains future work.
- **Audit proof:** [`audit_results/dpo_recovery_p2v3/audit_v2/SUMMARY.md`](audit_results/dpo_recovery_p2v3/audit_v2/SUMMARY.md)

---

## C. The fabric / cluster

### C.1 NCCL dual-rail RoCEv2 setup

Every recipe in [`recipes/`](recipes/) exports the same NCCL env block:

```bash
NCCL_IB_HCA=rocep1s0f0:1,roceP2p1s0f0:1   # capital P on rail 2
NCCL_IB_TC=104
NCCL_IB_TIMEOUT=23
NCCL_NET_GDR_LEVEL=0
NCCL_IB_RETRY_CNT=7
NCCL_TIMEOUT=1800
NCCL_SOCKET_IFNAME=enp1s0f0np0
GLOO_SOCKET_IFNAME=enp1s0f0np0
```

### C.2 4-Spark NCCL synth probe — the fabric-health verification

- **Synth probe results (busbw + per-rank exit + iteration count):** [`proof_of_run/nccl_synth_probe_results.md`](proof_of_run/nccl_synth_probe_results.md)
- **Result:** 10.23 GB/s (50 iters) sustaining to 12.57 GB/s (160-collective run) on `reduce_scatter` 218M-numel fp32; clean exit on all 4 ranks; no `IBV_WC_RETRY_EXC_ERR`
- **Stack:** ConnectX-7 firmware 28.45.4028 + NCCL 2.28.9

The 218M-numel size matches the failing collective size in the Phase 3 4-Spark wedge (see [`README.md`](README.md) §3.1) — confirming the fabric is healthy at that size and the wedge cause was corpus-pressure, not fabric.

---

## D. Diagnostic and exploratory recipes (preserved for forensic record)

| Path | Purpose | Notes |
|---|---|---|
| [`recipes/launch_diagnostic_2x2.sh`](recipes/launch_diagnostic_2x2.sh) | 2x2 crossed-control matrix for Phase 3 wedge isolation ("Cell B" reproduced the wedge in 10 minutes) | The diagnostic that localized Phase 3 wedge to corpus, not checkpoint or fabric |
| [`recipes/launch_phase_combined_v1_tail.sh`](recipes/launch_phase_combined_v1_tail.sh), [`recipes/launch_phase_combined_v1_tail_v2.sh`](recipes/launch_phase_combined_v1_tail_v2.sh) | Continuation-on-top-of-SFT attempts | Both regressed; preserved as forensic record |
| [`recipes/launch_standard_dpo_vanilla.sh`](recipes/launch_standard_dpo_vanilla.sh) | Vanilla DPO control | The comparison point that motivated Config A → A2 |

---

## E. What is *not* claimed (cross-link to [`README.md`](README.md) §7 + [`METRICS_PROVENANCE.md`](METRICS_PROVENANCE.md))

Every retracted / scrubbed / corrected claim is enumerated in [`README.md`](README.md) §7 with the reason. Those numbers do not have rows in [`METRICS_PROVENANCE.md`](METRICS_PROVENANCE.md) — that's the audit-trail discipline.

If you encounter a number in this artifact that does not have a `METRICS_PROVENANCE.md` row or a §7 retraction entry, please open an issue. The intent is full coverage.
