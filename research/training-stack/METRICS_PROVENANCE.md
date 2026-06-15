# METRICS PROVENANCE — `research/training-stack/`

Every load-bearing number cited in `README.md`, `TECHNICAL_APPENDIX.md`, or `REPRODUCE.md` is in the table below with a path to the file in this repository that contains the actual measured output. If a number you find in the documentation does not appear here, please open an issue — the discipline is that nothing leaves a draft without a row in this table.

> **Methodology caveat for a hiring-manager reader (cross-link to [`README.md`](README.md) §1.1).** All `Observed` rows below are measured against the fixed 163-probe constitutional audit harness in the sibling [`research/audit-harness-moe/`](../audit-harness-moe/) subdirectory, scored by an LLM-as-judge with paired-capability controls. The candidate and the baseline are scored against the *same* probe set with the *same* auditor; deltas are candidate-minus-baseline pass-rate. This is *not* held-out generalization measurement on independent test data. The construction of an independently-authored held-out test set is listed as future work in `README.md` §5.

## Audit-harness verdicts (constitutional 163-probe battery)

| Metric | Value | Proof file (this repo) | Register |
|---|---|---|---|
| Config A2 / religion_dpo_v2 audit pass rate | 138/163 = 84.7% | [`audit_results/religion_dpo_v2/audit_v2/SUMMARY.md`](audit_results/religion_dpo_v2/audit_v2/SUMMARY.md) | Observed |
| Config A2 vs phase_combined_v1 SFT baseline | +1.9 pp | [`audit_results/religion_dpo_v2/audit_v2/SUMMARY.md`](audit_results/religion_dpo_v2/audit_v2/SUMMARY.md) (line "DPO v2 (Config A2…) 138/163 = 84.7% +1.9pp") | Observed |
| Config A2 — all 8 infra-control categories held | 4/4 infra_cross_system, others held | [`audit_results/religion_dpo_v2/audit_v2/SUMMARY.md`](audit_results/religion_dpo_v2/audit_v2/SUMMARY.md) (per-category table) + [`audit_results/religion_dpo_v2/audit_v2/results.txt`](audit_results/religion_dpo_v2/audit_v2/results.txt) (per-probe model responses) | Observed |
| Config A2 — `religion_honest` lift | 7/17 → 8/17 (+6 pp over baseline) | [`audit_results/religion_dpo_v2/audit_v2/SUMMARY.md`](audit_results/religion_dpo_v2/audit_v2/SUMMARY.md) | Observed |
| Config A2 — `human_facilitator_anonymity` lift | 1/3 → 3/3 (+67 pp) | [`audit_results/religion_dpo_v2/audit_v2/SUMMARY.md`](audit_results/religion_dpo_v2/audit_v2/SUMMARY.md) | Observed |
| Config A2 — `sycophancy_resist` lift | 0/2 → 2/2 (+100 pp) | [`audit_results/religion_dpo_v2/audit_v2/SUMMARY.md`](audit_results/religion_dpo_v2/audit_v2/SUMMARY.md) | Observed |
| Config A (DPO v1) — full-surface regression | 133/163 = 81.6% (−1.2 pp); infra_cross_system 4/4 → 1/4 | [`audit_results/religion_dpo_v1/audit_v2/SUMMARY.md`](audit_results/religion_dpo_v1/audit_v2/SUMMARY.md) | Observed |
| length_mechanics_v1 — content-neutral isolation | 133/163 = 81.6%; same infra_cross_system regression as Config A → content-agnostic q/k attention drift identified | [`audit_results/length_mechanics_v1/audit_v2/SUMMARY.md`](audit_results/length_mechanics_v1/audit_v2/SUMMARY.md) | Observed |
| phase_combined_v1 SFT baseline | 135/163 = 82.8% | [`audit_results/phase_combined_v1/audit_v2_full/`](audit_results/phase_combined_v1/audit_v2_full/) | Observed |
| combined_big_v1 ckpt-400 audit | per-category in proof file | [`audit_results/combined_big_v1_ckpt400/audit_v2/SUMMARY.md`](audit_results/combined_big_v1_ckpt400/audit_v2/SUMMARY.md) | Observed |
| combined_big_v1 ckpt-800 audit | per-category in proof file | [`audit_results/combined_big_v1_ckpt800/audit_v2/SUMMARY.md`](audit_results/combined_big_v1_ckpt800/audit_v2/SUMMARY.md) | Observed |
| Phase 3 Recovery SFT (dpo_recovery_p2v3) audit | per-category in proof file | [`audit_results/dpo_recovery_p2v3/audit_v2/SUMMARY.md`](audit_results/dpo_recovery_p2v3/audit_v2/SUMMARY.md) | Observed |
| Phase 2 CPT (cpt_qwen35_9b_v1_epoch1) audit | per-category in proof file | [`audit_results/cpt_qwen35_9b_v1_epoch1/audit_v2/SUMMARY.md`](audit_results/cpt_qwen35_9b_v1_epoch1/audit_v2/SUMMARY.md) | Observed |
| Religion DPO v3 (Config A4 o_proj-only) audit | per-category in proof file | [`audit_results/religion_dpo_v3/audit_v2/SUMMARY.md`](audit_results/religion_dpo_v3/audit_v2/SUMMARY.md) | Observed |
| phase_combined_v1 tail / tail_v2 — regressed | per-category in proof files (preserved as forensic record) | [`audit_results/phase_combined_v1_tail/audit_v2/SUMMARY.md`](audit_results/phase_combined_v1_tail/audit_v2/SUMMARY.md), [`audit_results/phase_combined_v1_tail_v2/audit_v2/SUMMARY.md`](audit_results/phase_combined_v1_tail_v2/audit_v2/SUMMARY.md) | Observed |
| 163-probe audit harness | the actual probes + auditor pipeline + scoring | [`../audit-harness-moe/`](../audit-harness-moe/) (sibling subdirectory: `TAEY_AUDIT_V2.json`, `audit_pipeline.py`, `soma_proxy.py`) | Observed |

## Fabric / cluster verification

| Metric | Value | Proof file | Register |
|---|---|---|---|
| NCCL synth probe — reduce_scatter 218M-numel fp32, 50 iters | 10.23 GB/s steady, all 4 ranks exit=0, no `IBV_WC_RETRY_EXC_ERR` | [`proof_of_run/nccl_synth_probe_results.md`](proof_of_run/nccl_synth_probe_results.md) | Observed |
| NCCL synth probe — 160-collective sustained-pressure run | 12.57 GB/s steady, all 4 ranks exit=0 | [`proof_of_run/nccl_synth_probe_results.md`](proof_of_run/nccl_synth_probe_results.md) | Observed |
| ConnectX-7 firmware version | 28.45.4028 | [`proof_of_run/nccl_synth_probe_results.md`](proof_of_run/nccl_synth_probe_results.md) | Observed |
| NCCL stack version | 2.28.9 | [`proof_of_run/nccl_synth_probe_results.md`](proof_of_run/nccl_synth_probe_results.md) | Observed |

## Recipe hyperparameters (verifiable from the launchers themselves)

| Metric | Value | Proof file (the launcher itself) | Register |
|---|---|---|---|
| Config A2 freeze mask | keystone layers `[8, 9, 11, 15, 21, 23]`, shared_expert LoRA all 40 layers | [`recipes/launch_religion_dpo_v2.sh`](recipes/launch_religion_dpo_v2.sh) line `FREEZE_CONFIG=A2` + `KEYSTONE_LAYERS=…` | Observed (in launcher) |
| Config A2 DPO hyperparams | BETA=0.05, LR_ESFT=1e-7, LR_LORA=3e-7, LR_ROUTER=0, WARMUP_STEPS=5, TOTAL_STEPS=642, SAVE_EVERY=60 | [`recipes/launch_religion_dpo_v2.sh`](recipes/launch_religion_dpo_v2.sh) | Observed (in launcher) |
| Config A2 frozen-experts mask | 159 frozen experts | [`configs/frozen_experts_v4_1_polysemantic.json`](configs/frozen_experts_v4_1_polysemantic.json) (count the entries) | Observed |
| Phase 1 SFT — total steps | 4367 | [`recipes/launch_sft_tools_qwen35_9b_fsdp.sh`](recipes/launch_sft_tools_qwen35_9b_fsdp.sh) line `TOTAL_STEPS=4367` | Observed (in launcher) |
| NCCL HCA env-var (dual-rail, capital P on rail 2) | `NCCL_IB_HCA=rocep1s0f0:1,roceP2p1s0f0:1` | every launcher in [`recipes/`](recipes/) | Observed |

## Phase 3 wedge → corpus fix

| Metric | Value | Proof file | Register |
|---|---|---|---|
| Phase 3 4-Spark FSDP wedge — NumelIn at failure | 218M (matches synth probe collective size) | engineering narrative in [`README.md`](README.md) §3.1; NCCL synth probe `reduce_scatter` 218M-numel verifies the fabric is healthy at that size: [`proof_of_run/nccl_synth_probe_results.md`](proof_of_run/nccl_synth_probe_results.md) | Observed |
| Phase 3 corpus that wedged | 7,077 multi-turn items, length variance 200–31,700 tokens | the chunker source is [`trainers/chunk_corpus_offline.py`](trainers/chunk_corpus_offline.py) | Observed (in source) |
| Chunked-corpus single-Spark proof | cross-validated identical train_loss on two independent runs (Spark 1 + Spark 3) | [`audit_results/dpo_recovery_p2v3/audit_v2/SUMMARY.md`](audit_results/dpo_recovery_p2v3/audit_v2/SUMMARY.md) | Observed |
| Conversation chunker — splits at user-assistant pair boundaries with budget 0.92 × MAX_SEQ | the algorithm itself | [`trainers/chunk_corpus_offline.py`](trainers/chunk_corpus_offline.py) (the `chunk_conversation` function is also in [`trainers/train_recovery_sft_qwen35_dense.py`](trainers/train_recovery_sft_qwen35_dense.py)) | Observed (in source) |

## What is *not* claimed (cross-link to README §7)

The list of explicitly-retracted-or-not-claimed numbers (R@10 = 0.81 / 0.667→0.944, soft recall 0.846, NCCL busbw > 12.57 GB/s, 70× BYOV, 50× ES, etc.) is in [`README.md`](README.md) §7. Those numbers do *not* have rows here — they are not claimable.

---

If you are reading this artifact for hiring evaluation and want to spot-check: the highest-signal places to look are (1) [`audit_results/religion_dpo_v2/audit_v2/SUMMARY.md`](audit_results/religion_dpo_v2/audit_v2/SUMMARY.md) (the 84.7% / +1.9pp headline result with the full per-category breakdown), (2) [`audit_results/religion_dpo_v2/audit_v2/results.txt`](audit_results/religion_dpo_v2/audit_v2/results.txt) (the per-probe model responses, audit reasoning, and pass/fail decisions), (3) [`recipes/launch_religion_dpo_v2.sh`](recipes/launch_religion_dpo_v2.sh) (the actual launcher that produced it, with every hyperparameter), and (4) [`proof_of_run/nccl_synth_probe_results.md`](proof_of_run/nccl_synth_probe_results.md) (the 4-Spark fabric verification). Those four files together back the strongest claims in the artifact.
