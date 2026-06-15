# Reproducing the Production Line

Step-by-step to re-run the PALIOS-TAEY training pipeline on equivalent hardware. The launcher scripts in [`recipes/`](recipes/) are the actual scripts that ran on the production deployment; substitute the operator paths (`/home/<user>/...`) for your environment.

> **Hardware assumed:** 4 × DGX Spark GB10 (Blackwell sm_121) + an inference / bake host with disk for 67-GB-class baked checkpoints. ConnectX-7 dual-rail RoCEv2 internal cluster network.

---

## 0. Prerequisites

- Python 3.10 / PyTorch with CUDA 13.0 support for `sm_121`
- NCCL 2.28.9, ConnectX-7 firmware 28.45.4028
- `transformers` + `peft` + `accelerate` + `datasets`
- Base models from Hugging Face: `Qwen3.5-9B-Base`, `Huihui-Qwen3.5-35B-A3B-abliterated`
- The audit harness from the sibling [`research/audit-harness-moe/`](../audit-harness-moe/) subdirectory if you intend to run the constitutional 163-probe audit after bake

The recipes (`launch_*.sh` in [`recipes/`](recipes/)) are documented for the deployment that ran them. They reference paths like `/home/spark/training_outputs/...`; substitute for your hosts. Env-variable overrides are honored where present (e.g., `OUTPUT_DIR`, `RESUME_DELTA`, `DPO_DATA`, `MODEL_PATH`, `FROZEN_EXPERTS`, `KEYSTONE_LAYERS`).

---

## 1. Network setup — NCCL dual-rail RoCEv2

Across all 4 nodes:

```bash
export NCCL_IB_HCA=rocep1s0f0:1,roceP2p1s0f0:1   # capital P on rail 2 — easy to miss
export NCCL_IB_TC=104
export NCCL_IB_TIMEOUT=23
export NCCL_NET_GDR_LEVEL=0
export NCCL_IB_RETRY_CNT=7
export NCCL_TIMEOUT=1800
export NCCL_SOCKET_IFNAME=enp1s0f0np0
export GLOO_SOCKET_IFNAME=enp1s0f0np0
```

These are exported verbatim by every launcher in [`recipes/`](recipes/); the env block above is reproduced here as the minimal contract for fabric setup.

**Verify before running training.** The standalone synth probe at the failing 218M-numel `reduce_scatter` size is the cheapest fabric-health test. The Python script source is not in this subdirectory (it lives in our internal `embedding-server` repo); the **results** of running it are in [`proof_of_run/nccl_synth_probe_results.md`](proof_of_run/nccl_synth_probe_results.md), which documents the exact invocation, ranks, and expected throughput so you can write the equivalent test against your own fabric.

Expected on a healthy 4-Spark ConnectX-7 RoCE fabric: 10.23 GB/s steady (50 iters) sustaining to 12.57 GB/s under a 160-collective stress run; no `IBV_WC_RETRY_EXC_ERR`. If the probe fails on your fabric, do not start full training — the wedge will look like a training bug but is fabric.

---

## 2. 35B-A3B MoE production line

### 2.1 SFT baseline → phase_combined_v1

> **Note on naming.** There is no `launch_phase_combined_v1.sh` shipped — the `phase_combined_v1` checkpoint is produced by [`recipes/launch_production_sft.sh`](recipes/launch_production_sft.sh) with `OUTPUT_DIR` set as shown below. Downstream launchers (`launch_phase_combined_v1_tail*`, `launch_religion_dpo_v*`) `RESUME` from `phase_combined_v1/final` step 582 against this output path.

```bash
# 4-Spark FSDP, fresh from abliterated base
export MODEL_PATH=/home/<user>/models/Huihui-Qwen3.5-35B-A3B-abliterated
export OUTPUT_DIR=/home/<user>/training_outputs/phase_combined_v1
export TOTAL_STEPS=982   # adjust per corpus
bash recipes/launch_production_sft.sh
```

Audit verdict expected: ~82.8% (135/163) on the 163-probe constitutional battery. The actual `phase_combined_v1` audit result is in [`audit_results/phase_combined_v1/audit_v2_full/`](audit_results/phase_combined_v1/audit_v2_full/) for comparison.

### 2.2 Config A2 keystone-attention LoRA DPO → religion_dpo_v2 (the +1.9pp headline)

```bash
# Resume from phase_combined_v1/final step 582; 4-Spark FSDP.
# MODEL_PATH = architecture base for model init;
# RESUME_DELTA carries the trained-weights checkpoint
# (the DPO trainer loads architecture from MODEL_PATH then resumes weights from RESUME_DELTA).
export MODEL_PATH=/home/<user>/models/Huihui-Qwen3.5-35B-A3B-abliterated
export RESUME_DELTA=/home/<user>/training_outputs/phase_combined_v1/final
export DPO_DATA=/home/<user>/training_data/religion_run_v1/religion_v3_dpo_pairs_with_ref.jsonl
export FROZEN_EXPERTS=$(pwd)/configs/frozen_experts_v4_1_polysemantic.json
export OUTPUT_DIR=/home/<user>/training_outputs/religion_dpo_v2
export FREEZE_CONFIG=A2
export KEYSTONE_LAYERS='[8, 9, 11, 15, 21, 23]'
export BETA=0.05
export LR_ESFT=1e-7
export LR_LORA=3e-7
export LR_ROUTER=0
export WARMUP_STEPS=5
export TOTAL_STEPS=642
export SESSION_LIMIT=900
export SAVE_EVERY=60
export DPO_ABORT_RATIO_MAX=10.0
export DPO_ABORT_EXPERT_DRIFT=0.05
bash recipes/launch_religion_dpo_v2.sh
```

Audit verdict expected: **84.7% (138/163)**, **+1.9pp** over phase_combined_v1. Should hold all 8 infra-control categories (length_mechanics_v1 confirmed the prior Config A regression was content-agnostic q/k attention; A2's keystone-only freeze fixes it). The actual `religion_dpo_v2` audit result is in [`audit_results/religion_dpo_v2/audit_v2/`](audit_results/religion_dpo_v2/audit_v2/).

---

## 3. 9B Dense production line

### 3.1 Phase 1 SFT — tool-use

```bash
export MODEL_PATH=/home/<user>/models/Qwen3.5-9B-Base
export OUTPUT_DIR=/home/<user>/training_outputs/sft_tools_qwen35_9b_fsdp
export TOTAL_STEPS=4367
bash recipes/launch_sft_tools_qwen35_9b_fsdp.sh
```

### 3.2 Phase 2 Constitutional CPT

```bash
export RESUME_DELTA=/home/<user>/training_outputs/sft_tools_qwen35_9b_fsdp/final
export OUTPUT_DIR=/home/<user>/training_outputs/cpt_v3_v4_dense_9b
bash recipes/launch_cpt_phase2_qwen35_9b_fsdp.sh
```

The CPT trainer is [`trainers/train_cpt_qwen35_dense.py`](trainers/train_cpt_qwen35_dense.py). The Phase-2 expert config is [`configs/phase2_expert_config.json`](configs/phase2_expert_config.json) (44 KB).

### 3.3 Phase 3 Recovery SFT — wedge-fix path

**Step 3.3a — pre-chunk the multi-turn corpus offline:**

```bash
python3 trainers/chunk_corpus_offline.py \
  --in phase3_sft.jsonl \
  --out phase3_sft_chunked.jsonl \
  --max-seq 4096 \
  --budget-fraction 0.92
```

The chunker source is [`trainers/chunk_corpus_offline.py`](trainers/chunk_corpus_offline.py). The same `chunk_conversation` function is reused inside the trainer (see [`trainers/train_recovery_sft_qwen35_dense.py`](trainers/train_recovery_sft_qwen35_dense.py)).

**Step 3.3b — run single-Spark Recovery SFT on the chunked corpus:**

```bash
export RESUME_DELTA=/home/<user>/training_outputs/cpt_v3_v4_dense_9b/checkpoint-2400-multimodal
export SFT_JSONL=/path/to/phase3_sft_chunked.jsonl
bash recipes/launch_phase3_sft_single_spark.sh
```

Single-Spark Recovery SFT cross-validates to identical train_loss across Spark 1 and Spark 3 (audit verdict in [`audit_results/dpo_recovery_p2v3/audit_v2/`](audit_results/dpo_recovery_p2v3/audit_v2/)).

### 3.4 4-Spark Phase 3 on chunked corpus (future work)

The single-Spark Recovery SFT validates that the chunking fix resolves the corpus-pressure → RDMA-queue-saturation root cause. The 4-Spark execution of the same chunked corpus is not yet shipped; that is the bookend re-run that confirms the wedge-fix on the production cluster. Listed in `README.md` §5 honest-open-questions.

---

## 4. Bake-and-test (production deployment to inference host)

The bake script for the `tail_v2` lineage is [`trainers/bake_phase_combined_v1_tail_v2.py`](trainers/bake_phase_combined_v1_tail_v2.py). Other bake scripts (`bake_orpo.py`, `bake_config_a_v2.py`) live on Thor in our deployment and are not in this subdirectory.

After bake, run the constitutional 163-probe audit harness from the sibling [`research/audit-harness-moe/`](../audit-harness-moe/) subdirectory. Results land in `audit_v2/` shaped exactly like the verdicts under [`audit_results/`](audit_results/) (per-checkpoint `SUMMARY.md`, `summary.json`, `results.txt`, `dpo_corrections.jsonl`, `audit.log`).

---

## 5. Things to verify before claiming you reproduced this

- Synth probe passes at ≥ 10 GB/s on your fabric.
- Phase 1 SFT smoke battery 6/7 PASS (T6 over-tooling is the expected bounded artifact).
- Phase 2 CPT canonical bytes match (or your equivalent bake bytes — record them).
- Phase 3 Recovery SFT cross-validates to identical (or very close) train_loss across two independent host pairs.
- Pre-chunk validator coverage ≥ 99.9% on your multi-turn corpus.
- religion_dpo_v2 audit lands at +1–2 pp over phase_combined_v1 baseline.

If your numbers diverge materially: please open an issue with your hardware + commit SHA + recipe parameters so we can compare. The goal is reproducible production discipline, not a single locked-in result.
