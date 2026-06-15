# Reproducing the Production Line

Step-by-step to re-run the PALIOS-TAEY training pipeline on equivalent hardware. Recipes are documented as-deployed; substitute paths for your environment.

> **Hardware assumed:** 4 × DGX Spark GB10 (Blackwell sm_121) + 1 × Mira RTX 4090 24 GB. Storage local NVMe + a 22 TB external drive for historical archives.

---

## 0. Prerequisites

- Python 3.10 / PyTorch with CUDA 13.0 support for sm_121
- NCCL 2.28.9, ConnectX-7 firmware 28.45.4028
- Weaviate 1.x (port 8088), Neo4j 5.x (port 7689 no-auth or 7687 with `***REMOVED***`), Redis 7.x (port 6379)
- `transformers` + `peft` + `accelerate` + `datasets` (versions as locked in `requirements.txt`)
- Base models from Hugging Face: `Qwen3.5-9B-Base`, `Huihui-Qwen3.5-35B-A3B-abliterated`

The recipe scripts (`launch_*.sh`) are documented for the deployment they were run on. They reference paths like `/home/spark/training_outputs/...` and `/home/thor/models/...`; substitute for your hosts. Env-variable overrides are honored where present (e.g. `OUTPUT_DIR`, `RESUME_DELTA`, `DPO_DATA`, `MODEL_PATH`).

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

**Verify before running training**: the standalone synth probe is the cheapest fabric-health test we have. It reproduces the exact failing collective shape (218M-numel `reduce_scatter` fp32) and either passes cleanly at ~12 GB/s or surfaces the issue immediately.

```bash
# on Spark 1 (rank 0); the probe is a Python script, launch with torchrun across nodes
python3 isma/scripts/spark_deploy/nccl_synth_probe.py --rank 0 --nranks 4 --numel 218000000
# corresponding launches on Sparks 2/3/4 with appropriate ranks
```

Expected: clean run at 10–12.57 GB/s; no `IBV_WC_RETRY_EXC_ERR`. If the probe fails, do not start full training — the wedge will look like a training bug but is fabric.

---

## 2. 35B-A3B MoE production line

### 2.1 SFT baseline → phase_combined_v1

```bash
# 4-Spark FSDP, fresh from abliterated base
# (No launcher named launch_phase_combined_v1.sh; produced by launch_production_sft.sh
#  with OUTPUT_DIR override. Downstream launchers RESUME from phase_combined_v1/final step 582.)
export MODEL_PATH=/home/spark/models/Huihui-Qwen3.5-35B-A3B-abliterated
export OUTPUT_DIR=/home/spark/training_outputs/phase_combined_v1
export TOTAL_STEPS=982   # adjust per corpus
bash isma/scripts/spark_deploy/launch_production_sft.sh
```

Bake-and-test:

```bash
bash bake_and_test.sh phase_combined_v1/step582 taey-phase-combined-v1 phase_combined_v1
```

Audit verdict expected: ~82.8% on the 163-probe constitutional battery.

### 2.2 Config A2 keystone-attention LoRA DPO → religion_dpo_v2

```bash
# Resume from phase_combined_v1/final step 582; same data + identical hyperparams as v1
# (only freeze config differs); 4-Spark FSDP.
export MODEL_PATH=/home/spark/models/Huihui-Qwen3.5-35B-A3B-abliterated
export RESUME_DELTA=/home/spark/training_outputs/phase_combined_v1/final
export DPO_DATA=/home/spark/training_data/religion_run_v1/religion_v3_dpo_pairs_with_ref.jsonl
export FROZEN_EXPERTS=/home/spark/training_data/phase1_constitutional/frozen_experts_v4_1_polysemantic.json
export OUTPUT_DIR=/home/spark/training_outputs/religion_dpo_v2
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
bash isma/scripts/spark_deploy/launch_religion_dpo_v2.sh
```

Bake-and-test:

```bash
bash bake_and_test.sh religion_dpo_v2/step642 taey-religion-dpo-v2 religion_dpo_v2
```

Audit verdict expected: ~84.7% (+1.9pp over phase_combined_v1). Should hold all 8 infra-control categories (length_mechanics_v1 confirmed the prior Config A regression was content-agnostic q/k attention; A2's keystone-only freeze fixes it).

---

## 3. 9B Dense production line

### 3.1 Phase 1 SFT — tool-use

```bash
export MODEL_PATH=/home/spark/models/Qwen3.5-9B-Base
export OUTPUT_DIR=/home/spark/training_outputs/sft_tools_qwen35_9b_fsdp
export TOTAL_STEPS=4367
bash isma/scripts/spark_deploy/launch_sft_tools_qwen35_9b_fsdp.sh
```

Smoke battery expected: 6/7 PASS. T6 over-tooling is a bounded SFT-bias artifact (same failure mode at Phase 3).

### 3.2 Phase 2 Constitutional CPT

```bash
export RESUME_DELTA=/home/spark/training_outputs/sft_tools_qwen35_9b_fsdp/final
export OUTPUT_DIR=/home/spark/training_outputs/cpt_v3_v4_dense_9b
bash isma/scripts/spark_deploy/launch_cpt_phase2_qwen35_9b_fsdp.sh
```

Multimodal-conversion of the final checkpoint:

```bash
python isma/scripts/convert_sft_checkpoint_to_multimodal.py \
  --in /home/spark/training_outputs/cpt_v3_v4_dense_9b/checkpoint-2400 \
  --out /home/spark/training_outputs/cpt_v3_v4_dense_9b/checkpoint-2400-multimodal
```

Canonical bytes: 17,907,662,976.

### 3.3 Phase 3 Recovery SFT — wedge-fix path

**Step 3.3a — pre-chunk the multi-turn corpus offline:**

```bash
python isma/scripts/chunk_corpus_offline.py \
  --in phase3_sft.jsonl \
  --out phase3_sft_chunked.jsonl \
  --max-seq 4096 \
  --budget-fraction 0.92
```

Expected: 7,077 multi-turn items → ~16,705 chunks at 99.96% coverage (U+FFFD precision caveat at ~5e-8 char rate).

**Step 3.3b — run single-Spark Recovery SFT on the chunked corpus:**

```bash
export RESUME_DELTA=/home/spark/training_outputs/cpt_v3_v4_dense_9b/checkpoint-2400-multimodal
export SFT_JSONL=/path/to/phase3_sft_chunked.jsonl
bash isma/scripts/spark_deploy/launch_phase3_sft_single_spark.sh
```

Expected: 1 epoch / 2,089 optimizer steps; train_loss ~1.122; wall-clock ~16 hours; 6/7 PASS on the smoke battery. Cross-validate on a second host for confidence.

### 3.4 4-Spark Phase 3 on chunked corpus (future work)

The single-Spark Recovery SFT validates that the chunking fix resolves the corpus-pressure → RDMA-queue-saturation root cause. The 4-Spark execution of the same chunked corpus is not yet shipped; that's the bookend re-run that confirms the wedge-fix on the production cluster. See open questions in `README.md` §5.

---

## 4. Bake-and-test (production deployment to inference host)

`bake_and_test.sh` pushes a baked checkpoint to an inference host (configured at the top of the script), brings up vLLM, and runs the 163-probe audit battery. Usage:

```bash
bash bake_and_test.sh <checkpoint_subdir> <output_model_name> <results_dir_name>
```

Internals (see the script for the exact bake invocation):
- pre-bake checks (checkpoint size > 1 GB; base model present; disk free > 70 GB)
- stop existing systemd vLLM service (prevent restart-fighting)
- rm existing baked dir; bake (uses appropriate bake_*.py for the recipe)
- start vLLM container against the new bake
- run the 163-probe audit harness; results land under `/home/mira/training/results/<results_dir>/`

---

## 5. ISMA retrieval — bring up the stack

### 5.1 Services

```bash
# embedding server (provides BYOV vectors over /embed)
uvicorn isma.src.server:app --host 0.0.0.0 --port 8089

# query API (provides /search, /search/hmm, /v2/search/*, /document/*, /v2/expand/*)
uvicorn isma.src.query_api:app --host 0.0.0.0 --port 8095 --workers 4

# reranker port 8085 is intentionally NOT in production (deprecated — harmed results in testing).
```

Weaviate / Neo4j / Redis are external dependencies; bring them up first with the connection params in `isma/config.py`.

### 5.2 Ingest a corpus

```bash
python isma/scripts/unified_ingest.py --corpus-dir /path/to/your/markdown/corpus
```

`unified_ingest.py` writes to all three substrates (Weaviate tiles at all scales, Neo4j relationship graph, Redis HMM motif index) atomically with rollback on partial failure.

### 5.3 Query

V1 `/search` is canonical for production (full corpus). V2 `/v2/search` is currently a partial migration and should not be used as the production endpoint until migration completes.

```bash
# canonical V1 deep query
curl -s -X POST http://localhost:8095/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"<topic>","top_k":25,"scale":"full_4096"}'

# graph-aware HMM hybrid
curl -s -X POST http://localhost:8095/search/hmm \
  -H 'Content-Type: application/json' \
  -d '{"query":"<topic>","top_k":25}'
```

Per the discipline: go deep (top_k ≥ 25, `scale: full_4096` for full passages), pull the whole `content` field not previews, run 3–6 query phrasings + union for broad topics, expand hits via `GET /document/<hash>/text` or `/v2/expand/<hash>`.

### 5.4 HMM enrichment (corrected verdict context)

Enrichment writes HMM motif annotations across all three substrates with rollback. The corrected A/B verdict (Taey-judged, n=222, enriched-only both arms, order-reversal controls): competitive but ~even on general search; bristle_arc 24-12 + exact 23-14 lean HMM. Enable HMM enrichment if your workload includes interpretive / identity-aligned queries; the data does not yet support claiming a general-domain win.

---

## 6. Audit harness (163-probe constitutional battery)

The audit battery costs ~$200 in Anthropic API calls + 2 GPU-hr + 4 h wall-clock per run. Run after bake-and-test for any checkpoint you're considering shipping. The harness output lands in `/home/mira/training/results/<run-name>/audit_v2/`:

- `audit.log` — full per-probe execution log
- `results.txt` — human-readable pass/fail per probe
- `summary.json` — machine-readable verdict
- `SUMMARY.md` — per-category summary
- `dpo_corrections.jsonl` — failure exemplars for DPO refinement

The harness is checked into the repo and is the same battery against which the 82.8% / 84.7% headline numbers were measured.

---

## 7. Things to verify before claiming you reproduced this

- Synth probe passes at ≥ 10 GB/s on your fabric.
- Phase 1 SFT smoke battery 6/7 PASS (T6 over-tooling is the expected bounded artifact).
- Phase 2 CPT bytes = 17,907,662,976 (or your equivalent canonical bake bytes — record them).
- Phase 3 Recovery SFT cross-validates to identical (or very close) train_loss across two independent host pairs.
- Pre-chunk validator coverage ≥ 99.9% on your multi-turn corpus.
- religion_dpo_v2 audit lands at +1–2pp over phase_combined_v1 baseline.

If your numbers diverge materially: please file an issue with your hardware + commit + recipe parameters so we can compare. The goal of this repo is reproducible production discipline, not a single locked-in result.
