#!/bin/bash
# 4-node FSDP launcher for Qwen3.5-9B-Base tools+chat SFT (Phase 1 per
# /home/mira/infra-soul/TOOLS.md). Run on EACH Spark; the script detects its
# own fabric IP and assigns rank from it.
#
# Adapted from launch_fsdp_bare_metal.sh (proven on 35B-A3B). The NCCL recipe,
# rank-by-IP detection, and accelerate-launch pattern are unchanged. Differences:
#   - MODEL_PATH: Qwen3.5-9B-Base (dense) instead of the 35B-A3B abliterated
#   - DATA_PATH: tools+chat SFT corpus (68K samples)
#   - Accelerate config: fsdp_dense_9b.yaml (Qwen3_5DecoderLayer wrap)
#   - Script: train_sft_tools_qwen35_dense.py (HF Trainer, full FT, bucket batch)
#
# Why this NCCL config (vs. the broken first attempt now archived):
#   - NCCL_IB_HCA names the RoCE HCAs explicitly across both NICs. Without it
#     NCCL hunts for a phantom IB device and hangs at first all_gather.
#   - NCCL_NET_GDR_LEVEL=0 (not 5). Perplexity recommended 5 but it doesn't
#     work on this fleet — the proven 35B runs use 0.
#   - TORCH_NCCL_DUMP_ON_TIMEOUT=1 so a future hang produces a flight-recorder
#     dump instead of a silent freeze.

set -eo pipefail

# ── Environment ───────────────────────────────────────────────────────────
export PATH="/home/spark/.local/bin:/usr/local/cuda-13.0/bin:$PATH"
export CUDA_HOME="/usr/local/cuda-13.0"
export LD_LIBRARY_PATH="/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH"
export PYTHONPATH="/home/spark/embedding-server:$PYTHONPATH"

# ── NCCL — Blackwell / DGX Spark proven recipe (verbatim from Phase 1 SFT,
# commit dd9e12e — that config ran 4367 steps clean over 9 hours). All today's
# additions (GID_INDEX, SOCKET_IFNAME, ALGO=Ring, MIN_NCHANNELS, capital-P fix,
# AVOID_RECORD_STREAMS, TRACE_BUFFER_SIZE) were unproven theory on top of working
# config. Reverted 2026-05-10 21:58 — restoring exact Phase 1 SFT env.
export NCCL_IB_HCA=rocep1s0f0:1,roceP2p1s0f0:1
export NCCL_IB_TC=104
export NCCL_IB_TIMEOUT=23
export NCCL_NET_GDR_LEVEL=0
export NCCL_IB_RETRY_CNT=7
export NCCL_TIMEOUT=1800
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
# GID INDEX PIN 2026-05-12 — see SFT launcher for rationale
export NCCL_IB_GID_INDEX=3
# QPS_PER_CONNECTION 2026-05-12 — see SFT launcher
export NCCL_IB_QPS_PER_CONNECTION=4

# ── FLA / Triton — GB10 sm_121 hardening ──────────────────────────────────
export FLA_USE_TMA=0
export TRITON_AUTOTUNE_DISABLE=1
export FLA_DISABLE_CAUSAL_CONV1D=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.8"
export TOKENIZERS_PARALLELISM=false

# ── Training paths ────────────────────────────────────────────────────────
# IMPORTANT: CPT must start from Phase 1 SFT (NOT base). Prior cycle bug:
# defaulted to base, threw away 9h of Phase 1 SFT compute (tools+chat).
# The trained-base invariant is documented in plans/canonical_dense_9b_recipe_v1.md.
export MODEL_PATH="${MODEL_PATH:-/home/spark/training_outputs/sft_tools_qwen35_9b_fsdp/final/converted}"
# train_fsdp_dense_9b.py: CPT mode is selected when CPT_DATA is set AND SFT_DIR is
# either empty OR not-a-directory. Orchestrator forwards env vars only when non-empty,
# AND the trainer defaults SFT_DIR to /var/spark/isma/training/sft (a real dir on the
# Sparks) when not set. So passing SFT_DIR="" gets dropped by the orchestrator and
# the trainer then thinks SFT mode is desired. Use an explicit sentinel (non-empty,
# clearly non-dir) so the orchestrator forwards it and the trainer routes to CPT mode.
export SFT_DIR="/nonexistent/cpt_mode_sentinel"
# CPT_DATA must be the rebuilt v3 corpus matching the canonical recipe. NO default
# to prevent a future bug from launching against a stale/wrong corpus. Caller MUST set.
export CPT_DATA="${CPT_DATA:?ERROR: CPT_DATA must be set to a v3 corpus jsonl path; do not default to a stale corpus}"
export GENERAL_DIR="${GENERAL_DIR:-}"
export OUTPUT_DIR="${OUTPUT_DIR:-/home/spark/training_outputs/cpt_v3_dense_9b}"
mkdir -p "$OUTPUT_DIR"

# Pre-flight: refuse to launch if MODEL_PATH points at base (catches the prior cycle bug)
if [[ "$MODEL_PATH" == */Qwen3.5-9B-Base* ]] || [[ "$MODEL_PATH" == */qwen3.5-9b-base* ]]; then
    echo "ERROR: MODEL_PATH appears to be the base model: $MODEL_PATH" >&2
    echo "       CPT must start from Phase 1 SFT artifact." >&2
    echo "       If this is intentional (re-run from base), set FORCE_BASE=1." >&2
    if [[ "${FORCE_BASE:-0}" != "1" ]]; then exit 1; fi
fi
# Pre-flight: refuse to launch on the known wedge corpus
if [[ "$CPT_DATA" == *cpt_merged_clean.jsonl ]]; then
    echo "ERROR: CPT_DATA points at cpt_merged_clean.jsonl (known wedge corpus)." >&2
    echo "       This corpus is 174M tokens, 95.87% discussion-tier, audited QUARANTINE 2026-05-07." >&2
    exit 1
fi

# ── Trainer knobs (defaults from 2026-05-08 Family consult: Gemini + Grok converge) ─────────
# MAX_SEQ=16384 — Phase 1 SFT proven, both consult responses converge on this value.
#                Per Apr 21 methodology + GitHub issues, packing is unsafe (Qwen3.5 GDN NaN at step 1).
#                Per Family consult dissent, full-pad-to-MAX_SEQ wedges the cluster (both prior 4-Spark
#                CPT attempts failed with this pattern). Trainer CPT branch must be patched to return
#                variable-length tokens; collate_fn does dynamic batch-max padding.
# BATCH_SIZE_PER_RANK=8 — Phase 1 SFT proven on this exact stack (Grok recommends 8; Gemini argues 4
#                for safety margin — going 8 since it's the proven value).
# LR=1e-5 / WARMUP_STEPS=100 — Phase 1 SFT default, Family consult convergence.
export MAX_SEQ="${MAX_SEQ:-4096}"
# BATCH=2 per 5/5 Family consult 2026-05-10 (Claude regime-separation argument).
# CPT corpus is uniformly near-MAX vs SFT's mostly-below-MAX, so per-step mean
# memory is ~3.75x higher than SFT at same BATCH; halving batch acknowledges
# regime difference.
# 5/5 Family consult round 3 convergent: BATCH=1 reduces per-step peak thermal/power
# envelope (Claude documented GB10 thermal pattern), and reduces _REDUCE_SCATTER_BASE
# pressure per step. GRAD_ACCUM=4 maintains effective batch of 16 across 4 ranks.
export BATCH_SIZE_PER_RANK="${BATCH_SIZE_PER_RANK:-1}"
export GRAD_ACCUM="${GRAD_ACCUM:-4}"
# TOTAL_STEPS depends on corpus size after re-chunk at chunk_tokens=15800. Caller MUST set explicitly
# based on the v3 manifest after gemini's rebuild.
export TOTAL_STEPS="${TOTAL_STEPS:?ERROR: TOTAL_STEPS must be set; depends on cpt_v3_v3 corpus row count}"
# Resume from a saved checkpoint when set (relative or absolute path)
export RESUME_DELTA="${RESUME_DELTA:-}"
export SAVE_EVERY="${SAVE_EVERY:-200}"
export SESSION_LIMIT="${SESSION_LIMIT:-200}"
export WARMUP_STEPS="${WARMUP_STEPS:-100}"
export LR="${LR:-1e-5}"

# ── Multi-node configuration ──────────────────────────────────────────────
MASTER_ADDR="${MASTER_ADDR:-192.168.100.10}"
MASTER_PORT="${MASTER_PORT:-29500}"
NUM_NODES="${NUM_NODES:-4}"
GPUS_PER_NODE=1

# Detect rank from local fabric IP. Mapping is fixed by the cluster wiring:
#   192.168.100.10 = Spark 1 = rank 0  (master)
#   192.168.100.11 = Spark 2 = rank 1
#   192.168.100.12 = Spark 3 = rank 2
#   192.168.100.13 = Spark 4 = rank 3
MY_IP=$(ip addr | grep -E "192.168.100." | awk '{print $2}' | cut -d/ -f1 | head -n 1)

case "$MY_IP" in
    "192.168.100.10") RANK=0 ;;
    "192.168.100.11") RANK=1 ;;
    "192.168.100.12") RANK=2 ;;
    "192.168.100.13") RANK=3 ;;
    *)
        echo "ERROR: Unknown fabric IP '$MY_IP' on $(hostname). Expected 192.168.100.{10,11,12,13}." >&2
        exit 1
        ;;
esac

echo "FSDP tools+chat SFT on $(hostname) (IP: $MY_IP, Rank: $RANK / $((NUM_NODES - 1)))"
echo "  MODEL:  $MODEL_PATH"
echo "  DATA:   $DATA_PATH"
echo "  OUTPUT: $OUTPUT_DIR"
echo "  MASTER: $MASTER_ADDR:$MASTER_PORT"
echo ""

# train_fsdp_dense_9b.py reads ALL config from environment variables (no
# argparse) — same pattern as train_fsdp_v3.py. The env vars set above are
# what it consumes: MODEL_PATH, SFT_DIR, CPT_DATA, GENERAL_DIR, OUTPUT_DIR,
# MAX_SEQ, TOTAL_STEPS, SAVE_EVERY, SESSION_LIMIT, WARMUP_STEPS, LR_LORA,
# LR_ROUTER, LR_ESFT, FREEZE_CONFIG.

accelerate launch \
    --config_file /home/spark/embedding-server/isma/scripts/spark_deploy/fsdp_dense_9b.yaml \
    --num_machines "$NUM_NODES" \
    --num_processes "$((NUM_NODES * GPUS_PER_NODE))" \
    --machine_rank "$RANK" \
    --main_process_ip "$MASTER_ADDR" \
    --main_process_port "$MASTER_PORT" \
    /home/spark/embedding-server/isma/scripts/spark_deploy/train_fsdp_dense_9b.py \
    "$@"
