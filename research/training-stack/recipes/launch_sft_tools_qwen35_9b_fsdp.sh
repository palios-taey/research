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

# ── NCCL — Blackwell / DGX Spark proven recipe ────────────────────────────
# DUAL-RAIL FIX 2026-05-12: actual RoCE device names are rocep1s0f0 (lower p)
# and roceP2p1s0f0 (CAPITAL P). Prior lowercase 'rocep2s0f0' silently fell back
# to single-rail because the second device doesn't exist under that name. CPT
# and Phase 1 SFT survived on single-rail (200Gb) bandwidth, but Phase 3 SFT's
# variable-length burstier collectives peaked above 200Gb → IBV_WC_RETRY_EXC_ERR
# → Spark 2 wedged. Verified device names via /sys/class/infiniband on all 4
# Sparks. Both rails are ACTIVE 200Gb/sec = 400Gb total fabric.
export NCCL_IB_HCA=rocep1s0f0:1,roceP2p1s0f0:1
export NCCL_IB_TC=104
export NCCL_IB_TIMEOUT=23
export NCCL_NET_GDR_LEVEL=0
export NCCL_IB_RETRY_CNT=7
export NCCL_TIMEOUT=1800
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
# GID INDEX PIN 2026-05-12: per Perplexity Deep Research, NCCL auto-selects GID
# per node and can diverge across hosts when network is provisioned by different
# tools (netplan vs hand-crafted NM). Verified GID tables are currently identical
# (RoCEv2 IPv4 at index 3 on all 4 Sparks) but explicit pin removes the variability.
# Canonical NVIDIA recommendation for RoCEv2 IPv4 fabrics.
export NCCL_IB_GID_INDEX=3
# QPS_PER_CONNECTION 2026-05-12: Perplexity recommends =4 for multi-path ECMP
# entropy. Default is 1. Higher values (8/16) don't help — VQoS bug 4222773
# threshold is 350 QPs on the device; 4 ranks × 3 peer-pairs × 4 QPs = 48 QPs,
# nowhere near 350. =4 provides path diversity without WR-queue overhead.
export NCCL_IB_QPS_PER_CONNECTION=4

# ── FLA / Triton — GB10 sm_121 hardening ──────────────────────────────────
export FLA_USE_TMA=0
export TRITON_AUTOTUNE_DISABLE=1
export FLA_DISABLE_CAUSAL_CONV1D=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.8"
export TOKENIZERS_PARALLELISM=false

# ── Training paths ────────────────────────────────────────────────────────
export MODEL_PATH="${MODEL_PATH:-/home/spark/models/Qwen3.5-9B-Base}"
# train_fsdp_dense_9b.py expects SFT_DIR (a dir containing *.jsonl files), not
# a single file. Point at the dir; corpus filename inside is fixed.
export SFT_DIR="${SFT_DIR:-/var/spark/isma/training/tools_sft_dir}"
export CPT_DATA="${CPT_DATA:-}"
export GENERAL_DIR="${GENERAL_DIR:-}"
export OUTPUT_DIR="${OUTPUT_DIR:-/home/spark/training_outputs/sft_tools_qwen35_9b_fsdp}"
mkdir -p "$OUTPUT_DIR"

# ── Trainer knobs (proven train_fsdp_v3.py reads these from env) ─────────
export MAX_SEQ="${MAX_SEQ:-16384}"             # safe cap below 32K SDPA NaN, fits all but tiny outliers
export BATCH_SIZE_PER_RANK="${BATCH_SIZE_PER_RANK:-8}"    # batch=8 / rank × 4 ranks = effective batch 32 (proven v8.4 baseline)
# Total steps for one epoch: 139748 samples (no-truncate corpus) / (BATCH × world)
# At batch=8 / rank × 4 ranks: 139748 / 32 = 4367 steps per epoch.
export TOTAL_STEPS="${TOTAL_STEPS:-4367}"
# Resume from a saved checkpoint when set (relative or absolute path)
export RESUME_DELTA="${RESUME_DELTA:-}"
export SAVE_EVERY="${SAVE_EVERY:-200}"
export SESSION_LIMIT="${SESSION_LIMIT:-99999}"     # disable periodic session-limit; run straight to TOTAL_STEPS
export WARMUP_STEPS="${WARMUP_STEPS:-100}"
export LR="${LR:-1e-5}"                            # full-FT learning rate (TOOLS.md spec)

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
