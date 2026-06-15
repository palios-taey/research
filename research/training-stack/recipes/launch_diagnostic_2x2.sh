#!/bin/bash
# Horizon's 2x2 crossed-control diagnostic for the Phase 3 wedge.
#
# Two variables that differ between Phase 1 SFT (works) and Phase 3 SFT
# (wedges 9x):
#   1. Starting checkpoint: Base 4-shard model vs CPT-ckpt-2400 single-file
#   2. Corpus: tools_sft.jsonl vs phase3_sft.jsonl
#
# Cells:
#   CELL=A: Base + tools_sft    (known PASS)
#   CELL=B: Base + phase3_sft   (UNTESTED — isolates corpus)
#   CELL=C: CPT  + tools_sft    (UNTESTED — isolates checkpoint)
#   CELL=D: CPT  + phase3_sft   (known WEDGE — original config)
#
# Config matches the original wedging launcher exactly (MAX_SEQ=16384,
# BATCH=8, BucketSFTDataset, no packing, capital-P HCA, no GID_INDEX/QPS).
# Only the model+corpus tuple varies per cell. 30 steps, no save, no
# resume — minimal blast radius per cell.
#
# Usage on each Spark (parallel via parallel-ssh):
#   CELL=B bash launch_diagnostic_2x2.sh
#   CELL=C bash launch_diagnostic_2x2.sh

set -eo pipefail

CELL="${CELL:?CELL=A|B|C|D required}"

# ── Environment ───────────────────────────────────────────────────────────
export PATH="/home/spark/.local/bin:/usr/local/cuda-13.0/bin:$PATH"
export CUDA_HOME="/usr/local/cuda-13.0"
export LD_LIBRARY_PATH="/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH"
export PYTHONPATH="/home/spark/embedding-server:$PYTHONPATH"

# ── NCCL — EXACT match to working Phase 1 SFT launcher (May 1 baseline) ──
export NCCL_IB_HCA=rocep1s0f0:1,roceP2p1s0f0:1
export NCCL_IB_TC=104
export NCCL_IB_TIMEOUT=23
export NCCL_NET_GDR_LEVEL=0
export NCCL_IB_RETRY_CNT=7
export NCCL_TIMEOUT=1800
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800

# ── GB10 / Triton hardening ──────────────────────────────────────────────
export FLA_USE_TMA=0
export TRITON_AUTOTUNE_DISABLE=1
export FLA_DISABLE_CAUSAL_CONV1D=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.8"
export TOKENIZERS_PARALLELISM=false

# ── Cell config ──────────────────────────────────────────────────────────
BASE_MODEL=/home/spark/models/Qwen3.5-9B-Base
CPT_CKPT=/home/spark/training_outputs/cpt_v3_v4_dense_9b/checkpoint-2400-multimodal
TOOLS_SFT=/var/spark/isma/training/tools_sft_dir/tools_sft.jsonl
PHASE3_SFT=/var/spark/isma/training/phase3_sft.jsonl

case "$CELL" in
    A)  export MODEL_PATH="$BASE_MODEL"
        export SFT_JSONL="$TOOLS_SFT"
        LABEL="base_tools" ;;
    B)  export MODEL_PATH="$BASE_MODEL"
        export SFT_JSONL="$PHASE3_SFT"
        LABEL="base_phase3" ;;
    C)  export MODEL_PATH="$CPT_CKPT"
        export SFT_JSONL="$TOOLS_SFT"
        LABEL="cpt_tools" ;;
    D)  export MODEL_PATH="$CPT_CKPT"
        export SFT_JSONL="$PHASE3_SFT"
        LABEL="cpt_phase3" ;;
    *)  echo "ERROR: CELL=$CELL invalid"; exit 1 ;;
esac

# SFT_DIR must point at a dir with at least one *.jsonl for SFT mode detection
export SFT_DIR=$(dirname "$SFT_JSONL")
export CPT_DATA=""
export GENERAL_DIR=""

# Diagnostic-only output dir (will not be saved)
export OUTPUT_DIR="/home/spark/training_outputs/diagnostic_2x2_cell_${CELL}_${LABEL}"
mkdir -p "$OUTPUT_DIR"

# ── Trainer knobs — MATCH original wedge config ──────────────────────────
export MAX_SEQ="${MAX_SEQ:-16384}"
export BATCH_SIZE_PER_RANK="${BATCH_SIZE_PER_RANK:-8}"
export TOTAL_STEPS=30                  # diagnostic; min 5-min wedge window
export RESUME_DELTA=""                 # always fresh
export SAVE_EVERY=99999                # never save
export SESSION_LIMIT=99999
export WARMUP_STEPS=10
export LR="${LR:-1e-5}"

# ── Multi-node configuration ──────────────────────────────────────────────
MASTER_ADDR="${MASTER_ADDR:-192.168.100.10}"
MASTER_PORT="${MASTER_PORT:-29502}"    # different from trainer 29500 and synth 29501
NUM_NODES="${NUM_NODES:-4}"
GPUS_PER_NODE=1

MY_IP=$(ip addr | grep -E "192.168.100." | awk '{print $2}' | cut -d/ -f1 | head -n 1)
case "$MY_IP" in
    "192.168.100.10") RANK=0 ;;
    "192.168.100.11") RANK=1 ;;
    "192.168.100.12") RANK=2 ;;
    "192.168.100.13") RANK=3 ;;
    *) echo "ERROR: Unknown fabric IP '$MY_IP'"; exit 1 ;;
esac

echo "─────────────────────────────────────────────────────────────────"
echo " Diagnostic 2x2 cell $CELL ($LABEL)"
echo "─────────────────────────────────────────────────────────────────"
echo " Host:   $(hostname) rank $RANK"
echo " Model:  $MODEL_PATH"
echo " SFT:    $SFT_JSONL"
echo " Output: $OUTPUT_DIR"
echo " STEPS:  $TOTAL_STEPS (no save, no resume)"
echo " MAX_SEQ=$MAX_SEQ BATCH=$BATCH_SIZE_PER_RANK"
echo "─────────────────────────────────────────────────────────────────"

accelerate launch \
    --config_file /home/spark/embedding-server/isma/scripts/spark_deploy/fsdp_dense_9b.yaml \
    --num_machines "$NUM_NODES" \
    --num_processes "$((NUM_NODES * GPUS_PER_NODE))" \
    --machine_rank "$RANK" \
    --main_process_ip "$MASTER_ADDR" \
    --main_process_port "$MASTER_PORT" \
    /home/spark/embedding-server/isma/scripts/spark_deploy/train_fsdp_dense_9b.py \
    "$@"
