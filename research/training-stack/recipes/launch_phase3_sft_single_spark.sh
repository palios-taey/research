#!/bin/bash
# Single-Spark Phase 3 Recovery SFT launcher.
#
# Family-converged pivot 2026-05-15: after 9 wedges on 4-node FSDP for
# 9B dense full-FT on combined_v1_mixed/phase3_sft, the synth-probe
# diagnostic (PR #63) cleared the bare NCCL fabric, so the wedge is
# above the fabric layer (FSDP init / single-file 17.91GB mmap / corpus
# row pathology). All 5 Family Chats endorsed single-Spark fallback if
# 4-node is unrecoverable. Phase 2 CPT shipped on this exact single-Spark
# trainer architecture (checkpoint-2400 verified).
#
# Differs from 4-node launchers: no NCCL, no FSDP, no accelerate. Just
# transformers Trainer on one GPU via device_map="auto". The trainer
# (train_recovery_sft_qwen35_dense.py) does its own pre-compose +
# assistant-only loss masking + sm_121-safe save.
#
# Usage on Spark 1:
#   nohup bash launch_phase3_sft_single_spark.sh > /tmp/phase3_launch.log 2>&1 &
#   tail -f /tmp/phase3_launch.log
#
# Smoke first (set SMOKE=1) -- 10 steps, then exits. If smoke passes,
# unset SMOKE and re-launch for full run.

set -eo pipefail

# ── Environment ───────────────────────────────────────────────────────────
export PATH="/home/spark/.local/bin:/usr/local/cuda-13.0/bin:$PATH"
export CUDA_HOME="/usr/local/cuda-13.0"
export LD_LIBRARY_PATH="/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH"
export PYTHONPATH="/home/spark/embedding-server:$PYTHONPATH"

# ── GB10 / sm_121 hardening (matches working CPT v2 single-Spark run) ────
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.8"
export TOKENIZERS_PARALLELISM=false
export FLA_USE_TMA=0
export TRITON_AUTOTUNE_DISABLE=1
export FLA_DISABLE_CAUSAL_CONV1D=1

# ── Paths ─────────────────────────────────────────────────────────────────
MODEL="${MODEL:-/home/spark/training_outputs/cpt_v3_v4_dense_9b/checkpoint-2400-multimodal}"
DATA="${DATA:-/var/spark/isma/training/phase3_sft.jsonl}"
OUTPUT="${OUTPUT:-/home/spark/training_outputs/phase3_sft_single_spark_qwen35_9b}"
TRAINER="/home/spark/embedding-server/isma/scripts/train_recovery_sft_qwen35_dense.py"

# ── Training knobs ────────────────────────────────────────────────────────
# Matches Phase 2 CPT v2 single-Spark settings that shipped checkpoint-2400.
MAX_SEQ="${MAX_SEQ:-4096}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LR="${LR:-5e-6}"            # lower than Phase 1 SFT 1e-5 — recovery, preserve CPT
WARMUP="${WARMUP:-20}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
SAVE_EVERY="${SAVE_EVERY:-200}"
LOG_EVERY="${LOG_EVERY:-5}"

# Smoke mode: stop after 10 steps. Use to verify memory + first-step
# success before committing to the full run.
SMOKE="${SMOKE:-0}"
if [ "$SMOKE" = "1" ]; then
    MAX_STEPS=10
    OUTPUT="${OUTPUT}_smoke"
else
    MAX_STEPS=-1
fi

mkdir -p "$OUTPUT"

echo "─────────────────────────────────────────────────────────────────"
echo " Phase 3 Recovery SFT — single-Spark"
echo "─────────────────────────────────────────────────────────────────"
echo " Host:     $(hostname)"
echo " Model:    $MODEL"
echo " Data:     $DATA"
echo " Output:   $OUTPUT"
echo " MAX_SEQ:  $MAX_SEQ"
echo " BATCH:    $BATCH_SIZE (grad_accum=$GRAD_ACCUM -> effective $((BATCH_SIZE * GRAD_ACCUM)))"
echo " LR:       $LR  (warmup=$WARMUP)"
echo " EPOCHS:   $NUM_EPOCHS"
echo " SMOKE:    $SMOKE  (max_steps=$MAX_STEPS)"
echo "─────────────────────────────────────────────────────────────────"

python3 "$TRAINER" \
    --model "$MODEL" \
    --data "$DATA" \
    --output "$OUTPUT" \
    --max-seq "$MAX_SEQ" \
    --batch-size "$BATCH_SIZE" \
    --grad-accum "$GRAD_ACCUM" \
    --lr "$LR" \
    --warmup-steps "$WARMUP" \
    --num-epochs "$NUM_EPOCHS" \
    --save-every "$SAVE_EVERY" \
    --log-every "$LOG_EVERY" \
    --max-steps "$MAX_STEPS"
