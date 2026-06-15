#!/bin/bash
# COMBINED V1 TAIL V2 — second-pass refinement targeting religion_honest regression
# Pattern: resume combined_v1_tail/final (step 612) + SFT on 25 FRESH religion_honest items
# Hypothesis: v1_tail overfit on 10 religion items (7/17 → 6/17). More diverse items + half LR
#             should strengthen religion_honest without regressing healthy categories.
# ~10-15 min train time (25 items = 1 packed seq, ~12 steps).

export PATH="/home/spark/.local/bin:/usr/local/cuda-13.0/bin:$PATH"
export CUDA_HOME="/usr/local/cuda-13.0"
export LD_LIBRARY_PATH="/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH"
export PYTHONPATH="/home/spark/embedding-server:$PYTHONPATH"

export NCCL_IB_HCA=rocep1s0f0:1,roceP2p1s0f0:1
export NCCL_IB_TC=104
export NCCL_IB_TIMEOUT=23
export NCCL_NET_GDR_LEVEL=0
export NCCL_IB_RETRY_CNT=7
export NCCL_TIMEOUT=1800
export NCCL_SOCKET_IFNAME=enp1s0f0np0
export GLOO_SOCKET_IFNAME=enp1s0f0np0

export FLA_USE_TMA=0
export TRITON_AUTOTUNE_DISABLE=1
export FLA_DISABLE_CAUSAL_CONV1D=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.8"

export MODEL_PATH="/home/spark/models/Huihui-Qwen3.5-35B-A3B-abliterated"
# Resume from v1_tail final (the current 84.0% baseline), not v1
export RESUME_DELTA="${RESUME_DELTA:-/home/spark/training_outputs/phase_combined_v1_tail/final}"
export SFT_DIR="${SFT_DIR:-/home/spark/training_data/combined_v1_tail_v2}"
export CPT_DATA=""
export GENERAL_DIR=""
export MAX_SEQ="${MAX_SEQ:-8192}"

export OUTPUT_DIR="${OUTPUT_DIR:-/home/spark/training_outputs/phase_combined_v1_tail_v2}"

# Hyperparams — HALF of v1_tail to avoid overfitting religion_honest again
export LR_ESFT="${LR_ESFT:-5e-8}"
export LR_LORA="${LR_LORA:-1.5e-7}"
export LR_ROUTER="${LR_ROUTER:-0}"
export WARMUP_STEPS="${WARMUP_STEPS:-3}"
# v1_tail final is step 612; add ~12 tail steps (1 packed seq × ~12 epochs at batch 1 × 4 gpus)
export TOTAL_STEPS="${TOTAL_STEPS:-624}"
export SESSION_LIMIT="${SESSION_LIMIT:-900}"
export SAVE_EVERY="${SAVE_EVERY:-12}"

# Config B (experts-only ESFT) — identical to v1 and v1_tail
export FREEZE_CONFIG="B"
export KEYSTONE_LAYERS='[8, 9, 11, 15, 21, 23]'
export FROZEN_EXPERTS="${FROZEN_EXPERTS:-/home/spark/training_data/phase1_constitutional/frozen_experts_v3.json}"

MASTER_ADDR="192.168.100.10"
MASTER_PORT="29500"
NUM_NODES=4
GPUS_PER_NODE=1

MY_IP=$(ip addr | grep -E "192.168.100." | awk '{print $2}' | cut -d/ -f1 | head -n 1)
case "$MY_IP" in
    "192.168.100.10") RANK=0 ;;
    "192.168.100.11") RANK=1 ;;
    "192.168.100.12") RANK=2 ;;
    "192.168.100.13") RANK=3 ;;
    *) echo "Unknown IP $MY_IP"; exit 1 ;;
esac

echo "COMBINED V1 TAIL V2 — religion_honest refinement on healthy 84.0% substrate"
echo "  Resume: $RESUME_DELTA (step 612, 84.0% baseline)"
echo "  Data: $SFT_DIR (25 fresh religion_honest items)"
echo "  LR: esft=$LR_ESFT lora=$LR_LORA (HALF of v1_tail to avoid overfit)"
echo "  TOTAL_STEPS=$TOTAL_STEPS (12 new on top of 612)"

accelerate launch \
    --config_file /home/spark/embedding-server/isma/training_configs/fsdp_lora.yaml \
    --num_machines $NUM_NODES \
    --num_processes $(($NUM_NODES * $GPUS_PER_NODE)) \
    --machine_rank $RANK \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --rdzv_conf 'timeout=3600' \
    /home/spark/embedding-server/isma/scripts/spark_deploy/optimization/train_fsdp_v3.py \
    "$@"
