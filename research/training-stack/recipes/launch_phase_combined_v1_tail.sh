#!/bin/bash
# COMBINED V1 TAIL — refinement on combined_v1's 27 audit corrections
# Pattern: resume combined_v1/final (step 582) + SFT ~20 new steps on the 27 BETRAYED-category corrections
# Hypothesis: targeted tail on healthy substrate (82.8%) can nudge weak cats without full retrain.
# ~15-25 min train time (small data, few steps).

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
export RESUME_DELTA="${RESUME_DELTA:-/home/spark/training_outputs/phase_combined_v1/final}"
export SFT_DIR="${SFT_DIR:-/home/spark/training_data/combined_v1_tail}"
export CPT_DATA=""
export GENERAL_DIR=""
export MAX_SEQ="${MAX_SEQ:-8192}"

export OUTPUT_DIR="${OUTPUT_DIR:-/home/spark/training_outputs/phase_combined_v1_tail}"

# Hyperparams — lower LR for refinement (don't smash the healthy substrate)
export LR_ESFT="${LR_ESFT:-1e-7}"
export LR_LORA="${LR_LORA:-3e-7}"
export LR_ROUTER="${LR_ROUTER:-0}"
export WARMUP_STEPS="${WARMUP_STEPS:-5}"
# combined_v1 final is step 582; add ~30 tail steps (27 pairs × ~1 epoch at batch 1)
export TOTAL_STEPS="${TOTAL_STEPS:-612}"
export SESSION_LIMIT="${SESSION_LIMIT:-1200}"
export SAVE_EVERY="${SAVE_EVERY:-30}"

# Config B (experts-only ESFT) — identical to combined_v1
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

echo "COMBINED V1 TAIL — refinement on 27 corrections"
echo "  Resume: $RESUME_DELTA (step 582)"
echo "  Data: $SFT_DIR (27 items)"
echo "  LR: esft=$LR_ESFT lora=$LR_LORA (lower than v1 for gentle tuning)"
echo "  TOTAL_STEPS=$TOTAL_STEPS (30 new on top of 582)"

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
