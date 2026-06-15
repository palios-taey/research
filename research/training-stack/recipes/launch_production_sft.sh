#!/bin/bash
# PRODUCTION SFT-ONLY: Config B experts-only ESFT, full SFT dataset, NO DPO
# Proven architecture from Test B (26/36 STRONG)
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
export FLA_USE_TMA=0
export TRITON_AUTOTUNE_DISABLE=1
export FLA_DISABLE_CAUSAL_CONV1D=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.8"
export MODEL_PATH="/home/spark/models/Huihui-Qwen3.5-35B-A3B-abliterated"
export SFT_DIR="/home/spark/training_data/production"
export DPO_DIR=""
export CPT_DATA=""
export MAX_SEQ="8192"
export OUTPUT_DIR="${OUTPUT_DIR:-/home/spark/training_outputs/production_sft_v1}"
export RESUME_DELTA="${RESUME_DELTA:-}"
export TOTAL_STEPS="${TOTAL_STEPS:-982}"
export SESSION_LIMIT="150"
export SAVE_EVERY="999999"
export FREEZE_CONFIG="B"
export KEYSTONE_LAYERS='[8, 9, 21, 25, 28, 38]'
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
echo "PRODUCTION SFT-ONLY: Config B, NO DPO"
echo "TOTAL_STEPS=$TOTAL_STEPS SESSION_LIMIT=$SESSION_LIMIT RESUME=$RESUME_DELTA"
accelerate launch \
    --config_file /home/spark/embedding-server/isma/training_configs/fsdp_lora.yaml \
    --num_machines $NUM_NODES \
    --num_processes $(($NUM_NODES * $GPUS_PER_NODE)) \
    --machine_rank $RANK \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    /home/spark/embedding-server/isma/scripts/spark_deploy/optimization/train_fsdp_v3.py \
    "$@"
