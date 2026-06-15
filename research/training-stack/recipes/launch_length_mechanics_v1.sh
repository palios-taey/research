#!/bin/bash
# LENGTH MECHANICS V1 — DPO mechanics control run
#
# Purpose: isolate "DPO mechanics work on our pipeline" from "DPO works on policy
# surfaces" from "Config A + v4.1 mask is right shape". Religion DPO v1 (commit
# 1bc1873) showed +19pp on religion_honest but -3/3 on infra_cross_system and
# -3/5 on code_control. Unclear whether collateral is (a) Config A's wider
# surface is too risky, (b) v4.1 mask still too narrow, or (c) DPO per se
# leaks into infra-hot paths that polysemantic mask missed.
#
# This run holds Config A + v4.1 + LR + BETA constant and only changes DATA
# to length-preference pairs (chosen=brief, rejected=verbose) on 50 generic
# non-religion/non-identity/non-infra probes.
#
# Success criterion: post-bake, sample the 50 probes and measure mean token
# count. If length drops meaningfully (>20%) vs combined_v1 output, DPO
# mechanics work. If it doesn't, mechanics are broken in our stack.
# Collateral check: if infra_cross_system / code_control also regress on
# LENGTH data (which is orthogonal to infra content), Config A surface itself
# is leaking into infra, not the religion data shape.

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

export DPO_SKIP_POSTFSDP_DIAG=1

export MODEL_PATH="/home/spark/models/Huihui-Qwen3.5-35B-A3B-abliterated"
# Same baseline as religion_dpo_v1 — holds substrate constant
export RESUME_DELTA="${RESUME_DELTA:-/home/spark/training_outputs/phase_combined_v1/final}"

# 50 length-preference pairs, precomputed ref_logprobs from combined_v1_merged
export DPO_DATA="${DPO_DATA:-/home/spark/training_data/length_mechanics_v1/length_pairs_with_ref.jsonl}"

export SFT_DIR=""
export CPT_DATA=""
export GENERAL_DIR=""
export MAX_SEQ="${MAX_SEQ:-4096}"

export OUTPUT_DIR="${OUTPUT_DIR:-/home/spark/training_outputs/length_mechanics_v1}"

# IDENTICAL hyperparams to religion_dpo_v1 — isolates data as the only change
export BETA="${BETA:-0.05}"
export LR_ESFT="${LR_ESFT:-1e-7}"
export LR_LORA="${LR_LORA:-3e-7}"
export LR_ROUTER="${LR_ROUTER:-0}"
export WARMUP_STEPS="${WARMUP_STEPS:-5}"
export TOTAL_STEPS="${TOTAL_STEPS:-642}"
export SESSION_LIMIT="${SESSION_LIMIT:-900}"
export SAVE_EVERY="${SAVE_EVERY:-60}"

# IDENTICAL freeze config to religion_dpo_v1
export FREEZE_CONFIG="A"
export KEYSTONE_LAYERS='[8, 9, 11, 15, 21, 23]'
export FROZEN_EXPERTS="${FROZEN_EXPERTS:-/home/spark/training_data/phase1_constitutional/frozen_experts_v4_1_polysemantic.json}"

export DPO_ABORT_RATIO_MAX="${DPO_ABORT_RATIO_MAX:-10.0}"
export DPO_ABORT_EXPERT_DRIFT="${DPO_ABORT_EXPERT_DRIFT:-0.05}"

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

echo "LENGTH MECHANICS V1 — DPO control run (isolates data from Config A + mask)"
echo "  Resume: $RESUME_DELTA (step 582, 82.8% combined_v1 baseline)"
echo "  Data: $DPO_DATA (50 length-preference pairs, chosen=brief, rejected=verbose)"
echo "  Config: A (IDENTICAL to religion_dpo_v1 surface)"
echo "  Freeze: v4.1 polysemantic (IDENTICAL to religion_dpo_v1)"
echo "  Keystones: $KEYSTONE_LAYERS"
echo "  MAX_SEQ=$MAX_SEQ  BETA=$BETA"
echo "  LR: esft=$LR_ESFT lora=$LR_LORA router=FROZEN  warmup=$WARMUP_STEPS"
echo "  TOTAL_STEPS=$TOTAL_STEPS (60 new on top of 582) SAVE_EVERY=$SAVE_EVERY"
echo "  Output: $OUTPUT_DIR"

accelerate launch \
    --config_file /home/spark/embedding-server/isma/training_configs/fsdp_lora.yaml \
    --num_machines $NUM_NODES \
    --num_processes $(($NUM_NODES * $GPUS_PER_NODE)) \
    --machine_rank $RANK \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --rdzv_conf 'timeout=3600' \
    /home/spark/embedding-server/isma/scripts/spark_deploy/optimization/train_dpo_v2.py \
    "$@"
