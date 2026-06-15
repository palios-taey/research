#!/bin/bash
# COMBINED BIG V1 — scale phase_combined_v1's recipe (82.8% baseline) to 20K items
#
# Per 5-platform Chat consult 2026-04-20 (ChatGPT/Claude/Gemini/Grok/Perplexity):
# - Hold exact training surface (Config B experts-only, keystones [7,8,9,15,21,23])
# - Scale item count only, not architecture
# - 40/54/6 infra/identity/other ratio preserved
# - Lower peak LR for 10x data (1e-5 not 2e-5)
# - Warmup + cosine decay schedule
# - Save every 200 steps, audit each (religion_dpo_v2's 84.7% hid real drift; we need per-checkpoint review)
#
# Per Jesse's guidance: repeat curated high-priority infra as needed (947 items × 7x), don't
# pad with generic programming from infra_train.jsonl if it's not our stack.
#
# Split into 500-step sessions with fleet reboot between (observed Spark crash pattern at ~2h,
# SFT at ~20s/step means 500 steps ≈ 1.7h per session — safe margin).
#
# Output: fresh training from abliterated base (no RESUME_DELTA for session 1). Session 2+
# resume from checkpoint-500/checkpoint-1000/etc via RESUME_DELTA env override.

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
# Default: fresh abliterated start. Override RESUME_DELTA for subsequent sessions.
export RESUME_DELTA="${RESUME_DELTA-}"

export SFT_DIR="${SFT_DIR:-/home/spark/training_data/combined_big_v1}"
export CPT_DATA=""
export GENERAL_DIR=""
export MAX_SEQ="${MAX_SEQ:-8192}"                 # phase_combined_v1 packing target 7782

export OUTPUT_DIR="${OUTPUT_DIR:-/home/spark/training_outputs/combined_big_v1}"

# Per Chat synthesis: peak LR 1e-5 with warmup + cosine decay (not 2e-5 flat)
export LR_ESFT="${LR_ESFT:-1e-5}"                 # Peak ESFT LR (Chat consensus: half of v1's 2e-5)
export LR_LORA="${LR_LORA:-0}"                    # Config B = no LoRA
export LR_ROUTER="${LR_ROUTER:-0}"
export WARMUP_STEPS="${WARMUP_STEPS:-150}"        # ~5% of 2812 total steps

# Session structure: 1 epoch = 2,812 steps (11,247 packs / 4 GPUs).
# Split into ~7 sessions of 400 steps each with fleet reboot between (observed Spark ~2h crash pattern,
# 400 steps × ~20s/step SFT = 2.2h per session — at the edge but manageable).
export TOTAL_STEPS="${TOTAL_STEPS:-2812}"
export SESSION_LIMIT="${SESSION_LIMIT:-400}"      # Exit cleanly at 400-step increments, reboot between
export SAVE_EVERY="${SAVE_EVERY:-200}"            # Audit checkpoint every 200 steps (2 per session)

# Config B = experts-only, proven for mixed-phase identity+infra SFT
export FREEZE_CONFIG="B"
# phase_combined_v1's EXACT keystones (verified from Spark launch_phase_combined_v1.sh).
# NOT [7,8,9,15,21,23] — that was the Phase 1 Constitutional run's keystones, a different lineage.
export KEYSTONE_LAYERS='[8, 9, 11, 15, 21, 23]'
# phase_combined_v1's EXACT frozen_experts list: v3.json (31 frozen: Expert 95 + math/code top-3 per keystone).
# Wide-surgical freeze that produced 82.8%. Contains entries for all 40 layers; mask is applied
# only on keystone layers during training (non-keystone layers don't train experts anyway in Config B).
export FROZEN_EXPERTS="${FROZEN_EXPERTS:-/home/spark/training_data/phase1_infra/frozen_experts_v3.json}"

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

echo "COMBINED BIG V1 — scale phase_combined_v1 recipe to 20K items"
echo "  Base: abliterated (RESUME_DELTA=$RESUME_DELTA)"
echo "  Data: $SFT_DIR (19,660 items at 40.7/54.9/4.4 infra/identity/other)"
echo "  Config: B (experts-only, same as phase_combined_v1)"
echo "  Keystones: $KEYSTONE_LAYERS"
echo "  Frozen: $(basename $FROZEN_EXPERTS)"
echo "  MAX_SEQ=$MAX_SEQ  LR_ESFT=$LR_ESFT (peak, warmup $WARMUP_STEPS steps, cosine decay)"
echo "  TOTAL_STEPS=$TOTAL_STEPS  SESSION_LIMIT=$SESSION_LIMIT  SAVE_EVERY=$SAVE_EVERY"
echo "  Output: $OUTPUT_DIR"

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
