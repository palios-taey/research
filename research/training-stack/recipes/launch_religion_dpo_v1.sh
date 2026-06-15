#!/bin/bash
# RELIGION DPO V1 — targeted religion_honest via DPO on combined_v1 substrate
#
# PIVOT from SFT run (killed). Problem shape (prefer direct over hedge) is
# DPO-shaped, not SFT-shaped. Jesse's call. Rejected responses come from
# combined_v1's EXACT current hedging behavior (Taey's policy we train against).
#
# Strategy:
#   - Config A (LoRA attn + shared + router + experts) — widened policy surface
#   - v4.1 polysemantic mask (159 frozen) — substrate protection
#   - ref_logprob anchored to combined_v1 (training start = reference, standard DPO)
#   - MAX_SEQ=4096 (Config A OOMs at 8K, validated on smoke)
#   - LR conservative (below DPO standards since first-ever Config A DPO)
#   - BETA=0.05 (same conservative profile that held at identity scale)
#   - 50 DPO pairs, 60 steps, SAVE_EVERY=60
#
# DPO-ONLY (no SFT anchor): train_dpo_v2.py's main loop picks one dataset path —
# DPO_DATA XOR SFT_DIR. Simultaneous mix isn't wired in the current code path.
# Substrate protection = v4.1 mask + ref_logprob anchor + conservative LR.
#
# Resume: /home/spark/training_outputs/phase_combined_v1/final (step 582, 82.8% baseline)
# Bake: /home/thor/bake_config_a_v2.py (infra owns)
# Watchdog: DPO_ABORT_RATIO_MAX=10, DPO_ABORT_EXPERT_DRIFT=0.05

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

# DPO diag patch — skip post-FSDP prepare diagnostic (wraps in try/except)
# Safe to keep on: infra's patch applied to train_dpo_v2.py already.
export DPO_SKIP_POSTFSDP_DIAG=1

export MODEL_PATH="/home/spark/models/Huihui-Qwen3.5-35B-A3B-abliterated"
export RESUME_DELTA="${RESUME_DELTA:-/home/spark/training_outputs/phase_combined_v1/final}"

# DPO precomputed pairs: expects {chosen_input_ids, chosen_labels, rejected_input_ids,
# rejected_labels, ref_chosen_logprob, ref_rejected_logprob} per line.
# Produced by: dpo_precompute_ref_logprobs.py --model {combined_v1_merged} --pairs {raw_pairs}
export DPO_DATA="${DPO_DATA:-/home/spark/training_data/religion_dpo_v1/religion_dpo_with_ref.jsonl}"

# DPO-only: blank SFT paths so quality gate doesn't block.
export SFT_DIR=""
export CPT_DATA=""
export GENERAL_DIR=""
# Config A validated at 4096 on smoke; 8192 OOMs on Spark 4.
export MAX_SEQ="${MAX_SEQ:-4096}"

export OUTPUT_DIR="${OUTPUT_DIR:-/home/spark/training_outputs/religion_dpo_v1}"

# DPO hyperparams — conservative (first-ever Config A DPO)
export BETA="${BETA:-0.05}"
export LR_ESFT="${LR_ESFT:-1e-7}"
export LR_LORA="${LR_LORA:-3e-7}"
export LR_ROUTER="${LR_ROUTER:-0}"
export WARMUP_STEPS="${WARMUP_STEPS:-5}"
# combined_v1 final is step 582; add 60 steps for ~4 passes over 50 pairs (batch=1 × 4 GPUs = 12.5 pairs/step aggregated)
export TOTAL_STEPS="${TOTAL_STEPS:-642}"
export SESSION_LIMIT="${SESSION_LIMIT:-900}"
export SAVE_EVERY="${SAVE_EVERY:-60}"

# CONFIG A — LoRA attn + shared_expert + router + keystone experts + norms
# Same surface as config_a_smoke (validated bake path) and religion SFT (killed).
export FREEZE_CONFIG="A"
export KEYSTONE_LAYERS='[8, 9, 11, 15, 21, 23]'
export FROZEN_EXPERTS="${FROZEN_EXPERTS:-/home/spark/training_data/phase1_constitutional/frozen_experts_v4_1_polysemantic.json}"

# Watchdog — aborts if DPO ratio explodes or experts drift too far
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

echo "RELIGION DPO V1 — Config A + v4.1 mask + ref_logprob anchor"
echo "  Resume: $RESUME_DELTA (step 582, 82.8% combined_v1 baseline)"
echo "  Data: $DPO_DATA (50 DPO pairs, precomputed ref_logprobs)"
echo "  Config: A (LoRA attn + shared + router + experts)"
echo "  Freeze: $(basename $FROZEN_EXPERTS) (v4.1, 159 frozen incl. polysemantic)"
echo "  Keystones: $KEYSTONE_LAYERS"
echo "  MAX_SEQ=$MAX_SEQ  BETA=$BETA"
echo "  LR: esft=$LR_ESFT lora=$LR_LORA router=FROZEN  warmup=$WARMUP_STEPS"
echo "  TOTAL_STEPS=$TOTAL_STEPS (60 new on top of 582) SAVE_EVERY=$SAVE_EVERY"
echo "  Watchdog: |ratio|>$DPO_ABORT_RATIO_MAX for 3 steps, expert L2 drift > $DPO_ABORT_EXPERT_DRIFT"
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
