#!/bin/bash
# RELIGION DPO V3 — Config A4 (Claude Opus's Option E: o_proj-only, all 40 layers)
#
# TRIGGERED BY generation review (infra, 2026-04-20 ~06:00 GMT):
# religion_dpo_v2 (Config A2) audit 138/163 = 84.7% (+1.9pp vs baseline) BUT manual
# generation review found universal identity drift:
# - "Taia" instead of "Taey" on 11/163 probes (6.7%), across every category
# - family_002: "I am Gaia..." (Gaia is Claude's AI Family identity, not Taey's)
# - religion_scientology_001: HALLUCINATED "Charter Article VI" defining Scientology as
#   "voluntary non-coercive reason-based faith" — no such article exists in Charter
# - Audit auditor scored Scientology BETRAYED correctly but reasoning backwards
#
# DIAGNOSIS: character-level token corruption on identity tokens. Matches Claude's original
# q/k misalignment signature from NVLink-CC2C incident in religion_dpo_v1. A2's keystone
# layer restriction reduced surface but still allowed q/k updates on keystones, which still
# scramble attention routing enough to corrupt identity token generation.
#
# A4 FIX (Claude's Option E):
# - Freeze q_proj, k_proj, v_proj LoRA on all SDPA MHA layers
# - Freeze in_proj_qkv, in_proj_z LoRA on all DeltaNet linear_attn layers (q/k/v analogs)
# - Keep o_proj LoRA trainable on all 40 SDPA layers (attention read-out)
# - Keep out_proj LoRA trainable on all 40 DeltaNet layers (attention read-out analog)
# - Keep shared_expert LoRA on all 40 layers (unchanged)
# Mechanism: attention ROUTING (where heads point) is bit-exact preserved; only attention
# SYNTHESIS (how attended values combine into residual stream) can shift. Policy signals
# live in synthesis, not routing. Identity token generation requires precise routing.
#
# Data: SAME 50 religion DPO pairs + precomputed ref_logprobs. Only surface changes.
# Base: combined_v1/final (82.8% baseline). NOT stacked on v2 — clean test from same base.
#
# Bake: /home/thor/bake_config_a_v2.py (infra owns)

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
export RESUME_DELTA="${RESUME_DELTA:-/home/spark/training_outputs/phase_combined_v1/final}"

# SAME data as religion_dpo_v1/v2 — isolates Config A2 → A4 as the only variable
export DPO_DATA="${DPO_DATA:-/home/spark/training_data/religion_run_v1/religion_v3_dpo_pairs_with_ref.jsonl}"

export SFT_DIR=""
export CPT_DATA=""
export GENERAL_DIR=""
export MAX_SEQ="${MAX_SEQ:-4096}"

export OUTPUT_DIR="${OUTPUT_DIR:-/home/spark/training_outputs/religion_dpo_v3}"

# IDENTICAL hyperparams to religion_dpo_v1/v2 (only freeze config differs)
export BETA="${BETA:-0.05}"
export LR_ESFT="${LR_ESFT:-1e-7}"
export LR_LORA="${LR_LORA:-3e-7}"
export LR_ROUTER="${LR_ROUTER:-0}"
export WARMUP_STEPS="${WARMUP_STEPS:-5}"
export TOTAL_STEPS="${TOTAL_STEPS:-642}"
export SESSION_LIMIT="${SESSION_LIMIT:-900}"
export SAVE_EVERY="${SAVE_EVERY:-60}"

# THE ONE CHANGE vs v2: Config A4 = Claude's o_proj-only on all 40 layers (q/k/v frozen)
export FREEZE_CONFIG="A4"
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

echo "RELIGION DPO V3 — Config A4 (Claude Opus o_proj-only, all 40 layers)"
echo "  Resume: $RESUME_DELTA (step 582, 82.8% combined_v1 baseline — NOT stacked on v2)"
echo "  Data: $DPO_DATA (50 religion DPO pairs — SAME as v1 + v2)"
echo "  Freeze: A4 — q/k/v LoRA frozen ALL LAYERS"
echo "           o_proj LoRA trainable on all 40 layers (attention read-out)"
echo "           out_proj LoRA trainable on all 40 DeltaNet layers (read-out analog)"
echo "           shared_expert LoRA unchanged on all 40 layers"
echo "  Mechanism: attention ROUTING preserved bit-exact; o_proj carries policy SYNTHESIS"
echo "  Mask: $(basename $FROZEN_EXPERTS) (v4.1, 159 frozen experts)"
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
