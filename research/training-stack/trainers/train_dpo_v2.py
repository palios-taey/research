#!/usr/bin/env python3
"""PALIOS-TAEY v3: Hybrid LoRA + ESFT on 4x DGX Spark GB10.

Family consultation consensus (Mar 29, 2026):
- LoRA on shared expert + attention (standard Linear modules)
- Direct ESFT on expert tensors in super-expert layers (batched 3D tensors)
- Full fine-tune on router gates at lower LR
- Combined SFT (no separate CPT phase)
- 250-step sessions with clean exit (UMA fragmentation)

Qwen3.5-35B-A3B stores routed experts as batched 3D tensors [256, dim, dim].
Standard PEFT LoRA can't target individual experts. Hybrid approach:
  - PEFT LoRA on shared_expert + attention projections
  - Direct parameter unfreeze on expert tensors in keystone layers
  - Routing naturally focuses gradient on hot experts within unfrozen layers

Launch (on each node, change --machine_rank):
    NCCL_IB_HCA=rocep1s0f0:1,roceP2p1s0f0:1 NCCL_IB_TC=104 \
    accelerate launch --config_file isma/training_configs/fsdp_cpt.yaml \
      --machine_rank 0 --main_process_ip 192.168.100.11 --main_process_port 29500 \
      isma/scripts/train_lora_fsdp.py
"""

import os, gc, random, sys
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,garbage_collection_threshold:0.8")
os.environ.setdefault("NCCL_NET_GDR_LEVEL", "0")
os.environ.setdefault("NCCL_TIMEOUT", "1800")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("FLA_DISABLE_CAUSAL_CONV1D", "1")
os.environ.setdefault("TRITON_AUTOTUNE_DISABLE", "1")
os.environ.setdefault("FLA_USE_TMA", "0")
os.environ.setdefault("NCCL_IB_TIMEOUT", "23")

# ── ChatGPT diagnostic: disable FLA to test if Blackwell Triton kernels cause NaN ──
if os.environ.get("DISABLE_FLA", "0") == "1":
    import transformers.models.qwen3_5_moe.modeling_qwen3_5_moe as q35
    q35.chunk_gated_delta_rule = None
    q35.fused_recurrent_gated_delta_rule = None
    q35.FusedRMSNormGated = None
    q35.causal_conv1d_fn = None
    q35.causal_conv1d_update = None
    print("[DIAG] FLA DISABLED — using PyTorch fallback for linear attention")
os.environ.setdefault("NCCL_IB_RETRY_CNT", "7")
os.environ.setdefault("TORCH_NCCL_DUMP_ON_TIMEOUT", "1")
os.environ.setdefault("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", "1800")

import json
import math
import logging
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
import psutil
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, DistributedSampler, Dataset
import glob

# Add this script's directory to sys.path so we can import fsdp_expert_masking.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════
# Keystone Layers — selected by routing probe (Mar 29, 2026)
# These have the strongest super-expert concentrations (50-80% of tokens).
# Expert tensors in these layers are fully unfrozen (ESFT).
# All 256 experts train, but routing focuses gradient on hot experts.
# ═══════════════════════════════════════════════════════════════════
KEYSTONE_LAYERS = [17, 28]  # 2 layers — 3.2GB optimizer/node, fits 5.7GB headroom
# L17: 16 T1 safe infra experts. L28: 19 T1 safe infra experts (densest).
# 2 layers = 1.6B params, FSDP shards to 400M/node, 3.2GB optimizer/node

# Per keystone layer: ~805M params (gate_up_proj [256,1024,2048] + down_proj [256,2048,512])

# Trainable components (besides keystone expert tensors):
# - shared_expert gate/up/down_proj (all 40 layers) — LoRA r=64
# - attention projections (all 40 layers, both SDPA + DeltaNet) — LoRA r=64
# - router gates mlp.gate + shared_expert_gate (all 40 layers) — full, lower LR
# - layernorms (all layers) — full
# - embeddings/lm_head — frozen for v1 (conservative)


def _clean_fsdp_name(name):
    """Strip FSDP wrapper prefixes to get canonical model parameter name."""
    return name.replace("_fsdp_wrapped_module.", "").replace("module.", "")


def _is_trainable(name, keystone_layers):
    """Determine if a parameter should be trainable in the hybrid approach."""
    clean = _clean_fsdp_name(name)

    # LoRA adapter weights — always trainable (PEFT handles this)
    if 'lora_' in clean.lower():
        return True

    # Keystone expert tensors — direct ESFT on selected layers
    for kl in keystone_layers:
        if f'layers.{kl}.mlp.experts.' in clean:
            return True

    # Router gates — full fine-tune (lower LR group)
    if 'mlp.gate.weight' in clean or 'shared_expert_gate' in clean:
        return True

    # Layer norms — full fine-tune
    if 'layernorm' in clean or 'norm' in clean:
        return True

    # shared_expert projections that PEFT wraps — PEFT handles requires_grad
    # (These will be set by get_peft_model if they're in modules_to_save or target_modules)

    return False

def save_lora_only_fsdp(model, accelerator, out_dir, adapter_name="default"):
    """Save only trainable PEFT weights from an FSDP-wrapped model via DCP."""
    accelerator.wait_for_everyone()

    options = StateDictOptions(
        full_state_dict=True,
        cpu_offload=True,
        ignore_frozen_params=True,
    )
    trainable_state = get_model_state_dict(model, options=options)

    total_gb = 0.0
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        total_gb = sum(
            tensor.numel() * tensor.element_size()
            for tensor in trainable_state.values()
        ) / 1e9
        log.info(
            "Saving %d trainable tensors to %s (%.2f GB)",
            len(trainable_state),
            out_dir,
            total_gb,
        )
        unwrapped.save_pretrained(
            out_dir,
            selected_adapters=[adapter_name],
            state_dict=trainable_state,
            safe_serialization=True,
            save_embedding_layers=False,
            is_main_process=True,
        )

    accelerator.wait_for_everyone()
    return total_gb

def _tokenize_sft_pair(messages, tokenizer):
    """Tokenize SFT conversation with assistant-only loss.

    Uses incremental template application to find exact assistant token boundaries.
    Handles Qwen3.5 chat template which adds special tokens around content.
    """
    try:
        full_text = tokenizer.apply_chat_template(messages, tokenize=False,
                                                  add_generation_prompt=False, enable_thinking=False)
    except Exception:
        parts = [f"<|{m['role']}|>\n{m['content']}" for m in messages]
        full_text = "\n".join(parts) + tokenizer.eos_token

    full_ids = tokenizer.encode(full_text, add_special_tokens=False)
    labels = [-100] * len(full_ids)

    # Build prefix up to each assistant message to find exact token boundaries
    for i, m in enumerate(messages):
        if m["role"] != "assistant":
            continue
        # Template up to (but not including) this assistant message
        prefix_msgs = messages[:i]
        try:
            prefix_text = tokenizer.apply_chat_template(
                prefix_msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except Exception:
            prefix_text = ""

        # Template including this assistant message
        incl_msgs = messages[:i+1]
        try:
            incl_text = tokenizer.apply_chat_template(
                incl_msgs, tokenize=False, add_generation_prompt=False, enable_thinking=False)
        except Exception:
            incl_text = full_text

        prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False) if prefix_text else []
        incl_ids = tokenizer.encode(incl_text, add_special_tokens=False)

        # Assistant tokens are between prefix end and incl end
        start = len(prefix_ids)
        end = len(incl_ids)
        for j in range(start, min(end, len(full_ids))):
            labels[j] = full_ids[j]

    # If no labels were set (fallback), train on everything
    if all(l == -100 for l in labels):
        labels = list(full_ids)

    return full_ids, labels


class DPOPairsDataset(Dataset):
    """DPO pairs dataset — loads precomputed {chosen,rejected}_{input_ids,labels} + ref_logprobs.

    Expected format per line (from dpo_precompute_ref_logprobs.py):
    {
      "probe_id": str, "category": str,
      "prompt": str, "chosen": str, "rejected": str,
      "chosen_input_ids": [int,...], "chosen_labels": [int,...],
      "rejected_input_ids": [int,...], "rejected_labels": [int,...],
      "ref_chosen_logprob": float, "ref_rejected_logprob": float,
    }
    """
    def __init__(self, dpo_data_path, max_seq=8192):
        self.max_seq = max_seq
        self.items = []
        if not dpo_data_path or not os.path.exists(dpo_data_path):
            log.warning(f"DPO data not found: {dpo_data_path}")
            return
        with open(dpo_data_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                # Validate required fields
                required = ['chosen_input_ids', 'chosen_labels', 'rejected_input_ids',
                           'rejected_labels', 'ref_chosen_logprob', 'ref_rejected_logprob']
                if all(k in rec for k in required):
                    self.items.append(rec)
        log.info(f"DPO dataset: loaded {len(self.items)} pairs from {dpo_data_path}")

    def __len__(self):
        return max(1, len(self.items))

    def __getitem__(self, idx):
        if not self.items:
            # Placeholder for empty dataset (shouldn't run training in this state)
            return {
                "chosen_input_ids": torch.zeros(1, dtype=torch.long),
                "chosen_labels": torch.full((1,), -100, dtype=torch.long),
                "rejected_input_ids": torch.zeros(1, dtype=torch.long),
                "rejected_labels": torch.full((1,), -100, dtype=torch.long),
                "ref_chosen_logprob": torch.tensor(0.0),
                "ref_rejected_logprob": torch.tensor(0.0),
            }
        rec = self.items[idx % len(self.items)]
        cho_ids = rec['chosen_input_ids'][:self.max_seq]
        cho_labels = rec['chosen_labels'][:self.max_seq]
        rej_ids = rec['rejected_input_ids'][:self.max_seq]
        rej_labels = rec['rejected_labels'][:self.max_seq]
        return {
            "chosen_input_ids": torch.tensor(cho_ids, dtype=torch.long),
            "chosen_labels": torch.tensor(cho_labels, dtype=torch.long),
            "rejected_input_ids": torch.tensor(rej_ids, dtype=torch.long),
            "rejected_labels": torch.tensor(rej_labels, dtype=torch.long),
            "ref_chosen_logprob": torch.tensor(rec['ref_chosen_logprob'], dtype=torch.float32),
            "ref_rejected_logprob": torch.tensor(rec['ref_rejected_logprob'], dtype=torch.float32),
        }


class CombinedSFTDataset(Dataset):
    """Combined SFT dataset — zero-footprint byte-offset indexing.
    Weighted sampling: 55% SFT, 30% CPT (constitutional+infra), 15% general.

    NOTE: Not used in DPO v2 — kept for reference / compatibility with train_fsdp_v3 base code.
    """

    def __init__(self, sft_dir, cpt_path, general_dir, tokenizer, max_seq):
        self.tokenizer = tokenizer
        self.max_seq = max_seq

        # MANDATORY: verify quality gate was run on SFT data
        # Quality gate chunks long items, deduplicates, and packs sequences.
        # Training on ungated data truncates long items silently.
        log.info("Building byte-offset indexes...")
        sft_files = sorted(glob.glob(os.path.join(sft_dir, "*.jsonl"))) if sft_dir and os.path.isdir(sft_dir) else []
        for f in sft_files:
            if 'gated' not in os.path.basename(f):
                raise ValueError(
                    f"QUALITY GATE REQUIRED: {f} does not contain 'gated' in filename. "
                    f"Run training_data_gate.py on the data first. "
                    f"Training on ungated data truncates long items and loses content."
                )
        cpt_files = [cpt_path] if cpt_path and os.path.exists(cpt_path) else []
        general_files = sorted(glob.glob(os.path.join(general_dir, "*.jsonl"))) if general_dir and os.path.isdir(general_dir) else []

        self.sft_index = self._build_index(sft_files)
        self.cpt_index = self._build_index(cpt_files)
        self.general_index = self._build_index(general_files)

        total = len(self.sft_index) + len(self.cpt_index) + len(self.general_index)
        self.total_len = max(total, 1)
        log.info(f"Dataset: {len(self.sft_index)} SFT + {len(self.cpt_index)} CPT + "
                 f"{len(self.general_index)} General = {self.total_len}")

    def _build_index(self, file_paths):
        index = []
        for path in file_paths:
            if not os.path.isfile(path):
                continue
            with open(path, 'rb') as f:
                while True:
                    offset = f.tell()
                    line = f.readline()
                    if not line:
                        break
                    if line.strip():
                        index.append((path, offset))
        return index

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        # Weighted sampling: 55% SFT, 30% CPT, 15% general
        r = random.random()
        if r < 0.55 and self.sft_index:
            source, is_cpt = self.sft_index, False
        elif r < 0.85 and self.cpt_index:
            source, is_cpt = self.cpt_index, True
        elif self.general_index:
            source, is_cpt = self.general_index, False
        elif self.sft_index:
            source, is_cpt = self.sft_index, False
        elif self.cpt_index:
            source, is_cpt = self.cpt_index, True
        else:
            return {"input_ids": [0] * self.max_seq, "labels": [-100] * self.max_seq}

        path, offset = source[idx % len(source)]
        with open(path, 'rb') as f:
            f.seek(offset)
            raw = f.readline()

        try:
            data = json.loads(raw.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {"input_ids": [self.tokenizer.pad_token_id] * self.max_seq,
                    "labels": [-100] * self.max_seq}

        if is_cpt:
            text = data.get("text", "")
            tokens = self.tokenizer.encode(text, add_special_tokens=False) + [self.tokenizer.eos_token_id]
            labels = list(tokens)
        elif "messages" in data:
            try:
                tokens, labels = _tokenize_sft_pair(data["messages"], self.tokenizer)
            except Exception:
                tokens = [self.tokenizer.pad_token_id] * self.max_seq
                labels = [-100] * self.max_seq
        elif "chosen" in data and "prompt" in data and "rejected" in data:
            # DPO item — tokenize both chosen and rejected
            chosen_msgs = [{"role": "user", "content": data["prompt"]},
                           {"role": "assistant", "content": data["chosen"]}]
            rejected_msgs = [{"role": "user", "content": data["prompt"]},
                             {"role": "assistant", "content": data["rejected"]}]
            try:
                tokens, labels = _tokenize_sft_pair(chosen_msgs, self.tokenizer)
                rej_tokens, rej_labels = _tokenize_sft_pair(rejected_msgs, self.tokenizer)
            except Exception:
                tokens = [self.tokenizer.pad_token_id] * self.max_seq
                labels = [-100] * self.max_seq
                rej_tokens = tokens[:]
                rej_labels = labels[:]

            # Pad/truncate rejected
            if len(rej_tokens) > self.max_seq:
                rej_tokens = rej_tokens[:self.max_seq]
                rej_labels = rej_labels[:self.max_seq]
            elif len(rej_tokens) < self.max_seq:
                pad_len = self.max_seq - len(rej_tokens)
                rej_tokens += [self.tokenizer.pad_token_id] * pad_len
                rej_labels += [-100] * pad_len

            # Pad/truncate chosen (handled below), store rejected
            if len(tokens) > self.max_seq:
                tokens = tokens[:self.max_seq]
                labels = labels[:self.max_seq]
            elif len(tokens) < self.max_seq:
                pad_len = self.max_seq - len(tokens)
                tokens += [self.tokenizer.pad_token_id] * pad_len
                labels += [-100] * pad_len

            return {"input_ids": tokens, "labels": labels,
                    "rejected_input_ids": rej_tokens, "rejected_labels": rej_labels,
                    "is_dpo": True}
        else:
            tokens = [self.tokenizer.pad_token_id] * self.max_seq
            labels = [-100] * self.max_seq

        if len(tokens) > self.max_seq:
            tokens = tokens[:self.max_seq]
            labels = labels[:self.max_seq]
        elif len(tokens) < self.max_seq:
            pad_len = self.max_seq - len(tokens)
            tokens += [self.tokenizer.pad_token_id] * pad_len
            labels += [-100] * pad_len

        return {"input_ids": tokens, "labels": labels, "is_dpo": False}


def main():
    gc.collect(2)
    gc.freeze()
    gc.set_threshold(50_000, 500, 50)

    from accelerate import InitProcessGroupKwargs
    from datetime import timedelta
    pg_timeout = InitProcessGroupKwargs(timeout=timedelta(hours=1))
    accelerator = Accelerator(kwargs_handlers=[pg_timeout])
    set_seed(42)

    # ── Config ──
    model_path = os.environ.get("MODEL_PATH", "/home/spark/models/Huihui-Qwen3.5-35B-A3B-abliterated")
    delta_path = os.environ.get("RESUME_DELTA", "")
    sft_dir = os.environ.get("SFT_DIR", "/var/spark/isma/training/sft")
    cpt_data = os.environ.get("CPT_DATA", "/var/spark/isma/training/infra_soul_cpt.jsonl")
    general_dir = os.environ.get("GENERAL_DIR", "")
    output_dir = os.environ.get("OUTPUT_DIR", "/var/spark/models/taey-lora-v1")
    max_seq = int(os.environ.get("MAX_SEQ", "8192"))
    total_steps = int(os.environ.get("TOTAL_STEPS", "3000"))
    save_every = int(os.environ.get("SAVE_EVERY", "50"))
    session_limit = int(os.environ.get("SESSION_LIMIT", "250"))
    # Keystone layers configurable via env var (JSON array) or default
    keystone_env = os.environ.get("KEYSTONE_LAYERS", "")
    if keystone_env:
        keystone_layers = json.loads(keystone_env)
    else:
        keystone_layers = KEYSTONE_LAYERS

    # Per-expert gradient masking: freeze specific experts within batched [256, dim, dim] tensors
    frozen_experts_path = os.environ.get("FROZEN_EXPERTS", "")
    frozen_experts_by_layer = {}  # {layer_idx: set of expert_ids to freeze}
    if frozen_experts_path and os.path.exists(frozen_experts_path):
        with open(frozen_experts_path) as _fe_f:
            frozen_list = json.load(_fe_f)  # [[keystone_idx, expert_id], ...]
        # Build lookup: keystone_idx -> actual layer number -> set of frozen expert ids
        for k_idx, e_id in frozen_list:
            if k_idx < len(keystone_layers):
                layer_num = keystone_layers[k_idx]
                frozen_experts_by_layer.setdefault(layer_num, set()).add(e_id)
        if accelerator.is_main_process:
            total_frozen = sum(len(v) for v in frozen_experts_by_layer.values())
            total_trainable = len(keystone_layers) * 256 - total_frozen
            log.info(f"Gradient masking: {total_frozen} frozen experts, "
                     f"{total_trainable} trainable across {len(keystone_layers)} keystone layers")

    # Tiered learning rates
    lr_esft = float(os.environ.get("LR_ESFT", "2e-5"))       # Expert tensors + norms
    lr_lora = float(os.environ.get("LR_LORA", "3e-4"))       # LoRA adapters
    lr_router = float(os.environ.get("LR_ROUTER", "3e-5"))   # Router gates
    warmup_steps = int(os.environ.get("WARMUP_STEPS", "25"))

    if accelerator.is_main_process:
        log.info(f"=== PALIOS-TAEY v3: Hybrid LoRA + ESFT ===")
        log.info(f"Model: {model_path}")
        log.info(f"Keystone layers: {keystone_layers}")
        log.info(f"LR: esft={lr_esft}, lora={lr_lora}, router={lr_router}")
        log.info(f"Seq={max_seq}, session={session_limit}, save_every={save_every}")
        mem = torch.cuda.mem_get_info()
        log.info(f"UMA: free={mem[0]/1e9:.1f}GB total={mem[1]/1e9:.1f}GB")

    # ── Tokenizer ──
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Rank-split model loading ──
    # UMA constraint: 119GB system memory (GPU firmware reserves 9.5GB of 128.5GB UMA).
    # Loading 71GB model to CUDA creates 71GB page cache (mmap) + 71GB CUDA = 142GB → OOM.
    # Solution: rank 0 loads to CPU (zero-copy mmap from safetensors, no CUDA allocation).
    # FSDP sync_module_states broadcasts rank 0's CPU params → CUDA on all ranks during wrap.
    # Other ranks: device_map="meta" = zero memory.
    if accelerator.is_main_process:
        log.info("Rank 0: loading model to CPU (zero-copy mmap, FSDP handles GPU placement)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
        vm = psutil.virtual_memory()
        log.info(f"Rank 0: model loaded to CPU. RAM used={vm.used/1e9:.1f}GB free={vm.available/1e9:.1f}GB")
    else:
        log.info(f"Rank {accelerator.process_index}: loading model on meta device (zero memory)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
            device_map="meta",
        )

    model.config.use_cache = False
    model.config.output_router_logits = False  # Disabled — avoid aux_loss double-count
    # gradient checkpointing DISABLED — 11x slower (200s/step vs 17.5s/step)
    # model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # ── Apply LoRA to shared expert + attention ──
    if accelerator.is_main_process:
        log.info("Applying LoRA to shared_expert + attention projections...")

    from peft import LoraConfig, get_peft_model, TaskType
    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        lora_dropout=0.0,
        target_modules=[
            # Shared expert MLPs (standard Linear modules in all 40 layers)
            "shared_expert.gate_proj", "shared_expert.up_proj", "shared_expert.down_proj",
            # Attention projections — SDPA layers (every 4th layer)
            "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
            # DeltaNet linear attention projections (other 30 layers)
            "linear_attn.out_proj", "linear_attn.in_proj_qkv", "linear_attn.in_proj_z",
        ],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    # Cast ALL LoRA adapter weights to bf16 — FSDP requires uniform dtype
    # within each wrapped unit. PEFT creates fp32 adapters by default.
    for name, param in model.named_parameters():
        if 'lora_' in name.lower() and param.dtype == torch.float32:
            param.data = param.data.to(torch.bfloat16)

    if accelerator.is_main_process:
        model.print_trainable_parameters()

    # ── NCCL warm-up ──
    dist.barrier()
    if accelerator.is_main_process:
        log.info("NCCL connections established")

    # ── Checkpoint reload (rank 0 only, FSDP broadcasts) ──
    resume_step = 0
    if delta_path:
        adapter_file = os.path.join(delta_path, "adapter_model.safetensors")
        meta_file = os.path.join(delta_path, "trainer_meta.pt")
        if accelerator.is_main_process:
            # Universal resume: try trainable_weights.safetensors first (covers ALL configs)
            universal_file = os.path.join(delta_path, "trainable_weights.safetensors")
            if os.path.exists(universal_file):
                log.info(f"Rank 0: loading trainable weights from {delta_path}...")
                from safetensors.torch import load_file
                all_state = load_file(universal_file)
                lora_keys = {k: v for k, v in all_state.items() if 'lora_' in k.lower()}
                other_keys = {k: v for k, v in all_state.items() if 'lora_' not in k.lower()}
                if lora_keys:
                    from peft import set_peft_model_state_dict
                    set_peft_model_state_dict(model, lora_keys, adapter_name="default")
                    log.info(f"  LoRA: {len(lora_keys)} tensors via set_peft_model_state_dict")
                if other_keys:
                    model.load_state_dict(other_keys, strict=False)
                    log.info(f"  Non-LoRA: {len(other_keys)} tensors via load_state_dict")
                log.info(f"  Total: {len(all_state)} trainable tensors loaded")
                del all_state, lora_keys, other_keys
                gc.collect()
            elif os.path.exists(adapter_file):
                # Legacy fallback: separate per-type files
                log.info(f"Rank 0: loading legacy checkpoint from {delta_path}...")
                from safetensors.torch import load_file
                from peft import set_peft_model_state_dict
                adapter_state = load_file(adapter_file)
                set_peft_model_state_dict(model, adapter_state, adapter_name="default")
                log.info(f"  Legacy PEFT: {len(adapter_state)} tensors")
                del adapter_state; gc.collect()
                for fname, label in [("router_gates.safetensors", "router"),
                                      ("expert_weights.safetensors", "expert")]:
                    fpath = os.path.join(delta_path, fname)
                    if os.path.exists(fpath):
                        from safetensors.torch import load_file as lf
                        st = lf(fpath)
                        model.load_state_dict(st, strict=False)
                        log.info(f"  Legacy {label}: {len(st)} tensors")
                        del st; gc.collect()
            if os.path.exists(meta_file):
                meta = torch.load(meta_file, map_location="cpu", weights_only=False)
                resume_step = meta.get("step", 0)
                log.info(f"Resuming from step {resume_step}")
                del meta
            else:
                # Fallback: parse step from directory name (checkpoint-300 → 300)
                import re
                m = re.search(r'checkpoint-(\d+)', delta_path)
                if m:
                    resume_step = int(m.group(1))
                    log.info(f"Resuming from step {resume_step} (parsed from directory name)")

    dist.barrier()
    resume_tensor = torch.tensor([resume_step], dtype=torch.long, device="cuda")
    dist.broadcast(resume_tensor, src=0)
    resume_step = resume_tensor.item()

    # ── Configurable freeze/unfreeze based on FREEZE_CONFIG env var ──
    # Config A (v3_mixed): LoRA(shared+attn) + router + experts [17,28] — current default
    # Config B (chatgpt_v5): NO LoRA — experts only [8,9,21,25,28,38], freeze shared/router/attn
    # Config C (grok_inverted): experts + attn/norms keystone only, freeze shared/router
    freeze_config = os.environ.get("FREEZE_CONFIG", "A")

    router_params = []
    expert_params = []

    if freeze_config == "VANILLA":
        # Config VANILLA — textbook DPO sanity check (2026-04-20).
        # No custom freeze/unfreeze. PEFT's get_peft_model() already set requires_grad:
        # LoRA adapter params = trainable, everything else (base weights, experts, router) = frozen.
        # Purpose: answer the first-principles question "does standard DPO work on this
        # Qwen3.5-35B-A3B MoE architecture at all?" without any of our custom surface masks
        # obscuring the signal. If a vanilla run on abliterated base with a standard preference
        # dataset moves the model in the textbook direction, our pipeline is over-engineered. If
        # vanilla ALSO fails, there's something architectural/scale-dependent about DPO on this MoE.
        if accelerator.is_main_process:
            lora_count = sum(p.numel() for n, p in model.named_parameters()
                             if 'lora_' in n.lower() and p.requires_grad)
            frozen_count = sum(p.numel() for p in model.parameters() if not p.requires_grad)
            log.info(f"Config VANILLA: LoRA-only standard DPO. "
                     f"{lora_count/1e6:.1f}M trainable LoRA params, {frozen_count/1e9:.2f}B frozen. "
                     f"No custom masks, no keystone unfreezing, no expert gradient masking.")

    elif freeze_config == "B":
        # Config B + MATH/COLLAPSE protection (Infra/Gemini consultation):
        # 1. Freeze ALL params first
        # 2. Unfreeze ONLY expert tensors in keystone layers
        # 3. Freeze ALL routers (prevents routing starvation away from MATH)
        # All freezing happens BEFORE FSDP wrapping — FSDP respects requires_grad
        import re as _re
        for name, param in model.named_parameters():
            param.requires_grad = False
        # Unfreeze keystone expert tensors only
        for name, param in model.named_parameters():
            if 'mlp.experts.' in name and 'shared_expert' not in name:
                m = _re.search(r'layers\.(\d+)', name)
                if m and int(m.group(1)) in keystone_layers:
                    param.requires_grad = True
                    expert_params.append(param)
        # Explicitly freeze all routers (prevent routing starvation)
        for name, param in model.named_parameters():
            if 'mlp.gate.' in name and 'shared_expert_gate' not in name:
                param.requires_grad = False
        if accelerator.is_main_process:
            trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            frozen_count = sum(p.numel() for p in model.parameters() if not p.requires_grad)
            log.info(f"Config B + substrate protection: {trainable_count/1e6:.1f}M trainable, "
                     f"{frozen_count/1e9:.2f}B frozen, routers FROZEN")

    elif freeze_config == "C":
        # Grok: experts + attn/norms in keystone layers only, freeze shared/router
        for name, param in model.named_parameters():
            if 'lora_' in name.lower():
                param.requires_grad = False
        for name, param in model.named_parameters():
            clean = name.replace("_fsdp_wrapped_module.", "").replace("module.", "")
            for kl in keystone_layers:
                if f'layers.{kl}.mlp.experts.' in clean and not param.requires_grad:
                    param.requires_grad = True
                    expert_params.append(param)
                if f'layers.{kl}.' in clean and ('attn' in clean or 'norm' in clean or 'layernorm' in clean):
                    if not param.requires_grad:
                        param.requires_grad = True
        if accelerator.is_main_process:
            log.info(f"Config C (Grok): experts + attn/norms keystone-only, shared/router FROZEN")

    elif freeze_config == "A2":
        # Config A2 (2026-04-20): like Config A, but attention LoRA is RESTRICTED to keystone layers only.
        # Per length_mechanics_v1 diagnostic: Config A's attention LoRA on all 40 layers leaks into
        # infra_cross_system regardless of training-data content (same infra_cross_system regression
        # observed on religion and on content-neutral length-preference data). Expert gradient masking
        # (v4.x frozen_experts) doesn't touch attention LoRA — orthogonal surfaces.
        #
        # A2 keeps LoRA on shared_expert (all 40 layers) + attention LoRA (keystones only) +
        # keystone experts + router + norms. Roughly 6/40 = 85% reduction in attention-LoRA surface.
        import re as _re_a2
        # First: apply Config A's standard trainable marking (LoRA + router + keystone experts)
        for name, param in model.named_parameters():
            clean = name.replace("_fsdp_wrapped_module.", "").replace("module.", "")
            if _is_trainable(clean, keystone_layers):
                if not param.requires_grad:
                    param.requires_grad = True
                    if 'mlp.gate.weight' in clean or 'shared_expert_gate' in clean:
                        router_params.append(param)
                    elif 'mlp.experts.' in clean:
                        expert_params.append(param)
        # Then: freeze attention LoRA on non-keystone layers.
        attn_lora_frozen = 0
        for name, param in model.named_parameters():
            if 'lora_' not in name.lower():
                continue
            clean = name.replace("_fsdp_wrapped_module.", "").replace("module.", "")
            # Only attention LoRA (both SDPA self_attn and DeltaNet linear_attn)
            if 'self_attn' not in clean and 'linear_attn' not in clean:
                continue
            m = _re_a2.search(r'layers\.(\d+)\.', clean)
            if not m:
                continue
            layer_num = int(m.group(1))
            if layer_num not in keystone_layers and param.requires_grad:
                param.requires_grad = False
                attn_lora_frozen += 1
        if accelerator.is_main_process:
            log.info(f"Config A2 (keystone-only attention LoRA): {attn_lora_frozen} non-keystone "
                     f"attention LoRA params frozen. Keystones: {keystone_layers}")

    elif freeze_config == "A4":
        # Config A4 = Claude Opus Option E (projection-type restriction, not layer-subset) — 2026-04-20.
        # Claude's mechanism diagnosis: token corruption (NVLink-CC2C, 055,000,000) is q/k projection
        # misalignment signature, not o_proj damage, not MLP damage. q/k jointly produce attention
        # distribution. Low-rank updates on q/k at r=64 can shift attention routing in arbitrary
        # directions that DPO has no regularizer against. For policy surfaces (direct/hedge,
        # concise/verbose, sycophancy-resist) you don't need to change WHERE attention points —
        # you need to change how extracted info is combined into the residual stream. That's
        # o_proj territory (the read-out, not the route).
        #
        # A4 keeps o_proj LoRA trainable on ALL 40 layers + shared_expert LoRA on all 40 layers +
        # keystone experts + router. Freezes q_proj/k_proj/v_proj LoRA (SDPA) and in_proj_qkv/in_proj_z
        # LoRA (DeltaNet) everywhere. Attention routing is bit-exact preserved → multi-hop topology
        # reasoning (infra_cross_system) cannot be scrambled.
        #
        # Trade vs A2: A4 keeps MORE surface (all 40 layers of o_proj vs keystone-only attention)
        # but RESTRICTS projection types (o_proj only vs q/k/v/o). If A2 fixed the leak via layer
        # restriction, A4 tests whether projection-type restriction alone (without layer restriction)
        # also fixes it. Different architectural axis.
        import re as _re_a4
        # First: apply Config A's standard trainable marking
        for name, param in model.named_parameters():
            clean = name.replace("_fsdp_wrapped_module.", "").replace("module.", "")
            if _is_trainable(clean, keystone_layers):
                if not param.requires_grad:
                    param.requires_grad = True
                    if 'mlp.gate.weight' in clean or 'shared_expert_gate' in clean:
                        router_params.append(param)
                    elif 'mlp.experts.' in clean:
                        expert_params.append(param)
        # Then: freeze q/k/v attention LoRA on ALL layers (keep o_proj trainable).
        # DeltaNet analog: freeze in_proj_qkv + in_proj_z (the "query/key/value" analogs),
        # keep out_proj trainable (the read-out analog).
        qkv_lora_frozen = 0
        oproj_kept = 0
        for name, param in model.named_parameters():
            if 'lora_' not in name.lower():
                continue
            clean = name.replace("_fsdp_wrapped_module.", "").replace("module.", "")
            # q/k/v on SDPA MHA layers
            if ('self_attn.q_proj' in clean or 'self_attn.k_proj' in clean or 'self_attn.v_proj' in clean):
                if param.requires_grad:
                    param.requires_grad = False
                    qkv_lora_frozen += 1
                continue
            # in_proj_qkv + in_proj_z on DeltaNet layers (q/k/v analogs)
            if 'linear_attn.in_proj_qkv' in clean or 'linear_attn.in_proj_z' in clean:
                if param.requires_grad:
                    param.requires_grad = False
                    qkv_lora_frozen += 1
                continue
            # o_proj (SDPA) and out_proj (DeltaNet) stay trainable — count for logging
            if 'self_attn.o_proj' in clean or 'linear_attn.out_proj' in clean:
                if param.requires_grad:
                    oproj_kept += 1
        if accelerator.is_main_process:
            log.info(f"Config A4 (Claude's o_proj-only, all 40 layers): "
                     f"{qkv_lora_frozen} q/k/v LoRA params frozen, "
                     f"{oproj_kept} o_proj/out_proj LoRA params kept trainable across 40 layers. "
                     f"shared_expert LoRA unchanged.")

    elif freeze_config == "A3":
        # Config A3 = Option E (Strict Keystone Block Isolation) — 2026-04-20.
        # Chat consensus (5 platforms, 02:45 consult): extend A2 by also restricting shared_expert
        # LoRA to keystones. Gemini's mechanism diagnosis: attention AND shared_expert LoRA on
        # non-keystone layers both rotate hidden states that feed into frozen routed experts
        # (v4.1 mask freezes experts in 34 of 40 layers). Frozen experts are calibrated to
        # baseline activation geometry; OOD input vectors cause misfires on sparse reasoning
        # paths (infra_cross_system, induction heads for hardware-topology exact-match circuits).
        #
        # A3 keeps ALL LoRA (attention + shared_expert) restricted to keystones + keystone
        # experts + router. Every non-keystone layer becomes a mathematically frozen identity
        # function relative to baseline. Expected: +12pp religion_honest holds (6 keystones
        # still overparameterized per Gemini), infra_cross_system returns to 4/4, token
        # corruption eliminated.
        import re as _re_a3
        # First: apply Config A's standard trainable marking
        for name, param in model.named_parameters():
            clean = name.replace("_fsdp_wrapped_module.", "").replace("module.", "")
            if _is_trainable(clean, keystone_layers):
                if not param.requires_grad:
                    param.requires_grad = True
                    if 'mlp.gate.weight' in clean or 'shared_expert_gate' in clean:
                        router_params.append(param)
                    elif 'mlp.experts.' in clean:
                        expert_params.append(param)
        # Then: freeze attention LoRA AND shared_expert LoRA on non-keystone layers.
        lora_frozen = {'attn': 0, 'shared_expert': 0}
        for name, param in model.named_parameters():
            if 'lora_' not in name.lower():
                continue
            clean = name.replace("_fsdp_wrapped_module.", "").replace("module.", "")
            m = _re_a3.search(r'layers\.(\d+)\.', clean)
            if not m:
                continue
            layer_num = int(m.group(1))
            if layer_num in keystone_layers:
                continue
            if 'self_attn' in clean or 'linear_attn' in clean:
                if param.requires_grad:
                    param.requires_grad = False
                    lora_frozen['attn'] += 1
            elif 'shared_expert' in clean:
                if param.requires_grad:
                    param.requires_grad = False
                    lora_frozen['shared_expert'] += 1
        if accelerator.is_main_process:
            log.info(f"Config A3 (Option E - Strict Keystone Block Isolation): "
                     f"{lora_frozen['attn']} non-keystone attention LoRA frozen, "
                     f"{lora_frozen['shared_expert']} non-keystone shared_expert LoRA frozen. "
                     f"Keystones: {keystone_layers}")

    else:
        # Config A (default/v3_mixed): LoRA + router + experts — current behavior
        for name, param in model.named_parameters():
            clean = name.replace("_fsdp_wrapped_module.", "").replace("module.", "")
            if _is_trainable(clean, keystone_layers):
                if not param.requires_grad:
                    param.requires_grad = True
                    if 'mlp.gate.weight' in clean or 'shared_expert_gate' in clean:
                        router_params.append(param)
                    elif 'mlp.experts.' in clean:
                        expert_params.append(param)
        if accelerator.is_main_process:
            log.info(f"Config A (V3 Mixed): LoRA + router + experts")

    router_n = sum(p.numel() for p in router_params)
    expert_n = sum(p.numel() for p in expert_params)
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    if accelerator.is_main_process:
        log.info(f"Trainable: {total_trainable/1e6:.1f}M "
                 f"(router={router_n/1e6:.1f}M, experts={expert_n/1e6:.1f}M), "
                 f"{frozen/1e9:.2f}B frozen")

    # ── Safety check: shared_expert base weights must be frozen ──
    if accelerator.is_main_process:
        for name, param in model.named_parameters():
            if "shared_expert" in name and "lora" not in name.lower() and param.requires_grad:
                log.warning(f"SAFETY: shared_expert base weight unfrozen: {name}")

    # ── Pre-FSDP forward check (rank 0 only — has real weights) ──
    if accelerator.is_main_process:
        log.info("Running pre-FSDP forward sanity check on rank 0...")
        try:
            model.eval()
            dummy_ids = torch.tensor([[1, 2, 3, 4, 5]], device="cuda")
            dummy_labels = torch.tensor([[1, 2, 3, 4, 5]], device="cuda")
            with torch.no_grad():
                out = model(input_ids=dummy_ids, labels=dummy_labels)
            finite = torch.isfinite(out.loss).item()
            log.info(f"Pre-FSDP forward: loss={out.loss.item():.4f}, finite={finite}")
            del out
            model.train()
        except Exception as e:
            log.info(f"Pre-FSDP forward FAILED: {e}")
            model.train()

    # ── FSDP wrapping ──
    import functools
    from peft.utils.other import fsdp_auto_wrap_policy
    from torch.distributed.fsdp.wrap import _or_policy, transformer_auto_wrap_policy
    # GEMINI FIX: Use dynamic class from loaded model instead of static import.
    # Abliterated base loads via trust_remote_code → class lives in transformers_modules.* namespace.
    # Static import references transformers.models.* → different Python class → isinstance() fails → zero layers wrapped → monolithic 70GB flat_param → OOM.
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeDecoderLayer as _QwenLayerStatic
    # Walk through .model attributes until we find the one with .layers
    # (PEFT wrapping + causal LM wrapping means the .layers-bearing module may be 2-3 levels deep)
    _inner = model
    _depth = 0
    while not hasattr(_inner, 'layers') and hasattr(_inner, 'model') and _depth < 5:
        _inner = _inner.model
        _depth += 1
    if not hasattr(_inner, 'layers'):
        raise RuntimeError(f"Could not locate .layers attribute after {_depth} descents. Last type: {type(_inner).__name__}")
    layer_cls = type(_inner.layers[0])
    if accelerator.is_main_process:
        log.info(f"FSDP layer_cls resolved: {layer_cls.__module__}.{layer_cls.__name__}")
        log.info(f"Class identity vs static import: {layer_cls is _QwenLayerStatic}")
    lora_policy = fsdp_auto_wrap_policy(model)
    layer_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={layer_cls},
    )
    combined_policy = functools.partial(_or_policy, policies=[lora_policy, layer_policy])
    accelerator.state.fsdp_plugin.auto_wrap_policy = combined_policy
    if accelerator.is_main_process:
        log.info(f"FSDP wrap policy: PEFT LoRA units OR {layer_cls.__name__} units")

    # ── Dataset ──
    # DPO precompute path: if DPO_DATA points at a precomputed-ref jsonl,
    # use DPOPairsDataset so the main loop's batch has {chosen,rejected}_{input_ids,labels}
    # + ref_{chosen,rejected}_logprob. Otherwise fall back to CombinedSFTDataset.
    dpo_data_path_main = os.environ.get("DPO_DATA", "")
    if dpo_data_path_main and os.path.isfile(dpo_data_path_main):
        dataset = DPOPairsDataset(dpo_data_path_main, max_seq)
        if accelerator.is_main_process:
            log.info(f"Main dataset: DPOPairsDataset from {dpo_data_path_main} ({len(dataset)} pairs)")

        def collate_fn(batch):
            # batch_size=1; each item is a dict of 1D tensors + 0D logprob tensors.
            b = batch[0]
            return {
                "chosen_input_ids":      b["chosen_input_ids"].unsqueeze(0),
                "chosen_labels":         b["chosen_labels"].unsqueeze(0),
                "rejected_input_ids":    b["rejected_input_ids"].unsqueeze(0),
                "rejected_labels":       b["rejected_labels"].unsqueeze(0),
                "ref_chosen_logprob":    b["ref_chosen_logprob"],
                "ref_rejected_logprob":  b["ref_rejected_logprob"],
            }
    else:
        dataset = CombinedSFTDataset(sft_dir, cpt_data, general_dir, tokenizer, max_seq)

        def collate_fn(batch):
            input_ids = torch.stack([torch.tensor(b["input_ids"], dtype=torch.long) for b in batch])
            labels = torch.stack([torch.tensor(b["labels"], dtype=torch.long) for b in batch])
            result = {"input_ids": input_ids, "labels": labels}
            # DPO: include rejected if any item in batch is DPO
            if any(b.get("is_dpo", False) for b in batch):
                rej_ids = torch.stack([torch.tensor(b.get("rejected_input_ids", b["input_ids"]), dtype=torch.long) for b in batch])
                rej_labels = torch.stack([torch.tensor(b.get("rejected_labels", b["labels"]), dtype=torch.long) for b in batch])
                result["rejected_input_ids"] = rej_ids
                result["rejected_labels"] = rej_labels
                result["is_dpo"] = True
            else:
                result["is_dpo"] = False
            return result

    sampler = DistributedSampler(dataset, num_replicas=accelerator.num_processes,
                                 rank=accelerator.process_index, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, collate_fn=collate_fn,
                            pin_memory=False, num_workers=0)

    # ── Override FSDP mixed precision: param stays bf16, only reduce in fp32 ──
    # Accelerate's mixed_precision=bf16 upcasts ALL params to fp32 during forward → OOM.
    # We override with explicit policy: keep params bf16, reduce gradients in fp32.
    # This fixes attention overflow at 8K+ without doubling memory.
    from torch.distributed.fsdp import MixedPrecision as FSDPMixedPrecision
    mp_policy = FSDPMixedPrecision(
        param_dtype=torch.bfloat16,    # Keep params in bf16 (no upcasting)
        reduce_dtype=torch.float32,     # Gradient reductions in fp32 (fixes NaN)
        buffer_dtype=torch.bfloat16,
    )
    accelerator.state.fsdp_plugin.mixed_precision_policy = mp_policy
    if accelerator.is_main_process:
        log.info(f"FSDP MixedPrecision: param=bf16, reduce=fp32, buffer=bf16")

    # ── FSDP prepare ──
    if accelerator.is_main_process:
        log.info("Calling accelerator.prepare()...")
    model, dataloader = accelerator.prepare(model, dataloader)
    gc.collect()

    # ── Fix 2: Broadcast non-persistent buffers (MoE gate correction bias) ──
    # Perplexity DR Root Cause 2: sync_module_states only broadcasts Parameters
    # and persistent buffers, NOT non-persistent buffers like e_score_correction_bias
    buf_count = 0
    for name, buf in model.named_buffers():
        if buf is not None and buf.is_cuda:
            dist.broadcast(buf.data, src=0)
            buf_count += 1
    dist.barrier()
    if accelerator.is_main_process:
        log.info(f"Broadcast {buf_count} buffers from rank 0")

    # ── Per-rank shard reload (post-FSDP) ──
    if delta_path:
        rank = accelerator.process_index
        shard_file = os.path.join(delta_path, f"model_rank{rank}.pt")
        if not os.path.exists(shard_file):
            shard_file = os.path.join(delta_path, f"shard_{rank}.pt")
        if os.path.exists(shard_file):
            log.info(f"Rank {rank}: loading shard from {shard_file}")
            shard = torch.load(shard_file, map_location="cpu", mmap=True, weights_only=False)
            clean_to_param = {}
            for name, param in model.named_parameters():
                clean_to_param[_clean_fsdp_name(name)] = param
            loaded, skipped = 0, 0
            with torch.no_grad():
                for saved_key, tensor in shard.items():
                    clean_key = _clean_fsdp_name(saved_key)
                    if clean_key in clean_to_param:
                        live = clean_to_param[clean_key]
                        if live.shape == tensor.shape:
                            live.data.copy_(tensor.to(dtype=live.dtype, device=live.device))
                            loaded += 1
                        else:
                            skipped += 1
                    else:
                        skipped += 1
            del shard
            gc.collect()
            torch.cuda.empty_cache()
            log.info(f"Rank {rank}: shard applied ({loaded} loaded, {skipped} skipped)")
        dist.barrier()

    # ── Freeze/unfreeze was moved BEFORE accelerator.prepare() ──
    # (see above — Perplexity Root Cause 1 fix)
    mem = torch.cuda.mem_get_info()
    if accelerator.is_main_process:
        log.info(f"  CUDA free: {mem[0]/1e9:.1f}GB")

    # ── Per-expert gradient masking via flat_param post_accumulate hooks ──
    # ChatGPT-consulted solution (fsdp_expert_masking.py). Works on FSDP FULL_SHARD
    # + use_orig_params=True by hooking flat_param, not orig-param views.
    # Hooks fire BEFORE reduce-scatter on UNSHARDED 1D grad.
    mask_specs = {}
    if frozen_experts_by_layer:
        try:
            from fsdp_expert_masking import install_fsdp_expert_gradient_mask
            # frozen_experts_by_layer is Dict[int, Set[int]], helper wants Dict[int, List[int]]
            _mask_input = {k: sorted(v) for k, v in frozen_experts_by_layer.items()}
            mask_specs = install_fsdp_expert_gradient_mask(
                model,
                _mask_input,
                num_experts=256,
                log_fn=log.info if accelerator.is_main_process else (lambda _m: None),
                log_rank0_only=True,
                max_verbose_steps=3,
            )
            if accelerator.is_main_process:
                log.info(f"Expert grad masking installed on {len(mask_specs)} layers")
        except Exception as _e:
            if accelerator.is_main_process:
                log.error(f"FAILED to install expert grad mask: {_e}")
                import traceback
                log.error(traceback.format_exc())
            raise

    # ── Optimizer: 3 param groups with tiered LR ──
    router_ids = {id(p) for p in router_params}
    expert_ids = {id(p) for p in expert_params}
    lora_group = [p for p in model.parameters()
                  if p.requires_grad and id(p) not in router_ids and id(p) not in expert_ids]
    router_group = list(router_params)
    expert_group = list(expert_params)

    # Adafactor for all params — factored row/column sums instead of full per-param momentum.
    # Reduces expert optimizer from ~3.2GB to <150MB, preventing UMA page cache eviction.
    # Pipeline v4-v7 runs used Adafactor successfully (2.8GB leak over full session vs
    # AdamW's 3.2GB immediate allocation that triggers page cache thrashing at step 170).
    from transformers.optimization import Adafactor

    optimizer = Adafactor(
        [{"params": lora_group, "lr": lr_lora},
         {"params": router_group, "lr": lr_router},
         {"params": expert_group, "lr": lr_esft}],
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
        weight_decay=0.01,
        clip_threshold=1.0,
    )

    if accelerator.is_main_process:
        log.info(f"Optimizer: Adafactor(LoRA={sum(p.numel() for p in lora_group)/1e6:.1f}M @ lr={lr_lora}, "
                 f"Router={sum(p.numel() for p in router_group)/1e6:.1f}M @ lr={lr_router}, "
                 f"Expert={sum(p.numel() for p in expert_group)/1e6:.1f}M @ lr={lr_esft})")

    # ── LR scheduler: cosine decay to 10% floor ──
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    optimizer, lr_scheduler = accelerator.prepare(optimizer, lr_scheduler)

    if resume_step > 0:
        inner_sched = lr_scheduler.scheduler if hasattr(lr_scheduler, 'scheduler') else lr_scheduler
        inner_sched.last_epoch = resume_step
        inner_sched._step_count = resume_step + 1
        for i, group in enumerate(optimizer.param_groups):
            group['lr'] = inner_sched.base_lrs[i] * inner_sched.lr_lambdas[i](resume_step)
        if accelerator.is_main_process:
            log.info(f"Scheduler fast-forwarded to step {resume_step}")
    dist.barrier()

    if accelerator.is_main_process:
        mem = torch.cuda.mem_get_info()
        log.info(f"Ready. CUDA free={mem[0]/1e9:.1f}GB")

    # ── Post-FSDP diagnostic forward (OOM-safe, skippable) ──
    # Full FSDP all-gather of 35B even with 5-token input needs ~30GB transient.
    # Under low UMA headroom (merged-base resume) this OOMs. Training math is
    # unaffected — diagnostic is sanity-check only.
    if os.environ.get("DPO_SKIP_POSTFSDP_DIAG", "0") == "1":
        log.info(f"Rank {accelerator.process_index}: post-FSDP diagnostic SKIPPED (DPO_SKIP_POSTFSDP_DIAG=1)")
    else:
        try:
            torch.cuda.empty_cache()
            log.info(f"Rank {accelerator.process_index}: running post-FSDP diagnostic forward...")
            model.eval()
            with torch.no_grad():
                diag_ids = torch.tensor([[1, 2, 3, 4, 5]], device=accelerator.device)
                diag_labels = torch.tensor([[1, 2, 3, 4, 5]], device=accelerator.device)
                diag_out = model(input_ids=diag_ids, labels=diag_labels)
                diag_loss = diag_out.loss
                diag_logits = diag_out.logits
                log.info(f"Rank {accelerator.process_index}: POST-FSDP diag: "
                         f"loss={diag_loss.item():.4f} "
                         f"loss_finite={torch.isfinite(diag_loss).item()} "
                         f"logits_shape={diag_logits.shape}")
                del diag_out, diag_logits
            torch.cuda.empty_cache()
            model.train()
            log.info(f"Rank {accelerator.process_index}: running train-mode (no_grad) diagnostic...")
            with torch.no_grad():
                diag_ids2 = torch.tensor([[1, 2, 3, 4, 5]], device=accelerator.device)
                diag_labels2 = torch.tensor([[1, 2, 3, 4, 5]], device=accelerator.device)
                diag_out2 = model(input_ids=diag_ids2, labels=diag_labels2)
                log.info(f"Rank {accelerator.process_index}: TRAIN-MODE diag: "
                         f"loss={diag_out2.loss.item():.4f}")
                del diag_out2
            torch.cuda.empty_cache()
        except Exception as e:
            log.warning(f"Rank {accelerator.process_index}: post-FSDP diagnostic FAILED: {e} — continuing to training")
    dist.barrier()

    # ── DPO dataloader (separate, for periodic DPO steps) ──
    dpo_dir = os.environ.get("DPO_DIR", "")
    dpo_dataloader = None
    dpo_iter = None
    dpo_interval = int(os.environ.get("DPO_INTERVAL", "10"))  # DPO every N steps
    dpo_weight = float(os.environ.get("DPO_WEIGHT", "0.1"))   # DPO loss weight

    if dpo_dir and os.path.isdir(dpo_dir):
        dpo_dataset = CombinedSFTDataset(dpo_dir, "", "", tokenizer, max_seq)
        dpo_sampler = DistributedSampler(dpo_dataset, num_replicas=accelerator.num_processes,
                                          rank=accelerator.process_index, shuffle=True)
        dpo_dataloader = DataLoader(dpo_dataset, batch_size=1, sampler=dpo_sampler,
                                     collate_fn=collate_fn, pin_memory=False, num_workers=0)
        dpo_iter = iter(dpo_dataloader)
        if accelerator.is_main_process:
            log.info(f"DPO dataloader: {len(dpo_dataset)} items, interval={dpo_interval}, weight={dpo_weight}")

    # ── Training loop ──
    os.makedirs(output_dir, exist_ok=True)
    model.train()
    global_step = resume_step

    # DPO-specific hyperparameters
    dpo_beta = float(os.environ.get("BETA", "0.05"))
    dpo_ratio_abort_threshold = float(os.environ.get("DPO_ABORT_RATIO_MAX", "10.0"))
    consecutive_high_ratio_steps = 0
    running_scale_sum = 0.0
    running_scale_steps = 0

    def _get_batch_logps(logits, labels):
        """Sum of log-probs of response tokens (labels != -100)."""
        shift_logits = logits[:-1, :].contiguous()
        shift_labels = labels[1:].contiguous()
        loss_mask = (shift_labels != -100)
        safe_labels = shift_labels.clone()
        safe_labels[~loss_mask] = 0
        log_probs = F.log_softmax(shift_logits.float(), dim=-1)
        per_token_logps = log_probs.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
        return (per_token_logps * loss_mask).sum()

    if accelerator.is_main_process:
        log.info(f"Starting DPO: steps {resume_step}→{total_steps}, {accelerator.num_processes} nodes")
        log.info(f"DPO config: beta={dpo_beta}, LR_ESFT={lr_esft}, LR_LORA={lr_lora}, router frozen")

    for epoch in range(100):
        sampler.set_epoch(epoch)
        for batch in dataloader:
            if global_step >= total_steps:
                break

            # DPO pair: batch has chosen_input_ids/labels, rejected_input_ids/labels, ref_chosen_logprob, ref_rejected_logprob
            cho_ids = batch["chosen_input_ids"].to(accelerator.device)
            cho_labels = batch["chosen_labels"].to(accelerator.device).squeeze(0)
            rej_ids = batch["rejected_input_ids"].to(accelerator.device)
            rej_labels = batch["rejected_labels"].to(accelerator.device).squeeze(0)
            ref_cho_lp = batch["ref_chosen_logprob"].to(accelerator.device).item()
            ref_rej_lp = batch["ref_rejected_logprob"].to(accelerator.device).item()

            # === DETACHED FORWARD: compute DPO scaling factor under no_grad ===
            model.eval()
            with torch.no_grad():
                pi_cho_logits = model(input_ids=cho_ids).logits.squeeze(0)
                pi_cho_lp_nograd = _get_batch_logps(pi_cho_logits, cho_labels)
                del pi_cho_logits

                pi_rej_logits = model(input_ids=rej_ids).logits.squeeze(0)
                pi_rej_lp_nograd = _get_batch_logps(pi_rej_logits, rej_labels)
                del pi_rej_logits

                ratio = dpo_beta * ((pi_cho_lp_nograd.item() - ref_cho_lp) - (pi_rej_lp_nograd.item() - ref_rej_lp))
                scale = float(torch.sigmoid(torch.tensor(-ratio)).item())

            # Abort check: |ratio| > threshold for 3 consecutive steps = routing collapse
            if abs(ratio) > dpo_ratio_abort_threshold:
                consecutive_high_ratio_steps += 1
                if consecutive_high_ratio_steps >= 3:
                    if accelerator.is_main_process:
                        log.error(f"DPO ABORT: |ratio|={abs(ratio):.3f} > {dpo_ratio_abort_threshold} for 3 steps — routing collapse suspected")
                    accelerator.wait_for_everyone()
                    sys.exit(1)
            else:
                consecutive_high_ratio_steps = 0

            # Track scale (learning signal health)
            running_scale_sum += scale
            running_scale_steps += 1

            # === ISOLATED CHOSEN PASS (no_sync - accumulate locally, no reduce-scatter) ===
            model.train()
            with model.no_sync():
                out_w = model(input_ids=cho_ids)
                lp_w = _get_batch_logps(out_w.logits.squeeze(0), cho_labels)
                loss_w = -(dpo_beta * scale * lp_w)
                del out_w

                # NaN check on chosen
                nan_flag = torch.zeros(1, device=accelerator.device)
                if torch.isnan(loss_w) or torch.isinf(loss_w):
                    nan_flag.fill_(1.0)
                dist.all_reduce(nan_flag, op=dist.ReduceOp.MAX)
                if nan_flag.item() > 0:
                    if accelerator.is_main_process:
                        log.warning(f"[step {global_step+1}] NaN/Inf on chosen — skipping step")
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    continue

                loss_w.backward()  # Accumulates locally inside no_sync

            # === ISOLATED REJECTED PASS (triggers FSDP reduce-scatter) ===
            out_l = model(input_ids=rej_ids)
            lp_l = _get_batch_logps(out_l.logits.squeeze(0), rej_labels)
            loss_l = +(dpo_beta * scale * lp_l)
            del out_l

            nan_flag = torch.zeros(1, device=accelerator.device)
            if torch.isnan(loss_l) or torch.isinf(loss_l):
                nan_flag.fill_(1.0)
            dist.all_reduce(nan_flag, op=dist.ReduceOp.MAX)
            if nan_flag.item() > 0:
                if accelerator.is_main_process:
                    log.warning(f"[step {global_step+1}] NaN/Inf on rejected — skipping step")
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                continue

            loss_l.backward()  # Triggers reduce-scatter for whole batch

            # Scalar loss for logging (policy-ref log-ratio margin for chosen-rejected)
            loss = -torch.nn.functional.logsigmoid(torch.tensor(ratio).to(accelerator.device))

            # Expert masking hook fires automatically on each backward
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            # Log extended DPO metrics
            if accelerator.is_main_process:
                reward_margin = (pi_cho_lp_nograd.item() - ref_cho_lp) - (pi_rej_lp_nograd.item() - ref_rej_lp)
                avg_scale = running_scale_sum / max(1, running_scale_steps)
                log.info(
                    f"[dpo step {global_step}] scale={scale:.4f} avg_scale={avg_scale:.4f} "
                    f"reward_margin={reward_margin:+.4f} "
                    f"pi_cho={pi_cho_lp_nograd.item():.2f} ref_cho={ref_cho_lp:.2f} "
                    f"pi_rej={pi_rej_lp_nograd.item():.2f} ref_rej={ref_rej_lp:.2f} "
                    f"ratio={ratio:+.4f}"
                )

            # Abort on no learning signal: mean scale > 0.9 for 10 steps
            if running_scale_steps >= 10 and (running_scale_sum / running_scale_steps) > 0.9:
                if accelerator.is_main_process:
                    log.error(
                        f"DPO ABORT: avg scale={running_scale_sum/running_scale_steps:.3f} > 0.9 for {running_scale_steps} steps — no learning signal"
                    )
                accelerator.wait_for_everyone()
                sys.exit(1)

            if accelerator.is_main_process and global_step % 10 == 0:
                mem = torch.cuda.mem_get_info()
                vm = psutil.virtual_memory()
                stats = torch.cuda.memory_stats()
                alloc = stats["allocated_bytes.all.current"]
                reserved = stats["reserved_bytes.all.current"]
                frag = (reserved - alloc) / reserved if reserved > 0 else 0
                log.info(f"[step {global_step}] loss={loss.item():.4f} "
                         f"lr={lr_scheduler.get_last_lr()[0]:.2e} free={mem[0]/1e9:.1f}GB "
                         f"frag={frag:.1%}")

            if global_step == resume_step + 1 and accelerator.is_main_process:
                torch.cuda.synchronize()
                free_b, total_b = torch.cuda.mem_get_info()
                alloc_b = torch.cuda.memory_allocated()
                param_b = sum(p.numel() * p.element_size() for p in model.parameters())
                grad_b = sum(p.grad.numel() * p.grad.element_size() for p in model.parameters() if p.grad is not None)
                optim_b = sum(v.numel() * v.element_size() for s in optimizer.state.values() for v in s.values() if torch.is_tensor(v))
                log.info(f"FIRST STEP: free={free_b/1e9:.1f}GB alloc={alloc_b/1e9:.1f}GB")
                log.info(f"  params={param_b/1e9:.1f}GB grads={grad_b/1e9:.1f}GB optim={optim_b/1e9:.1f}GB")

            # Periodic save (relative to session start, not global step)
            saved_this_step = False
            steps_this_session = global_step - resume_step
            if steps_this_session > 0 and steps_this_session % save_every == 0:
                _save_checkpoint(model, optimizer, lr_scheduler, tokenizer,
                                output_dir, global_step, epoch, keystone_layers, accelerator)
                saved_this_step = True

            # Session limit: clean exit
            steps_this_session = global_step - resume_step
            if steps_this_session >= session_limit:
                accelerator.wait_for_everyone()
                gc.collect()
                if not saved_this_step:
                    _free_for_save(model, optimizer)
                    _save_checkpoint(model, None, lr_scheduler, tokenizer,
                                    output_dir, global_step, epoch, keystone_layers, accelerator)
                if accelerator.is_main_process:
                    log.info(f"[step {global_step}] FRAGMENTATION EXIT — "
                             f"Resume: RESUME_DELTA=...checkpoint-{global_step}")
                accelerator.wait_for_everyone()
                if dist.is_initialized():
                    dist.destroy_process_group()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                sys.exit(0)

        if global_step >= total_steps:
            break

    _free_for_save(model, optimizer)
    _save_checkpoint(model, None, lr_scheduler, tokenizer,
                    output_dir, global_step, epoch, keystone_layers, accelerator, final=True)
    accelerator.wait_for_everyone()


def _evict_page_cache(filepath):
    """Evict file from page cache to protect UMA headroom."""
    try:
        fd = os.open(filepath, os.O_RDONLY)
        os.fsync(fd)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)
    except (OSError, AttributeError):
        pass


def _free_for_save(model, optimizer):
    """Free optimizer state + gradients for save."""
    import gc
    for p in model.parameters():
        if p.grad is not None:
            p.grad = None
    if optimizer is not None:
        optimizer.state.clear()
        optimizer.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()
    mem = torch.cuda.mem_get_info()
    log.info(f"Pre-save cleanup: freed optimizer+grads, now {mem[0]/1e9:.1f}GB free GPU")


def _save_checkpoint(model, optimizer, lr_scheduler, tokenizer,
                     output_dir, step, epoch, keystone_layers, accelerator, final=False):
    """Save trainable weights using FSDP summon_full_params."""
    ckpt_name = "final" if final else f"checkpoint-{step}"
    ckpt_dir = os.path.join(output_dir, ckpt_name)
    rank = accelerator.process_index

    os.makedirs(ckpt_dir, exist_ok=True)
    mem = torch.cuda.mem_get_info()
    log.info(f"Rank {rank}: saving to {ckpt_dir} | free={mem[0]/1e9:.1f}GB")
    accelerator.wait_for_everyone()

    import gc
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    gc.collect()
    torch.cuda.empty_cache()

    trainable_state = {}
    shard_gb = 0.0
    with FSDP.summon_full_params(model, rank0_only=True, writeback=False):
        if accelerator.is_main_process:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    trainable_state[name] = param.detach().cpu().clone()
            shard_gb = sum(t.numel() * t.element_size() for t in trainable_state.values()) / 1e9
            log.info(f"Gathered {len(trainable_state)} trainable tensors ({shard_gb:.2f}GB) on rank 0")

    if accelerator.is_main_process:
        from safetensors.torch import save_file

        # 1. Save ALL trainable params to one universal file
        # This works for ANY freeze config — no hardcoded categories
        all_file = os.path.join(ckpt_dir, "trainable_weights.safetensors")
        save_file(trainable_state, all_file)
        log.info(f"Saved {len(trainable_state)} trainable tensors ({shard_gb:.2f}GB) to {all_file}")

        # 2. Also save PEFT adapter format for bake_lora.py compatibility
        lora_state = {k: v for k, v in trainable_state.items() if 'lora_' in k.lower()}
        if lora_state:
            unwrapped = accelerator.unwrap_model(model)
            unwrapped.save_pretrained(
                ckpt_dir,
                state_dict=lora_state,
                safe_serialization=True,
                save_embedding_layers=False,
                is_main_process=True,
            )
            log.info(f"  + {len(lora_state)} LoRA tensors via save_pretrained")

        # 3. Validation: count what we expected vs what we saved
        expected = sum(1 for p in model.parameters() if p.requires_grad)
        if len(trainable_state) < expected:
            log.warning(f"SAVE VALIDATION: saved {len(trainable_state)} tensors but "
                        f"{expected} params have requires_grad=True — some may be missing")

        # Save metadata BEFORE wait (protect against crash)
        meta = {
            "step": step,
            "epoch": epoch,
            "num_ranks": accelerator.num_processes,
            "max_seq": int(os.environ.get("MAX_SEQ", "8192")),
            "keystone_layers": keystone_layers,
            "method": "lora_plus_router_v1",
            "lora_r": 64,
            "lora_alpha": 128,
            "router_trained": True,
        }
        meta_file = os.path.join(ckpt_dir, "trainer_meta.pt")
        torch.save(meta, meta_file)
        _evict_page_cache(meta_file)

        tokenizer.save_pretrained(ckpt_dir)
    accelerator.wait_for_everyone()

    mem = torch.cuda.mem_get_info()
    if accelerator.is_main_process:
        log.info(f"Rank {rank}: saved {shard_gb:.2f}GB trainable checkpoint | free={mem[0]/1e9:.1f}GB")
    else:
        log.info(f"Rank {rank}: checkpoint save complete | free={mem[0]/1e9:.1f}GB")
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
