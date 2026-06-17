#!/usr/bin/env python3
"""Dense Qwen3.5-9B-Base FSDP LoRA SFT — surgical adaptation of train_fsdp_v3.py.

Same proven 4-Spark FSDP+LoRA pipeline (rank-split mmap load,
accelerator.prepare(), explicit FSDP MixedPrecision policy, Adafactor
optimizer, summon_full_params save, session-limit fragmentation exit).
Only the MoE-specific lines were changed:

  1. Import dense decoder layer (Qwen3_5DecoderLayer) instead of the
     MoE one (Qwen3_5MoeDecoderLayer).
  2. LoRA target_modules: drop `shared_expert.*` (no shared expert on
     dense), add `mlp.{gate,up,down}_proj` (dense analogue).
  3. Drop `output_router_logits = False` (no router on dense).
  4. The FREEZE_CONFIG branches that reference `mlp.experts.` /
     `shared_expert_gate` are no-ops on dense — that's fine, LoRA is
     the trainable surface.
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
import psutil
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, DistributedSampler, Dataset
import glob

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


class BucketSFTDataset(Dataset):
    """Pre-tokenized, length-sorted SFT dataset for bucket batching.

    Per Perplexity DR (2026-04-28): the original CombinedSFTDataset used
    `random.random()` weighted sampling and ignored the DistributedSampler's
    indices, producing variable-length batches and probabilistic epoch
    coverage rather than partitioned. Replaced with a deterministic dataset
    that tokenizes everything up front, sorts by length, and lets the
    DistributedSampler partition the sorted indices across ranks.

    NO truncation: items longer than max_seq are split into overlapping
    windows (256-token overlap), preserving content per
    feedback_never_truncate.md.
    """

    def __init__(self, sft_jsonl_path, tokenizer, max_seq):
        self.tokenizer = tokenizer
        self.max_seq = max_seq

        # ── Cache key ─────────────────────────────────────────────────────
        # Hash of: corpus path + corpus mtime + tokenizer.name_or_path + max_seq
        # If any of those change, regenerate the cache.
        import hashlib, pickle, time
        corpus_st = os.stat(sft_jsonl_path)
        cache_key_parts = [
            os.path.abspath(sft_jsonl_path),
            str(corpus_st.st_size),
            str(int(corpus_st.st_mtime)),
            getattr(tokenizer, "name_or_path", "?"),
            f"max_seq={max_seq}",
        ]
        cache_key = hashlib.sha256("|".join(cache_key_parts).encode()).hexdigest()[:16]
        cache_dir = os.path.join(os.path.dirname(sft_jsonl_path), "tokenized_cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"sft_{cache_key}.pkl")

        # ── Try cache load ────────────────────────────────────────────────
        if os.path.exists(cache_path):
            try:
                t0 = time.time()
                with open(cache_path, "rb") as f:
                    self.samples = pickle.load(f)
                log.info(
                    f"BucketSFTDataset: loaded {len(self.samples)} pre-tokenized "
                    f"samples from cache {cache_path} in {time.time()-t0:.1f}s "
                    f"(saved ~7min of re-tokenization)"
                )
                return
            except Exception as e:
                log.warning(f"Cache load failed ({e}); falling back to fresh tokenize")

        # ── Fresh tokenize ────────────────────────────────────────────────
        self.samples = []  # list of (input_ids, labels)
        log.info(f"Pre-tokenizing SFT corpus from {sft_jsonl_path} (no cache hit)...")
        n_rows = 0
        n_split = 0
        with open(sft_jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                msgs = row.get("messages")
                if not isinstance(msgs, list) or not msgs:
                    continue
                try:
                    ids, labels = _tokenize_sft_pair(msgs, tokenizer)
                except Exception:
                    continue
                n_rows += 1
                if len(ids) <= max_seq:
                    self.samples.append((ids, labels))
                else:
                    # Split with overlap rather than truncate
                    stride = max_seq - 256
                    for start in range(0, len(ids), stride):
                        chunk_ids = ids[start:start + max_seq]
                        chunk_labels = labels[start:start + max_seq]
                        if len(chunk_ids) < 64 and start > 0:
                            break
                        self.samples.append((chunk_ids, chunk_labels))
                        n_split += 1
                        if start + max_seq >= len(ids):
                            break

        # REAL bucket batching: sort by length so a length-grouped sampler
        # (or simple sorted sampler) produces batches of similar-length
        # samples. Padding happens dynamically in collate_fn — pads only to
        # the batch max, NOT to max_seq.
        self.samples.sort(key=lambda s: len(s[0]))

        log.info(f"BucketSFTDataset: {n_rows} rows -> {len(self.samples)} samples "
                 f"({n_split} extra from outlier splits, length-sorted, "
                 f"DYNAMIC padding via collate)")

        # ── Save cache (atomic via temp+rename) ───────────────────────────
        # Multi-rank race-safe: all 4 ranks tokenize independently first time
        # (waste of CPU but correct), each writes to its own temp, last one
        # to rename wins. Subsequent launches all hit the cache.
        try:
            tmp_path = cache_path + f".tmp.{os.getpid()}"
            with open(tmp_path, "wb") as f:
                pickle.dump(self.samples, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.rename(tmp_path, cache_path)
            log.info(f"Saved tokenized cache to {cache_path}")
        except Exception as e:
            log.warning(f"Cache save failed ({e}); training will proceed without cache")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids, labels = self.samples[idx]
        return {"input_ids": ids, "labels": labels, "is_dpo": False}


class CombinedSFTDataset(Dataset):
    """Combined SFT dataset — zero-footprint byte-offset indexing.
    Weighted sampling: 55% SFT, 30% CPT (constitutional+infra), 15% general.
    """

    def __init__(self, sft_dir, cpt_path, general_dir, tokenizer, max_seq):
        self.tokenizer = tokenizer
        self.max_seq = max_seq

        log.info("Building byte-offset indexes...")
        sft_files = sorted(glob.glob(os.path.join(sft_dir, "*.jsonl"))) if sft_dir and os.path.isdir(sft_dir) else []
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
            assert len(tokens) <= self.max_seq, (
                f"CPT row exceeds max_seq: {len(tokens)}>{self.max_seq}; corpus must be pre-chunked"
            )
            return {"input_ids": tokens, "labels": labels, "is_dpo": False}
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
    # output_router_logits skipped — dense Qwen3.5-9B has no router
    # gradient checkpointing ENABLED for full-FT — without this the autograd
    # graph for all 8.95B trainable params at seq=8192 holds ~12GB activations
    # per rank for the backward pass, which silently OOMs UMA at first training
    # step (manifests as 96% GPU spin-wait at 18W). The 11x-slower comment in
    # train_fsdp_v3.py applied to the LoRA case where activation memory was
    # tiny; for full-FT this is the standard fix and matches every other
    # full-FT script in the repo (train_cpt_qwen35_dense.py:96,
    # train_recovery_sft_qwen35_dense.py:88, train_cpt_fsdp.py:117, etc.).
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # ── FULL-PARAMETER SFT — every weight is trainable ──
    # Per TOOLS.md Phase 1 plan ("Phase 1: SFT (~70K tools+chat examples,
    # 2-3 epochs)") and Perplexity DR diagnosis 2026-04-28, this is
    # full-parameter SFT, not LoRA. We unfreeze every parameter explicitly
    # before FSDP wraps the model so FSDP respects requires_grad=True
    # uniformly.
    for param in model.parameters():
        param.requires_grad_(True)
    if accelerator.is_main_process:
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        log.info(f"FULL-FT: {n_trainable/1e9:.2f}B trainable / {n_total/1e9:.2f}B total")

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

    # ── Full-parameter SFT — no FREEZE_CONFIG branches ──
    # All params already set to requires_grad=True after model load. No
    # router/expert categorization (no MoE on dense). No LoRA-vs-base
    # safety checks (no LoRA on full FT). Single trainable surface.
    router_params = []   # kept as empty so the optimizer-builder code below
    expert_params = []   # doesn't break; both groups are zero-length here.

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

    # ── FSDP wrapping (transformer-block layer only — no LoRA wrap) ──
    import functools
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer
    layer_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={Qwen3_5DecoderLayer},
    )
    accelerator.state.fsdp_plugin.auto_wrap_policy = layer_policy
    if accelerator.is_main_process:
        log.info("FSDP wrap policy: Qwen3_5DecoderLayer (full-param FT, no LoRA wrap)")

    # ── Dataset selection: SFT (Phase 1) vs CPT (Phase 2) ─────────────────
    # SFT mode: SFT_DIR points at a dir with *.jsonl messages-format data.
    #           Use BucketSFTDataset (length-sorted, dynamic-padding bucket).
    # CPT mode: SFT_DIR is empty AND CPT_DATA points at a jsonl with
    #           {"text": "..."} entries. Use CombinedSFTDataset which has the
    #           is_cpt branch that tokenizes raw text + supervises all tokens.
    #           Random sampling (no length sort) for CPT — fine because pure
    #           CPT has no chat-template structure to align.
    if cpt_data and (not sft_dir or not os.path.isdir(sft_dir)):
        if accelerator.is_main_process:
            log.info(f"CPT mode: dataset = CombinedSFTDataset(cpt_path={cpt_data})")
        dataset = CombinedSFTDataset(
            sft_dir="", cpt_path=cpt_data, general_dir="",
            tokenizer=tokenizer, max_seq=max_seq,
        )
    else:
        sft_jsonl = os.environ.get(
            "SFT_JSONL",
            os.path.join(sft_dir, "tools_sft.jsonl"),
        )
        dataset = BucketSFTDataset(sft_jsonl, tokenizer, max_seq)

    # Dynamic-padding collate: pad to BATCH-MAX (not max_seq), aligned to a
    # multiple of 64 for tensor-core friendliness. Per Perplexity DR
    # (SFT_PHASE1_ISSUES.md Issue 6) this is the actual throughput win that
    # bucket batching enables — short batches stay short, long batches grow.
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    PAD_TO_MULTIPLE_OF = 64

    def _round_up(n, k):
        return ((n + k - 1) // k) * k

    def collate_fn(batch):
        max_len = max(len(b["input_ids"]) for b in batch)
        target = _round_up(max_len, PAD_TO_MULTIPLE_OF)
        n = len(batch)
        input_ids = torch.full((n, target), pad_id, dtype=torch.long)
        labels = torch.full((n, target), -100, dtype=torch.long)
        for i, b in enumerate(batch):
            ids = b["input_ids"]
            lbl = b["labels"]
            input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            labels[i, : len(lbl)] = torch.tensor(lbl, dtype=torch.long)
        return {"input_ids": input_ids, "labels": labels, "is_dpo": False}

    # Bucket-batching sampler: partition the length-sorted dataset into
    # contiguous chunks of size (batch_size * world_size), shuffle the chunks
    # for some epoch-level randomization (don't shuffle within chunks — that
    # would defeat bucket batching). Each rank reads its slot from each chunk.
    train_batch_size = int(os.environ.get("BATCH_SIZE_PER_RANK", "4"))
    sampler = DistributedSampler(
        dataset,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        shuffle=False,  # the dataset is already length-sorted; shuffling defeats it
    )
    dataloader = DataLoader(
        dataset,
        batch_size=train_batch_size,
        sampler=sampler,
        collate_fn=collate_fn,
        pin_memory=False,
        num_workers=0,
    )

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

    # ── Optimizer: single param group, full-parameter Adafactor ──
    # Adafactor (not AdamW) for all 9B params — factored row/column sums
    # cut optimizer state from ~72 GB (AdamW fp32 m+v on 9B) to a few GB.
    # AdamW would not fit even sharded across 4 nodes.
    from transformers.optimization import Adafactor
    sft_lr = float(os.environ.get("LR", "1e-5"))
    optimizer = Adafactor(
        [p for p in model.parameters() if p.requires_grad],
        lr=sft_lr,
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
        weight_decay=0.01,
        clip_threshold=1.0,
    )
    if accelerator.is_main_process:
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log.info(f"Optimizer: Adafactor (single group, {n_trainable/1e9:.2f}B params @ lr={sft_lr})")

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

    # ── Post-FSDP diagnostic forward ──
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
                 f"logits_shape={diag_logits.shape} "
                 f"logits_nan%={100*torch.isnan(diag_logits).float().mean().item():.1f}% "
                 f"logits_inf%={100*torch.isinf(diag_logits).float().mean().item():.1f}% "
                 f"logits_min={diag_logits.min().item():.4f} "
                 f"logits_max={diag_logits.max().item():.4f} "
                 f"logits_mean={diag_logits[torch.isfinite(diag_logits)].mean().item():.4f}")
        # Check first few param norms post-FSDP
        for i, (name, p) in enumerate(model.named_parameters()):
            if i >= 5:
                break
            pn = p.float().norm().item() if p.numel() > 0 and not p.is_meta else -1
            log.info(f"  param[{i}] {_clean_fsdp_name(name)}: norm={pn:.4f} dtype={p.dtype} device={p.device}")
        del diag_out, diag_logits

    # Test train mode forward (still no_grad, to isolate train vs grad)
    model.train()
    log.info(f"Rank {accelerator.process_index}: running train-mode (no_grad) diagnostic...")
    with torch.no_grad():
        diag_ids2 = torch.tensor([[1, 2, 3, 4, 5]], device=accelerator.device)
        diag_labels2 = torch.tensor([[1, 2, 3, 4, 5]], device=accelerator.device)
        diag_out2 = model(input_ids=diag_ids2, labels=diag_labels2)
        log.info(f"Rank {accelerator.process_index}: TRAIN-MODE diag: "
                 f"loss={diag_out2.loss.item():.4f} "
                 f"finite={torch.isfinite(diag_out2.loss).item()}")
        del diag_out2
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

    if accelerator.is_main_process:
        log.info(f"Starting: steps {resume_step}→{total_steps}, {accelerator.num_processes} nodes")

    for epoch in range(100):
        sampler.set_epoch(epoch)
        for batch in dataloader:
            if global_step >= total_steps:
                break

            # Standard SFT loss (packed sequences)
            outputs = model(input_ids=batch["input_ids"], labels=batch["labels"])
            loss = outputs.loss
            del outputs

            # Periodic DPO step
            if dpo_dataloader and (global_step - resume_step) % dpo_interval == 0 and (global_step - resume_step) > 0:
                try:
                    dpo_batch = next(dpo_iter)
                except StopIteration:
                    dpo_iter = iter(dpo_dataloader)
                    dpo_batch = next(dpo_iter)
                # DPO uses chosen/rejected from the batch
                if dpo_batch.get("is_dpo", False):
                    dpo_beta = 0.1
                    chosen_out = model(input_ids=dpo_batch["input_ids"], labels=dpo_batch["labels"])
                    rej_out = model(input_ids=dpo_batch["rejected_input_ids"], labels=dpo_batch["rejected_labels"])
                    dpo_loss = -torch.nn.functional.logsigmoid(dpo_beta * (-chosen_out.loss - (-rej_out.loss)))
                    loss = loss + dpo_weight * dpo_loss
                    del chosen_out, rej_out

            # Distributed NaN veto
            nan_flag = torch.zeros(1, device=accelerator.device)
            if torch.isnan(loss) or torch.isinf(loss):
                nan_flag.fill_(1.0)
            dist.all_reduce(nan_flag, op=dist.ReduceOp.MAX)

            if nan_flag.item() > 0:
                if accelerator.is_main_process:
                    log.warning(f"[step {global_step+1}] NaN/Inf — ALL ranks skipping")
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                continue

            accelerator.backward(loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

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

        # Full-FT save: dump every trainable tensor (which is every tensor)
        # into a single safetensors file under the standard name the proven
        # resume code looks for (`trainable_weights.safetensors`). The earlier
        # v8 used `model.safetensors` which broke the RESUME_DELTA path.
        out_file = os.path.join(ckpt_dir, "trainable_weights.safetensors")
        save_file(trainable_state, out_file)
        log.info(f"Saved {len(trainable_state)} tensors ({shard_gb:.2f}GB) to {out_file}")

        # Save metadata
        meta = {
            "step": step,
            "epoch": epoch,
            "num_ranks": accelerator.num_processes,
            "max_seq": int(os.environ.get("MAX_SEQ", "8192")),
            "method": "full_ft_dense_9b_v1",
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
