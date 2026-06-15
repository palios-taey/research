#!/usr/bin/env python3
"""Continued Pre-Training (CPT) of Qwen3.5-9B-Base on PALIOS-TAEY constitutional corpus.

Single-node BF16 full-parameter CPT. Designed for a GB10 DGX Spark (128GB unified
memory) — no quantization, no FSDP, no sequence packing. Addresses Perplexity's
documented Qwen3.5 CPT quirks:

  1. Packing causes NaN gradients at step 1 -> packing DISABLED; each document
     is tokenized independently.
  2. Long sequences cause NaN at >65K tokens -> max_seq_length=4096 is well below.
  3. transformers >=5.0.0 required for qwen3_5 architecture (Spark has 5.3.0).
  4. BF16 only on GB10 / ARM64 (NVFP4 crashes).
  5. No chat template applied -> raw-text language modeling objective.
  6. MTP heads are not saved by HF Trainer -> post-hoc copy from base required
     if the resulting checkpoint will be served via vLLM speculative decoding.

No-truncate chunking: documents longer than max_seq are split into overlapping
windows with head/tail anchors per Seamless Packing (Yin et al. 2025). This is
the canonical anti-truncate path enforced by feedback_never_truncate.md.

Usage:
    export HF_TOKEN=$(cat /tmp/hf_token_secure)
    python3 train_cpt_qwen35_dense.py \\
        --model /home/spark/models/Qwen3.5-9B-Base \\
        --data /var/spark/isma/training/cpt_canonical.jsonl \\
        --output /home/spark/training_outputs/cpt_qwen35_9b_v1
"""

import argparse
import gc
import json
import logging
import os
import time
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


class SaveSafeTrainer(Trainer):
    """Trainer subclass with sm_121-safe checkpoint save.

    HF Trainer's default _save calls model.save_pretrained → safetensors.serialize_file
    which does a single-shot gather of all weights via tensor.to('cpu'). On
    Blackwell GB10 (sm_121) — past PyTorch's max-supported capability of 12.0 —
    this triggers `cudaErrorLaunchFailure` mid-write (CPT v2 first attempt
    crashed at step 200 with "CUDA error: unspecified launch failure" inside
    safetensors._tobytes → tensor.to('cpu')).

    Workaround: mirror Phase 1 SFT's working pattern from train_fsdp_dense_9b.py
    — gc.collect + torch.cuda.empty_cache before save, then iterate
    named_parameters one-at-a-time with .detach().cpu().clone() to build a CPU
    state dict, then safetensors.save_file from the CPU dict. The per-tensor
    GPU→CPU copy uses different CUDA APIs than safetensors' bulk path and has
    empirical precedent shipping on these exact Sparks.
    """

    def _save(self, output_dir=None, state_dict=None):
        output_dir = output_dir if output_dir else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        log.info(f"[save] starting safe save to {output_dir}")
        t0 = time.time()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            free_gb = torch.cuda.mem_get_info()[0] / 1e9
            log.info(f"[save] post-empty_cache free={free_gb:.1f}GB")

        # Save small text artifacts first (no GPU access)
        self.model.config.save_pretrained(output_dir)
        if hasattr(self.model, "generation_config") and self.model.generation_config is not None:
            self.model.generation_config.save_pretrained(output_dir)
        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)

        # Iterate weights one at a time, GPU→CPU per tensor (the working pattern).
        cpu_state = {}
        n_params = 0
        for name, param in self.model.named_parameters():
            cpu_state[name] = param.detach().cpu().clone()
            n_params += 1
        for name, buf in self.model.named_buffers():
            if name in cpu_state:
                continue
            cpu_state[name] = buf.detach().cpu().clone()

        bytes_total = sum(t.numel() * t.element_size() for t in cpu_state.values())
        log.info(f"[save] gathered {n_params} params + {len(cpu_state) - n_params} buffers"
                 f" → {bytes_total/1e9:.2f}GB on CPU")

        from safetensors.torch import save_file
        out_file = os.path.join(output_dir, "model.safetensors")
        save_file(cpu_state, out_file)
        log.info(f"[save] wrote {out_file} in {time.time()-t0:.1f}s total")

        # Free CPU memory after write so the next training step doesn't see a
        # large lingering host-side dict.
        del cpu_state
        gc.collect()


def load_jsonl(path: str) -> list:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = row.get("text", "").strip()
            if text and len(text) > 50:
                rows.append({"text": text,
                             "source_file": row.get("source_file", ""),
                             "tier": row.get("tier", "cpt")})
    return rows


def chunk_long_doc(text, tokenizer, max_seq, source_file="", tier="cpt",
                   anchor_size=512, overlap_tokens=1024):
    """Split docs longer than max_seq into overlapping windows with anchors.

    Mirrors build_training_data.chunk_document but uses the trainer's own
    tokenizer (no dep on the 35B-abliterated tokenizer that script reaches for).
    Short docs returned as-is.

    Per feedback_never_truncate: never lose content by truncation.
    """
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= max_seq:
        return [text]

    head_ids = ids[:anchor_size]
    tail_ids = ids[-anchor_size:]
    head_text = tokenizer.decode(head_ids, skip_special_tokens=True)
    tail_text = tokenizer.decode(tail_ids, skip_special_tokens=True)

    body_ids = ids[anchor_size:-anchor_size] if len(ids) > 2 * anchor_size else ids
    body_capacity = max_seq - (2 * anchor_size) - 64  # reserve for header/EOS
    if body_capacity < 256:
        return [text]  # Document is small enough that anchors+body fit

    overlap = min(overlap_tokens, body_capacity // 2)
    stride = body_capacity - overlap

    out = []
    chunk_idx = 0
    fname = os.path.basename(source_file) if source_file else "unknown"
    for start in range(0, len(body_ids), stride):
        body_slice = body_ids[start:start + body_capacity]
        if len(body_slice) < 256:
            break
        body_text = tokenizer.decode(body_slice, skip_special_tokens=True)
        out.append(
            f"[DOCUMENT: {fname} | CHUNK {chunk_idx + 1} | TIER: {tier}]\n\n"
            f"{head_text}\n\n[...continued...]\n\n"
            f"{body_text}\n\n[...end section...]\n\n{tail_text}"
        )
        chunk_idx += 1
    return out if out else [text]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to Qwen3.5-9B-Base")
    ap.add_argument("--data", required=True, help="JSONL with {text: ...}")
    ap.add_argument("--output", required=True, help="Output dir")
    ap.add_argument("--max-seq", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup-steps", type=int, default=100)
    ap.add_argument("--num-epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=-1,
                    help="If >0, stop after this many optimizer steps. "
                         "Use for smoke-testing the save mechanism.")
    args = ap.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)

    log.info(f"Model: {args.model}")
    log.info(f"Data:  {args.data}")
    log.info(f"Output: {args.output}")
    log.info(f"max_seq={args.max_seq} lr={args.lr} warmup={args.warmup_steps}")
    log.info(f"batch={args.batch_size} grad_accum={args.grad_accum} epochs={args.num_epochs}")

    log.info("Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    log.info("Loading model in bf16...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    log.info("Loading corpus...")
    rows = load_jsonl(args.data)
    log.info(f"Loaded {len(rows)} documents")

    log.info(f"Chunking docs longer than max_seq={args.max_seq} (no truncation)...")
    chunked = []
    n_split = 0
    for r in rows:
        pieces = chunk_long_doc(r["text"], tok, args.max_seq,
                                source_file=r.get("source_file", ""),
                                tier=r.get("tier", "cpt"))
        if len(pieces) > 1:
            n_split += 1
        for p in pieces:
            chunked.append({"text": p})
    log.info(f"After chunking: {len(rows)} docs -> {len(chunked)} sequences "
             f"({n_split} docs were split)")
    ds = Dataset.from_list(chunked)

    def tokenize_fn(batch):
        # No chat template — pure causal LM on raw text.
        # No truncation — chunk_long_doc above guarantees every text fits.
        # add_special_tokens=False matches chunk_long_doc's tokenization mode
        # so the boundary token counts agree.
        return tok(
            batch["text"],
            padding=False,
            add_special_tokens=False,
            return_special_tokens_mask=False,
        )

    log.info("Tokenizing...")
    tokenized = ds.map(
        tokenize_fn,
        batched=True,
        remove_columns=ds.column_names,
        num_proc=4,
    )
    log.info(f"Tokenized {len(tokenized)} sequences")

    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.num_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type="cosine",
        weight_decay=0.0,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-8,
        max_grad_norm=1.0,
        bf16=True,
        bf16_full_eval=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",
        logging_steps=args.log_every,
        save_steps=args.save_every,
        save_total_limit=4,
        report_to="none",
        dataloader_num_workers=2,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
    )

    log.info("Starting training...")
    trainer = SaveSafeTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=collator,
        processing_class=tok,
    )

    trainer.train()

    log.info(f"Training complete. Saving final model to {args.output}/final")
    trainer.save_model(f"{args.output}/final")
    tok.save_pretrained(f"{args.output}/final")
    log.info("DONE")


if __name__ == "__main__":
    main()
