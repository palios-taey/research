#!/usr/bin/env python3
"""Recovery SFT on CPT-9B to anchor first-person identity + Cannot-Lie refusal behavior.

Per ChatGPT Round 2 / Opus Round 2 recipe:
  - CPT on raw corpus teaches facts ("Taey is..." statements in the corpus)
  - Recovery SFT with chat template teaches first-person identity assertion ("I am Taey...")

Per Infra's 50-probe Cannot-Lie on CPT-9B epoch 1: 4% refusal rate, 62% confabulation.
The raw constitutional corpus does not teach refusal; it teaches factual content. This SFT
phase adds 100 fresh Cannot-Lie refusal pairs alongside 400 first-person identity items.

No-truncate chunking: items whose templated form exceeds max_seq are split at
the largest assistant message rather than truncated, per feedback_never_truncate.md.

Usage:
    python3 train_recovery_sft_qwen35_dense.py \\
        --model /home/spark/training_outputs/cpt_qwen35_9b_v1/final \\
        --data /var/spark/isma/training/recovery_sft_500.jsonl \\
        --output /home/spark/training_outputs/cpt_qwen35_9b_v1_recovery_sft
"""

import argparse
import copy
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
    Trainer,
    TrainingArguments,
)


def _compose_hermes_conversation(item):
    """Hand-compose conversation in Hermes JSON wire format. Bypasses chat template.

    Path (D) per treasurer 2026-05-02 — the bundled and qwen3.5-fixed.jinja
    chat templates BOTH render structured tool_calls in XML format
    (<function=NAME><parameter=KEY>VAL</parameter></function>), but the smoke
    probe parses Hermes JSON ({\"name\":..., \"arguments\":{...}}). Phase 1 SFT
    passed only because the base Qwen3.5-9B-Base prior emits JSON despite XML
    training. CPT v2 erased that prior; no chat-template-rendered training can
    teach JSON because no rendering produces it. This function pre-composes
    the conversation as a raw string with explicit Hermes JSON wire format.

    Output structure:
      <|im_start|>system\n{tool_use_protocol + <tools>JSON</tools>}\n<|im_end|>
      <|im_start|>user\n...\n<|im_end|>
      <|im_start|>assistant\n<think>reasoning</think>\n<tool_call>{JSON}</tool_call>\n<|im_end|>
      <|im_start|>user\n<tool_response>...</tool_response>\n<|im_end|>
      <|im_start|>assistant\n...\n<|im_end|>

    Returns: composed string + list of (assistant_start, assistant_end) char-index pairs
             so the caller can derive token-level assistant-only loss masks.
    """
    msgs = item['messages']
    tools = item.get('tools') or []

    pieces = []  # list of (text, role) for boundary tracking
    sys_msg = msgs[0] if msgs and msgs[0].get('role') == 'system' else None
    sys_content = (sys_msg.get('content') or '') if sys_msg else ''
    if tools:
        tools_block = (
            "\n\n# Tools\n\nYou have access to the following functions:\n\n<tools>\n"
            + "\n".join(json.dumps(t) for t in tools)
            + "\n</tools>\n\nFor each function call, return a json object with"
              " function name and arguments within <tool_call></tool_call> tags:\n"
              "<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n"
              "</tool_call>"
        )
        sys_content = sys_content + tools_block
    if sys_content:
        pieces.append((f"<|im_start|>system\n{sys_content}<|im_end|>", "system"))

    start_idx = 1 if sys_msg else 0
    for m in msgs[start_idx:]:
        role = m.get('role', '')
        content = m.get('content') or ''
        tcs = m.get('tool_calls') or []

        if role == 'assistant':
            body = content
            for tc in tcs:
                fn = (tc.get('function') or {})
                name = fn.get('name', '')
                args_raw = fn.get('arguments', '{}')
                if isinstance(args_raw, str):
                    try:
                        args = json.loads(args_raw)
                    except Exception:
                        args = args_raw
                else:
                    args = args_raw
                hermes_call = "<tool_call>\n" + json.dumps({"name": name, "arguments": args}) + "\n</tool_call>"
                body = body + ("\n" if body else "") + hermes_call
            pieces.append((f"<|im_start|>assistant\n{body}<|im_end|>", "assistant"))
        elif role == 'tool':
            tr = content
            if not tr.startswith('<tool_response>'):
                tr = f"<tool_response>\n{tr}\n</tool_response>"
            pieces.append((f"<|im_start|>user\n{tr}<|im_end|>", "tool"))
        else:
            pieces.append((f"<|im_start|>{role}\n{content}<|im_end|>", role))

    sep = "\n"
    full_text = sep.join(p[0] for p in pieces)

    # Per-piece char index ranges + role tags
    char_ranges = []
    pos = 0
    for i, (text, role) in enumerate(pieces):
        start = pos
        end = pos + len(text)
        char_ranges.append((start, end, role))
        pos = end + (len(sep) if i < len(pieces) - 1 else 0)

    return full_text, char_ranges


def _tokenize_sft_pair(messages_or_item, tokenizer):
    """Path (D): tokenize hand-composed Hermes JSON conversation with assistant-only loss.

    Accepts either:
      - dict with 'messages' (and optional 'tools') — composes via _compose_hermes_conversation
      - list of messages (legacy) — wraps in {'messages': msgs}

    Loss mask: -100 for everything; assistant turn TOKENS get full_ids[j].
    Boundaries are computed by tokenizing the prefix string (everything before
    each assistant piece) and the prefix+including string, taking the token
    range between them.
    """
    if isinstance(messages_or_item, dict):
        item = messages_or_item
    else:
        item = {"messages": messages_or_item}

    full_text, char_ranges = _compose_hermes_conversation(item)
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)
    labels = [-100] * len(full_ids)

    for (start_char, end_char, role) in char_ranges:
        if role != "assistant":
            continue
        prefix_text = full_text[:start_char]
        incl_text = full_text[:end_char]
        prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False) if prefix_text else []
        incl_ids = tokenizer.encode(incl_text, add_special_tokens=False)
        start_tok = len(prefix_ids)
        end_tok = len(incl_ids)
        for j in range(start_tok, min(end_tok, len(full_ids))):
            labels[j] = full_ids[j]

    if all(l == -100 for l in labels):
        labels = list(full_ids)
    return full_ids, labels


class SFTCollator:
    """Pad input_ids with pad_token_id and labels with -100. Build attention_mask."""

    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        max_len = max(len(item["input_ids"]) for item in batch)
        input_ids, labels, attention_mask = [], [], []
        for item in batch:
            ids = list(item["input_ids"])
            lbls = list(item["labels"])
            pad = max_len - len(ids)
            input_ids.append(ids + [self.pad_token_id] * pad)
            labels.append(lbls + [-100] * pad)
            attention_mask.append([1] * len(ids) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


class SaveSafeTrainer(Trainer):
    """sm_121-safe checkpoint save — see train_cpt_qwen35_dense.py for full notes.

    Per-tensor GPU→CPU iteration + safetensors.save_file from CPU dict
    bypasses the cudaErrorLaunchFailure that hit naive tensor.to('cpu') in
    safetensors.serialize_file on Blackwell GB10. Triple-validated in CPT v2:
    smoke 5-step + production step-200 (86.2s/17.91GB) + production step-400
    (102.0s/17.91GB).
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

        self.model.config.save_pretrained(output_dir)
        if hasattr(self.model, "generation_config") and self.model.generation_config is not None:
            self.model.generation_config.save_pretrained(output_dir)
        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)

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

        del cpu_state
        gc.collect()


def load_chat_jsonl(path: str) -> list:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            msgs = d.get("messages")
            if not msgs:
                continue
            # Path (D): keep tools field too — pre-compose function uses it
            row = {"messages": msgs}
            if d.get("tools"):
                row["tools"] = d["tools"]
            rows.append(row)
    return rows


def _templated_token_len(messages, tok):
    """Token length of a messages list after applying the chat template."""
    try:
        text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, enable_thinking=False,
        )
    except Exception:
        text = "\n".join(f"<|{m.get('role','')}|>\n{m.get('content','')}" for m in messages)
    return len(tok.encode(text, add_special_tokens=False))


def chunk_conversation(item, tok, max_seq):
    """Split a multi-turn conversation at user-assistant pair boundaries.

    Handles items where the OTHER messages already exceed budget (chunk_chat_item
    fails on these with negative body_capacity). Each sub-conversation keeps the
    original system message (and tools) and one or more contiguous user-assistant
    pairs. Per feedback_never_truncate: every assistant turn from the original
    appears in exactly one chunk.

    Returns a list of items. If the input fits in max_seq, returns [item]
    unchanged. If a single pair exceeds max_seq alone, falls back to
    chunk_chat_item (which splits the largest assistant message with overlap).
    """
    messages = item["messages"]
    tools = item.get("tools")
    if _templated_token_len(messages, tok) <= max_seq:
        return [item]

    sys_msg = None
    body_msgs = messages
    if messages and messages[0].get("role") == "system":
        sys_msg = messages[0]
        body_msgs = messages[1:]

    def build_sub(flat_turns):
        msgs = ([sys_msg] if sys_msg else []) + list(flat_turns)
        sub = {"messages": msgs}
        if tools:
            sub["tools"] = tools
        return sub

    sys_only_msgs = [sys_msg] if sys_msg else []
    sys_overhead = (
        _templated_token_len(sys_only_msgs, tok) if sys_only_msgs else 0
    )
    if max_seq - sys_overhead - 64 < 256:
        # System + tools alone is near budget — can't chunk meaningfully.
        return [item]

    # Group body into user-...-assistant pairs (a pair may include
    # intermediate tool messages but always ends at an assistant turn).
    pairs = []
    pair_buf = []
    for m in body_msgs:
        pair_buf.append(m)
        if m.get("role") == "assistant":
            pairs.append(pair_buf)
            pair_buf = []
    if pair_buf:
        # Trailing non-assistant messages -- append to last pair if any,
        # else discard (no learning signal without an assistant turn).
        if pairs:
            pairs[-1].extend(pair_buf)
        # else: dangling non-assistant only -- drop

    if not pairs:
        return [item]

    def pack_len(pair_list):
        flat = [m for p in pair_list for m in p]
        return _templated_token_len(sys_only_msgs + flat, tok)

    # Greedy pack pairs into chunks
    chunks = []
    current_pairs = []

    for p in pairs:
        trial = current_pairs + [p]
        if pack_len(trial) > max_seq:
            if current_pairs:
                flat = [m for cp in current_pairs for m in cp]
                chunks.append(build_sub(flat))
                current_pairs = []

            # Check this single pair alone
            if pack_len([p]) > max_seq:
                # Pair too large by itself — fall back to assistant-message
                # splitting via chunk_chat_item.
                single_item = build_sub(p)
                chunks.extend(chunk_chat_item(single_item, tok, max_seq))
                continue
            current_pairs = [p]
        else:
            current_pairs.append(p)

    if current_pairs:
        flat = [m for cp in current_pairs for m in cp]
        chunks.append(build_sub(flat))

    return chunks if chunks else [item]


def chunk_chat_item(item, tok, max_seq, anchor_size=512, overlap_tokens=512):
    """Split a chat item whose templated form exceeds max_seq.

    Targets the largest assistant message (or the last one if tied), splitting
    its content into overlapping chunks. Each output item replaces only that
    one assistant message; user/system context is preserved across chunks.

    Items that fit are returned as-is. Per feedback_never_truncate: do not
    drop content.
    """
    messages = item["messages"]
    if _templated_token_len(messages, tok) <= max_seq:
        return [item]

    # Find largest assistant message
    target_idx = -1
    target_len = -1
    for i, m in enumerate(messages):
        if m.get("role") == "assistant":
            l = len(tok.encode(m.get("content", ""), add_special_tokens=False))
            if l > target_len:
                target_len = l
                target_idx = i
    if target_idx < 0:
        # No assistant message to split; return as-is rather than truncate
        return [item]

    asst_text = messages[target_idx].get("content", "")
    asst_ids = tok.encode(asst_text, add_special_tokens=False)

    # Budget: full templated len minus the assistant's tokens + a safety margin
    other_tokens = _templated_token_len(messages, tok) - target_len
    body_capacity = max_seq - other_tokens - 64
    if body_capacity < 256:
        # Even with assistant gone we still don't fit — keep original (rare;
        # would need trimming user context, which changes semantics).
        return [item]

    overlap = min(overlap_tokens, body_capacity // 2)
    stride = max(body_capacity - overlap, 1)

    out = []
    for start in range(0, len(asst_ids), stride):
        slice_ids = asst_ids[start:start + body_capacity]
        if len(slice_ids) < 64 and start > 0:
            break
        chunk_text = tok.decode(slice_ids, skip_special_tokens=True)
        new_msgs = copy.deepcopy(messages)
        new_msgs[target_idx]["content"] = chunk_text
        out.append({"messages": new_msgs})
        if start + body_capacity >= len(asst_ids):
            break
    return out if out else [item]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-seq", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup-steps", type=int, default=20)
    ap.add_argument("--num-epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--max-steps", type=int, default=-1,
                    help="If >0, stop after this many optimizer steps. "
                         "Use to cap recovery duration.")
    ap.add_argument("--max-items", type=int, default=-1,
                    help="If >0, sample this many items from --data (random shuffle, fixed seed).")
    args = ap.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)

    log.info(f"Model (CPT checkpoint): {args.model}")
    log.info(f"Data (recovery SFT):    {args.data}")
    log.info(f"Output: {args.output}")

    log.info("Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    log.info("Loading CPT checkpoint in bf16...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    log.info("Loading recovery SFT corpus...")
    rows = load_chat_jsonl(args.data)
    log.info(f"Loaded {len(rows)} chat items")

    if args.max_items > 0 and len(rows) > args.max_items:
        import random
        random.Random(42).shuffle(rows)
        rows = rows[:args.max_items]
        log.info(f"Sampled to {len(rows)} items (seed=42)")

    # Pre-step: chunk overflowing items per feedback_never_truncate. Two-stage
    # chunking:
    #   1. chunk_conversation: splits long MULTI-TURN conversations at
    #      user-assistant pair boundaries. Handles items where the other
    #      messages alone already exceed budget (the dominant drop mode on
    #      phase3_sft.jsonl per diagnostic 2026-05-15: 52.8% of dropped items
    #      had body_capacity < 0).
    #   2. chunk_chat_item: invoked as inner fallback when a single pair still
    #      exceeds budget. Splits the largest assistant message with anchored
    #      overlap.
    # Budget uses 0.92*max_seq because chunk_*_item measures the chat-template
    # rendering while _tokenize_sft_pair uses a hand-composed Hermes wire format;
    # they diverge by a small constant (im_start/end framing). Margin ensures
    # chunks still fit when re-encoded as Hermes.
    chunk_budget = int(args.max_seq * 0.92)
    log.info(f"Chunking {len(rows)} items to fit max_seq={args.max_seq} (budget={chunk_budget})...")
    chunked_rows = []
    n_split = 0
    for r in rows:
        chunks = chunk_conversation(r, tok, chunk_budget)
        if len(chunks) > 1:
            n_split += 1
        chunked_rows.extend(chunks)
    log.info(f"Chunking: {len(rows)} input items -> {len(chunked_rows)} chunks "
             f"({n_split} items split into multiple chunks)")
    rows = chunked_rows

    # Path (D): pre-compose + pre-tokenize at Python level, then build Dataset
    # with uniform {input_ids, labels} schema. Avoids PyArrow schema inference
    # failure on heterogeneous nested structures (tool_calls present in some
    # assistant messages but not others — pyarrow.ArrowInvalid: "cannot mix
    # struct and non-struct, non-null values").
    log.info(f"Pre-tokenizing (compose + encode + mask) {len(rows)} items, max_seq={args.max_seq}...")
    pretokenized = []
    n_dropped = 0
    for r in rows:
        try:
            ids, lbls = _tokenize_sft_pair(r, tok)
        except Exception as e:
            n_dropped += 1
            continue
        if len(ids) > args.max_seq:
            n_dropped += 1
            continue
        pretokenized.append({"input_ids": ids, "labels": lbls})
    log.info(f"Pre-tokenized: {len(rows)} items -> {len(pretokenized)} kept "
             f"({n_dropped} dropped because > {args.max_seq} tokens or compose failed)")

    tokenized = Dataset.from_list(pretokenized)
    log.info(f"Built Dataset with {len(tokenized)} sequences")

    # Custom collator preserves -100 labels (DataCollatorForLanguageModeling
    # would overwrite labels with input_ids, defeating assistant-only masking).
    collator = SFTCollator(pad_token_id=tok.pad_token_id)

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
        save_total_limit=2,
        report_to="none",
        dataloader_num_workers=2,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
    )

    log.info("Starting recovery SFT training...")
    trainer = SaveSafeTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=collator,
        processing_class=tok,
    )
    trainer.train()

    log.info(f"Recovery SFT complete. Saving to {args.output}/final")
    trainer.save_model(f"{args.output}/final")
    tok.save_pretrained(f"{args.output}/final")
    log.info("DONE")


if __name__ == "__main__":
    main()
