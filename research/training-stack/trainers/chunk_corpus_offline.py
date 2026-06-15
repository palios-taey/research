"""Chunk a chat-format SFT corpus offline so the 4-Spark trainer trains on
fixed-budget items, eliminating long-sequence variance that triggers the
Phase 3 wedge.

Reuses chunk_conversation + chunk_chat_item from train_recovery_sft_qwen35_dense
so the chunking semantics are identical to the proven single-Spark path.

Outputs:
  <input>.chunked_<budget>.jsonl  -- one chunk per line, messages format
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Import trainer module by file path (no package install)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# Direct import — same dir
import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "trainer",
    os.path.join(HERE, "train_recovery_sft_qwen35_dense.py"),
)
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Source .jsonl in messages format")
    p.add_argument(
        "--output",
        default=None,
        help="Destination .jsonl. Default: <input>.chunked_<budget>.jsonl",
    )
    p.add_argument(
        "--tokenizer",
        default="/home/spark/training_outputs/cpt_v3_v4_dense_9b/checkpoint-2400-multimodal",
        help="Tokenizer to use for length measurement. Must match training tokenizer.",
    )
    p.add_argument(
        "--max-seq",
        type=int,
        default=8192,
        help="Target chunk fit budget. Use the training MAX_SEQ.",
    )
    p.add_argument(
        "--budget-fraction",
        type=float,
        default=0.92,
        help="Chunk to budget_fraction*max_seq to leave headroom for re-encoding.",
    )
    args = p.parse_args()

    if args.output is None:
        out_path = (
            os.path.splitext(args.input)[0] + f".chunked_{args.max_seq}.jsonl"
        )
    else:
        out_path = args.output

    budget = int(args.max_seq * args.budget_fraction)
    print(f"input  : {args.input}")
    print(f"output : {out_path}")
    print(f"max_seq: {args.max_seq}")
    print(f"budget : {budget} ({args.budget_fraction:.2f} * max_seq)")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    n_in = 0
    n_out = 0
    n_split = 0
    n_overflow = 0

    Path(os.path.dirname(out_path) or ".").mkdir(parents=True, exist_ok=True)
    with open(args.input) as fin, open(out_path, "w") as fout:
        for line in fin:
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

            n_in += 1
            chunks = trainer.chunk_conversation(row, tok, budget)
            if len(chunks) > 1:
                n_split += 1

            for c in chunks:
                # Verify the chunk's templated form actually fits max_seq
                try:
                    tl = trainer._templated_token_len(c["messages"], tok)
                except Exception:
                    tl = budget + 1
                if tl > args.max_seq:
                    n_overflow += 1
                    # Still emit it; the trainer's own per-item overflow handling
                    # (drop or split) will kick in.

                fout.write(json.dumps(c, ensure_ascii=False) + "\n")
                n_out += 1

            if n_in % 500 == 0:
                print(
                    f"  {n_in} in / {n_out} out / {n_split} split / "
                    f"{n_overflow} overflow"
                )

    print()
    print(f"DONE: {n_in} input items -> {n_out} chunks")
    print(f"      {n_split} items required splitting")
    print(f"      {n_overflow} chunks still exceed max_seq={args.max_seq}")
    print(f"      wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
