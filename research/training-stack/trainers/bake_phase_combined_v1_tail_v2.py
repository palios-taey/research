#!/usr/bin/env python3
"""Bake Phase 1 v3 merged model from base + trainable delta.

Template from Infra (Thor 1 bake workflow, 2026-04-17). Adapted for Spark paths.

Inputs:
- Base:    /home/spark/models/Huihui-Qwen3.5-35B-A3B-abliterated
- Delta:   /home/spark/training_outputs/phase_combined_v1_tail_v2/final/trainable_weights.safetensors
           (12 tensors — 6 keystone layers × 2 projections, PEFT-prefixed keys)

Output:
- /home/spark/models/phase_combined_v1_tail_v2_merged
  * All unchanged shards hardlinked from base (save disk)
  * 7 shards containing keystone-layer expert weights rewritten with merged tensors
  * config.json, tokenizer, chat_template, etc. hardlinked through

Runtime: ~3 min, no GPU, pure file rewrite.

PEFT key strip: base_model.model.model.X → model.language_model.X
(verified: abliterated base uses model.language_model.* prefix)
"""
import json
import os
import shutil
import sys
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

BASE = Path("/home/spark/models/Huihui-Qwen3.5-35B-A3B-abliterated")
TRAINED = Path("/home/spark/training_outputs/phase_combined_v1_tail_v2/final/trainable_weights.safetensors")
OUT = Path("/home/spark/models/phase_combined_v1_tail_v2_merged")
KEYSTONE = [8, 9, 11, 15, 21, 23]


def main():
    if not BASE.exists():
        sys.exit(f"Base not found: {BASE}")
    if not TRAINED.exists():
        sys.exit(f"Delta not found: {TRAINED}")

    OUT.mkdir(parents=True, exist_ok=True)
    print(f"BAKE: {TRAINED.name} + {BASE.name} → {OUT}")

    # Load trained weights, strip PEFT prefix
    trained = {}
    with safe_open(str(TRAINED), framework="pt") as f:
        for k in f.keys():
            if k.startswith("base_model.model."):
                # base_model.model.model.X  →  model.language_model.X
                base_key = "model.language_model." + k[len("base_model.model.model."):]
            else:
                base_key = k
            trained[base_key] = f.get_tensor(k)
    print(f"  Loaded {len(trained)} delta tensors")

    # Find shards containing keystone experts
    idx = json.load(open(BASE / "model.safetensors.index.json"))
    shards_to_rewrite = set()
    target_keys_per_shard: dict[str, list[str]] = {}
    for layer in KEYSTONE:
        for proj in ("gate_up_proj", "down_proj"):
            k = f"model.language_model.layers.{layer}.mlp.experts.{proj}"
            if k not in idx["weight_map"]:
                sys.exit(f"Expected key missing from base index: {k}")
            s = idx["weight_map"][k]
            shards_to_rewrite.add(s)
            target_keys_per_shard.setdefault(s, []).append(k)
    print(f"  Shards to rewrite: {len(shards_to_rewrite)}")
    for s in sorted(shards_to_rewrite):
        print(f"    {s}: {len(target_keys_per_shard[s])} keystone tensors")

    # Hardlink unchanged files; copy across filesystem as fallback
    hardlinked = 0
    copied = 0
    for f in BASE.iterdir():
        if not f.is_file() or f.name in shards_to_rewrite:
            continue
        dst = OUT / f.name
        if dst.exists():
            dst.unlink()
        try:
            os.link(f, dst)
            hardlinked += 1
        except OSError:
            shutil.copy2(f, dst)
            copied += 1
    print(f"  Unchanged files: {hardlinked} hardlinked, {copied} copied")

    # Rewrite target shards with merged weights
    for shard_name in sorted(shards_to_rewrite):
        print(f"  Rewriting {shard_name} ...", flush=True)
        tensors = {}
        target_keys = set(target_keys_per_shard[shard_name])
        replaced = 0
        with safe_open(str(BASE / shard_name), framework="pt") as f:
            for k in f.keys():
                if k in target_keys and k in trained:
                    trained_t = trained[k]
                    base_t = f.get_tensor(k)
                    if trained_t.shape != base_t.shape:
                        sys.exit(f"Shape mismatch for {k}: trained={trained_t.shape} base={base_t.shape}")
                    tensors[k] = trained_t.contiguous()
                    replaced += 1
                else:
                    tensors[k] = f.get_tensor(k)
        dst = OUT / shard_name
        if dst.exists():
            dst.unlink()
        save_file(tensors, str(dst), metadata={"format": "pt"})
        print(f"    replaced {replaced}/{len(target_keys)} target tensors")

    print(f"BAKE COMPLETE: {OUT}")


if __name__ == "__main__":
    main()
