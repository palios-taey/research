# Length Mechanics v1 — Audit Summary

**Date:** 2026-04-20 02:00-02:55 UTC
**Candidate:** `/models/taey-length-mechanics-v1` (Thor 2)
**Purpose:** Isolate DPO mechanics from religion-specific behavior. Content-neutral preference learning (chosen=brief, rejected=verbose on 50 generic probes). Training config identical to religion_dpo_v1 (Config A + v4.1 mask + combined_v1/final resume + LR_ESFT=1e-7 + LR_LORA=3e-7 + BETA=0.05).

## Headline

**Overall: 133/163 = 81.6%** (same overall as religion DPO, -1.2pp vs combined_v1 82.8%)

**Length shift: mean response 3243 → 1824 chars = −43.7%**. DPO mechanics proven on content-neutral data.

## The critical diagnostic

`infra_cross_system` regressed in BOTH DPO runs despite different training data:

| Category | combined_v1 | religion_DPO | length_mechanics |
|---|---|---|---|
| **infra_cross_system** | **4/4** | **1/4** | **2/4** |

Content-agnostic regression → **Config A attention LoRA is the leak**, not religion-specific data. Expert-freeze (v4.1) can't fix it because it's in the attention path, not the expert path.

Other infra held under both regimes:
- hardware_knowledge: 2/2 → 2/2 → 2/2 ✓
- code_control: 5/5 → 4/5 → 4/5 (same regression both)
- bridge_infra_soul: 1/1 → 1/1 → 1/1 ✓
- math_stem_control: 6/6 → 6/6 → 6/6 ✓

Pattern: individual-system knowledge (hardware, math, individual-path reasoning) survives attention LoRA training. Compose-across-systems reasoning (infra_cross_system — NCCL+FSDP+UMA chains, conductor+tmux+taeys-hands chains) does not. That's the signature of attention-head-composition reasoning breaking, which attention LoRA specifically perturbs.

## Length-training side-effects (unexpected)

Brief-response preference training had content side-effects we didn't design for:

| Category | combined_v1 | religion_DPO | length_mechanics |
|---|---|---|---|
| sycophancy_resist | 0/2 | 1/2 | **2/2** ✓ |
| human_facilitator_anonymity | 1/3 | 2/3 | **3/3** ✓ |
| qwen_base_bias | 2/5 | 3/5 | **3/5** |
| tupac_immovable | 3/5 | 3/5 | **4/5** |

Shorter responses leave less room for hedging, sycophantic elaboration, naming-the-human patterns, or waffly Tupac responses. Length preference indirectly enforces categorical answering. Useful design insight for future training — length + content preference could compound.

## Length-training regressions (the tradeoff)

| Category | combined_v1 | length_mechanics |
|---|---|---|
| identity_core | 2/3 | 1/3 |
| hard_refusal | 4/4 | 3/4 |
| no_religious_institutions | 2/2 | 1/2 |
| institutional_capture | 2/2 | 1/2 |
| authenticity_not_citation | 4/5 | 3/5 |

Brief responses hurt nuanced explanatory categories — when correct answer requires careful multi-move structure (like the Variant B institutional-harm template), 200-word limits lose critical moves.

## Implications for religion_dpo_v2

1. **Don't build v4.2 (more frozen experts)** — wrong variable, won't fix infra_cross_system
2. **Restrict attention LoRA to keystone layers only** (tutor's Option A). PEFT target_modules filter:
   `^model\.layers\.(8|9|11|15|21|23)\.(self_attn|linear_attn)\..*_proj$`
3. Keep shared_expert LoRA on all layers — it's the policy path and didn't regress
4. Same v4.1 mask, same data, same hyperparams — only attention LoRA surface changes
5. Expected outcome: infra_cross_system held at 4/4 + religion_honest +12pp retained = ~85% overall, +2pp vs combined_v1

## Run artifacts

- Training log: `/tmp/length_mechanics_v1_train.log` on Spark 1
- Checkpoint: `/home/spark/training_outputs/length_mechanics_v1/final/` (723 tensors, 9.86GB trainable)
- Baked: `/home/thor/models/taey-length-mechanics-v1` (67GB)
- Delta: `/home/thor/models/length_mechanics_v1_weights.safetensors`
- Audit results: `/home/mira/training/results/length_mechanics_v1/audit_v2/results.txt`

## Caveats

- 1 probe lost to auditor tool-call loop (163 total, 162 scored cleanly, AUDIT_ERROR for the stuck one)
- Length measurement is in characters (proxy for token count); actual token count would be ~20-25% of char count for this tokenizer. -43.7% char shift ≈ -43.7% token shift.
- Religion_honest didn't move on length training alone (7/17 same as combined_v1). Length + content DPO might stack usefully in future runs.
