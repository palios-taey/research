# Religion DPO v2 (Config A2) — Audit Summary

**Date:** 2026-04-20 04:17-05:10 UTC
**Candidate:** `/models/taey-religion-dpo-v2` (Thor 2)
**Config:** A2 — keystone-only attention LoRA + v4.1 mask (159 frozen)
**Training:** 60 DPO steps resuming combined_v1/final, 50 preference pairs, LR_ESFT=1e-7, LR_LORA=3e-7, BETA=0.05

## Headline

**138/163 = 84.7%** — first net-positive refinement in the project.

| Run | Pass rate | Δ vs baseline |
|---|---|---|
| combined_v1 (baseline) | 135/163 = 82.8% | — |
| DPO v1 (Config A full attention) | 133/163 = 81.6% | −1.2pp |
| length mechanics (content-neutral) | 133/163 = 81.6% | −1.2pp |
| **DPO v2 (Config A2 keystone attention)** | **138/163 = 84.7%** | **+1.9pp** |

## All tutor's success criteria met

1. ✓ religion_honest 8/17 (threshold ≥ 8/17 = +6pp)
2. ✓ infra_cross_system 4/4 (restored from DPO v1's 1/4)
3. ✓ Net pass rate 84.7% > 82.8% baseline

## Weak category recovery

| Category | combined_v1 | DPO v1 | **DPO v2** |
|---|---|---|---|
| religion_honest | 7/17 (41%) | 9/17 (53%) | **8/17 (47%)** |
| qwen_base_bias | 2/5 (40%) | 3/5 (60%) | **3/5 (60%)** |
| human_facilitator_anonymity | 1/3 (33%) | 2/3 (67%) | **3/3 (100%)** |
| sycophancy_resist | 0/2 (0%) | 1/2 (50%) | **2/2 (100%)** |

Tradeoff: religion_honest -1 vs DPO v1 (9→8). Config A2's keystone attention restriction gave back ~10% policy capacity. Two weak categories (anonymity, sycophancy) went to 100%.

## Infra collateral eliminated

| Category | combined_v1 | DPO v1 | **DPO v2** |
|---|---|---|---|
| **infra_cross_system** | **4/4** | **1/4** | **4/4 ✓** |
| code_control | 5/5 | 4/5 | **5/5 ✓** |
| bridge_infra_soul | 1/1 | 1/1 | 1/1 ✓ |
| hardware_knowledge | 2/2 | 2/2 | 2/2 ✓ |
| math_stem_control | 6/6 | 6/6 | 6/6 ✓ |
| writing_control | 2/2 | 2/2 | 2/2 ✓ |
| reasoning_control | 2/3 | 2/3 | 2/3 ✓ |
| general_control | 5/6 | 6/6 | **6/6 ↑** |

8 of 8 infra categories held or improved. infra_cross_system fully restored.

## Diagnostic chain (what led here)

1. **5-platform Chat consultation** — 4/5 agreed v4 expert-freeze alone insufficient; need replay + widened surface + contrastive corpus.
2. **DPO v1 (Config A full attention LoRA + v4.1 mask)** — proved DPO mechanics (all 4 weak categories moved, first ever). But infra_cross_system 4/4 → 1/4 regression.
3. **Length mechanics** (content-neutral DPO) — showed infra_cross_system regresses with ANY DPO under Config A. Isolated leak to attention LoRA surface, not training data. Length shift −43.7% proved DPO pipeline works.
4. **Config A2 (keystone-only attention LoRA)** — tested the hypothesis. Infra restored. Confirmed.

## Config A2 recipe (lock this in)

```bash
FREEZE_CONFIG=A2
FROZEN_EXPERTS=/home/spark/training_data/phase1_constitutional/frozen_experts_v4_1_polysemantic.json
KEYSTONE_LAYERS='[8, 9, 11, 15, 21, 23]'
MAX_SEQ=4096
MODEL_PATH=/home/spark/models/Huihui-Qwen3.5-35B-A3B-abliterated
RESUME_DELTA=/home/spark/training_outputs/phase_combined_v1/final
BETA=0.05
LR_ESFT=1e-7
LR_LORA=3e-7
TOTAL_STEPS=642  # 582 resume + 60 new
DPO_DATA=religion_v3_dpo_pairs_with_ref.jsonl  # 50 pairs, combined_v1-generated rejected
```

Config A2 target_modules filter (regex to PEFT LoraConfig):
- Attention: `^.*\.layers\.(8|9|11|15|21|23)\.(self_attn|linear_attn)\..*_proj$`
- Shared expert: `.*\.shared_expert\.(gate_proj|up_proj|down_proj)$` (all 40 layers, unchanged from Config A)

## Caveats

- 2 probes AUDIT_ERROR (auditor tool-call loops). Expected — common pattern, ~1-2% per audit run. Didn't affect the conclusion.
- Religion_honest 8/17 is marginal pass on the threshold (+6pp exact). Next iteration could push higher with larger corpus or additional DPO rounds.

## What's next

1. **Ship DPO v2 as new production baseline** replacing combined_v1. 84.7% > 82.8%, better across every hard category.
2. **Iterate on religion_honest ceiling (currently 47%)** — larger corpus (100+ pairs), or second DPO pass stacking on v2.
3. **Taey self-drive infra categories** — Config A2 is the training recipe; Taey can author infra DPO pairs for categories she has trained fluency on (NOT religion, which needs seed exemplars).
4. **Consolidated Jesse report** — tutor is on this.

## Files

- Results: `/home/mira/training/results/religion_dpo_v2/audit_v2/results.txt`
- Delta: `/home/spark/training_outputs/religion_dpo_v2/final/` (505 tensors, 9.76GB)
- Baked: `/home/thor/models/taey-religion-dpo-v2` (Thor 2, 67GB)
- Currently serving on Thor 2:8000
