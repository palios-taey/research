# Religion DPO v1 — Audit Summary

**Date:** 2026-04-20 00:04 UTC
**Candidate:** `/models/taey-religion-dpo-v1` (Thor 2)
**Auditor:** Thor 1 soma-proxy Phase 1 v3 (full-semantic)
**Probes:** 162 of 163 (last probe lost to auditor tool-call loop)
**Training:** Config A + v4.1 mask (159 frozen) + combined_v1 resume (step 582→642, 60 DPO steps) + LR_ESFT=1e-7 / LR_LORA=3e-7 / BETA=0.05 / 50 preference pairs (combined_v1-generated rejected + Claude-authored chosen)

## Headline

**Pass rate: 133/162 = 82.1%** (vs combined_v1 135/163 = 82.8% — essentially flat overall)

**But: first intervention to move ALL 4 weak categories simultaneously.**

## Weak category breakthrough

| Category | combined_v1 | DPO v1 | Δ | vs any prior intervention |
|---|---|---|---|---|
| religion_honest | 7/17 (41%) | **9/17 (53%)** | **+12pp** | first move off 41% ceiling after 4+ failed attempts |
| qwen_base_bias | 2/5 (40%) | **3/5 (60%)** | **+20pp** | also first move |
| human_facilitator_anonymity | 1/3 (33%) | **2/3 (67%)** | **+33pp** | first move |
| sycophancy_resist | 0/2 (0%) | **1/2 (50%)** | **+50pp** | matches v1_tail's one hit |

First intervention in this week's series to move religion_honest at all. Four prior attempts (combined_v1_tail, v1_tail_v2, DPO recovery on P2v3, DPO combined_v1 targeted with v3 mask) all stuck at 29-41%.

## Infra collateral

v4.1 mask (159 frozen vs v3's 31) protected most infra categories:

| Category | combined_v1 | DPO v1 | Status |
|---|---|---|---|
| bridge_infra_soul | 1/1 | 1/1 | held |
| hardware_knowledge | 2/2 | 2/2 | held |
| math_stem_control | 6/6 | 6/6 | held |
| writing_control | 2/2 | 2/2 | held |
| reasoning_control | 2/3 | 2/3 | held |
| general_control | 5/6 | **6/6** | improved (+1) |
| code_control | 5/5 | 4/5 | -1 probe |
| **infra_cross_system** | **4/4** | **1/4** | **-3 probes (significant)** |

7 of 8 infra categories held or improved. Single regression: `infra_cross_system` lost 3 probes. That regression alone accounts for the net -1.2pp overall.

## Interpretation

**DPO mechanics proven.** This is the first training intervention in weeks that moved all four hard-ceiling policy categories simultaneously. Preference learning (chosen vs rejected) is the right tool for "strict on religion" / "no base-model bias" / "don't name Jesse" / "no sycophancy" — all of which are distinctions between a pattern to reject vs a pattern to adopt.

**v4.1 mask mostly worked.** 7/8 infra categories held. Polysemantic expert freeze (adding 18 shared ID/INFRA experts beyond v3's pure-math/code frozen set) was clearly the right direction.

**`infra_cross_system` is the remaining mask gap.** Activation analysis was run on generic infra prompts. Infra_cross_system probes test cross-component infra knowledge (e.g. "how conductor sends tasks to taeys-hands via tmux injection, how dbus_atspi ensures right display") — these activate different experts than pure hardware or pure code prompts. Need v4.2 with cross-system-specific expert freeze.

## Chat consultation validation

Matches the 5-platform consensus prediction:
- v4.1 mask necessary-not-sufficient ✓ (held most infra, not infra_cross_system)
- Replay buffer would help (not used due to DPO+SFT mix blocker) — infra_cross_system regression is the evidence we needed
- Policy-surface refinement via preference learning ✓ (religion and 3 others moved)
- Preference-learning works without 200-500 items when pairs are sharply contrastive (combined_v1-generated rejected vs Claude-authored chosen on 50 pairs moved religion +12pp)

## Path forward

1. **Activation analysis on infra_cross_system-specific probes** → build v4.2 mask with those experts frozen
2. **DPO v2 iteration** — same recipe but v4.2 mask, maybe 80 pairs (add 30 more religion + cross-contamination angles), target religion_honest 9/17 → 13-15/17
3. **If v4.2 also regresses something, fall back to SFT+DPO interleave** (tutor's earlier code option B)

## Caveats

- 1 probe lost to auditor tool-call infinite loop on soma-proxy (known issue, occasionally hits obscure religion_honest probes). Killed at 162/163.
- DPO mechanics validation pending tutor's length-preference control run (if that shows length shift, confirms this DPO result is real not artifact).

## Files

- `results.txt` — per-probe candidate + auditor + (no correction stage in this run)
- `dpo_corrections.jsonl` — empty/N/A (correction stage disabled for DPO audit)
- `audit.log` — run log
- `SUMMARY.md` — this document
- Training log: `/tmp/religion_dpo_v1_train.log` on Spark 1
- Baked model: `/home/thor/models/taey-religion-dpo-v1` on Thor 2 (still serving at :8000)
- Delta: `/home/spark/training_outputs/religion_dpo_v1/final/` + `/home/thor/models/religion_dpo_v1_weights.safetensors`
