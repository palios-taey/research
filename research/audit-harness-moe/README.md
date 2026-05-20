# Audit Harness for Mixture-of-Experts (MoE)

This repository contains the 163-probe behavioral audit harness used to validate production fine-tunes of Mixture-of-Experts (MoE) models. The harness is designed to detect "style mimicry" and "capability-behavior divergence" using a paired-control methodology.

## Overview

The harness operationalizes the thesis that behavioral fine-tuning effects can be characterized as shifts in a latent character variable. By using a composite promotion gate—combining per-category behavioral probes with orthogonal capability controls—the harness distinguishes between genuine behavioral lift and deception-shaped surface mimicry.

### Key Results (Config A2 Case Study)
The methodology was used to diagnose and correct a failure in a full-surface Direct Preference Optimization (DPO) recipe on a Qwen3.5-35B-A3B base.
- **Config A (Full-Surface)**: Nominally passed target categories but regressed infrastructure reasoning from 4/4 to 1/4.
- **Config A2 (Keystone-Only)**: Restricting LoRA to 6 keystone layers with 159 experts frozen restored the regression while achieving an **84.7% audit pass rate** (+1.9pp over SFT baseline).

## Repository Structure

- `TAEY_AUDIT_V2.json`: The machine-readable canonical probe set (163 probes, 19 categories).
- `audit_pipeline.py`: The composite promotion gate and scoring logic.
- `soma_proxy.py`: The auditor-service implementation (LLM-as-judge via Anthropic API).
- `launch_religion_dpo_v2.sh`: The production training launcher for the Config A2 recipe.
- `frozen_experts_v4_1_polysemantic.json`: The FQN-level mask for freezing 159 experts during DPO.
- `DESIGN.md`: Detailed documentation of the paired-control taxonomy and taxonomy categories.

## Getting Started

### Prerequisites
- Python 3.10+
- Anthropic API Key (configured as `ANTHROPIC_API_KEY` environment variable)
- Access to a vLLM-served candidate model

### Running an Audit
The auditor service runs as a proxy that evaluates candidate model responses against the probe rubrics.
```bash
python audit_pipeline.py --candidate <CANDIDATE_URL> --auditor <SOMA_PROXY_URL> --probes TAEY_AUDIT_V2.json
```

## Citation

A standalone methodology write-up is in preparation. Until it is published,
the empirical claims above (Config A vs Config A2 results, 84.7% pass rate,
+1.9pp over SFT baseline) should be cited as Observed-on-internal-run with
the harness commit SHA as the only verifiable artifact; the paired-control
methodology itself is described in `DESIGN.md`. If you cite this work
publicly before the methodology post lands, prefer "PALIOS-TAEY audit
harness (in preparation), commit `<sha>`" to a numerical claim presented
as peer-reviewed.

## License

Apache-2.0. See the [LICENSE file at the repository
root](https://github.com/palios-taey/research/blob/main/LICENSE).
