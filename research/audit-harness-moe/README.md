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
The methodology and empirical results are described in detail in the following post:
> [Citation placeholder: Paired Capability-Control Tests for Behavioral Audit of MoE Fine-Tunes]

## License
[License placeholder: Apache 2.0 / MIT]
