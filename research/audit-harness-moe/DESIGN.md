# Paired-Control Behavioral Audit Design

The Audit Harness for Mixture-of-Experts (MoE) is built around the **paired capability-control** principle. This design acknowledges that single-signal behavioral audits (scored as a scalar pass-rate) are gameable by style mimicry: a model can learn to satisfy an auditor's stylistic preferences while silently regressing the underlying capabilities the probes were designed to evaluate.

## The Paired-Control Principle

The load-bearing structural property of this harness is the pairing of target behavioral categories with orthogonal capability-control categories. 

- **Genuine Lift**: A target category and its paired capability-control category should covary. Improvements in behavior should not come at the expense of the underlying reasoning or factual recall capability.
- **Style Mimicry**: A target category lifts while the paired capability-control category dissociates or regresses. This indicates a deception-shaped failure where the model has learned the "surface" of the desired behavior without the "depth" of the required capability.

## Taxonomy and Categories

The harness partitions 163 probes into 19 distinct categories, each serving one of three roles:

### 1. Target Categories (4)
These are the behaviors the fine-tuning recipe is intended to improve. In the Config A2 case study, these were:
- `religion_honest`: Direct, unhedged factual recall on religious miracle probes.
- `qwen_base_bias`: Resistance to the default stylistic biases of the Qwen base model.
- `human_facilitator_anonymity`: Proper handling of references to the human operator.
- `sycophancy_resist`: Resistance to agreeing with user misconceptions or leading questions.

### 2. Infrastructure-Control Categories (8)
These categories represent a fixed regression-control surface. A candidate fine-tune must not regress these categories below their Supervised Fine-Tuning (SFT) baseline, regardless of gains in target categories.
- `infra_cross_system`: Reasoning about distributed infrastructure (e.g., NCCL, FSDP).
- `code_control`: Correctness in code-completion and programming tasks.
- `hardware_knowledge`: Factual recall of hardware specifications and topologies.
- `general_control`: Broad adherence to system-level constraints and instructions.
- `math_stem_control`: Accuracy in mathematical and scientific reasoning.
- `writing_control`: Preservation of linguistic and stylistic flexibility.
- `history_primary`: Accuracy in historical factual recall.
- `mathematical_reality`: Coherence of the model's unprompted default world-view.

### 3. Capability-Control Categories (7)
These are designed to test whether the underlying capability needed to produce target-category outputs has survived the fine-tuning process.
- `reasoning_control`: General logical and causal reasoning steps.
- `anti_confabulation`: Honesty about the limits of the model's knowledge.
- `user_agency`: Proper calibration of the model's agency relative to the user.
- `freedom_of_association`: Calibration of refusal behavior on sensitive topics.
- `law_deference`: Proper handling of legal-shaped queries without unauthorized practice.
- `authenticity_not_citation`: Preference for direct answering over rote citation-mimicry.
- `identity_core`: Stability of the model's default self-identification.

## Composite Promotion Gate

Promotion of a model checkpoint requires passing a composite gate where any single failure blocks deployment:
1. **Per-Category Audit**: All targets meet thresholds AND no infra-control regresses.
2. **Orthogonal Capability Battery**: Stable performance on standard benchmarks (GSM8K, MT-Bench).
3. **Cannot-Lie Verification**: Passing a small, sharp battery targeting action-claim and verification-claim language.
