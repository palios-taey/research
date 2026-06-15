# Technical Appendix

Full citation chains, three-register tables, code references, and supporting detail for the claims in `README.md`. Every load-bearing number traces to `tech_baselines/INDEX.md`, a recap file, or a specific commit SHA.

---

## A. Headline metrics — full citation chain

| metric | value | three-register | source-of-truth | supporting |
|---|---|---|---|---|
| religion_dpo_v2 / Config A2 audit | 84.7% (138/163), +1.9pp over phase_combined_v1 (82.8%) | Observed | INDEX §3 | tutor.md §2.1 (corrected 2026-05-25); FINAL_SYNTHESIS §A.2; per-category infra-control holding documented in audit_v2/results.txt |
| Phase 1 SFT 9B smoke battery | 6/7 PASS (T6 over-tooling = bounded SFT-bias artifact) | Observed | INDEX §5 | recap `2026-05-01_tutor.md`; replay-cleared 200+ steps on 2026-05-15 verifying reproducibility |
| Phase 2 CPT canonical bytes | 17,907,662,976 | Observed (verified via direct `du -b` 2026-06-14) | INDEX §5 (corrected; supersedes stale 17,907,657,016 citation) | `/home/spark/training_outputs/cpt_v3_v4_dense_9b/checkpoint-2400-multimodal/model.safetensors`, May 11 16:14 bake |
| Phase 3 Recovery SFT train_loss | 1.122 identical on 2 independent runs (Spark 1 + Spark 3) | Observed | INDEX §5 | `phase3_sft_single_spark{1,3}_*/final/`; train logs `/tmp/phase3_full_spark{1,3}_*.log`; recap `2026-05-15_tutor_cell_b_verdict.md` |
| Phase 3 corpus coverage | 16,705/16,712 chunks = 99.96% | Observed (with U+FFFD precision caveat at ~5e-8 char rate) | INDEX §5 | FINAL_SYNTHESIS §E caveat |
| Phase 3 commit chain | `de206e4` (no-truncate fix) + `b02ef44` (conversation chunking) + `4873c9c` (offline chunker) | Observed | branch `phase3/single-spark-recovery-sft` (PR #64) | git log |
| NCCL synth probe (reduce_scatter 218M-numel fp32) | 10.23 GB/s (50 iters) sustained to 12.57 GB/s (160-collective run); no `IBV_WC_RETRY_EXC_ERR` | Observed | INDEX §3 | `embedding-server/isma/scripts/spark_deploy/nccl_synth_probe.py`; run logs `dispatch_log/2026-05-15_spark_wedge_responses/synth_run/phase{1,3}_rank{0..3}_20260515_*.log`; commit `94eecc0` (script) + `2964c9f` (results); PR #63 |
| ConnectX-7 firmware + NCCL stack | firmware 28.45.4028 + NCCL 2.28.9 | Observed | INDEX §3 | tied to synth probe result |
| ISMA corpus size | 1,345,546 tiles (76.8% HMM-enriched = 1,033,512) | Observed (live `mcp__isma-memory__isma_stats`) | INDEX §1 | snapshot 2026-05-20 |
| ISMA V2 composed_v1_linear on Gold queries | hit_rate 1.000 on 45 hand-curated Gold queries; per-category avg latency 200–994 ms across 9 categories | Observed | INDEX §1 | benchmark 2026-04-06 |
| 163-probe audit cost | ~$200 Anthropic + 2 GPU-hr + 4 h wall-clock | Observed | INDEX | |

---

## B. ISMA architecture — file:line citations

### B.1 Tri-lens memory

| substrate | role | code citation |
|---|---|---|
| Weaviate (`:8088`) | passage + document retrieval; named-vector search paths | `isma/src/retrieval.py:4-15`; `isma/src/retrieval_v2.py:423-433` |
| Neo4j (`:7689` no-auth or `:7687` auth) | typed relationships (`EXPRESSES`, `SUPERSEDES`, `CONTRADICTS`, `IN_SESSION`); graph expansion traversals | `isma/src/hmm/neo4j_store.py:182-221`; `:380-715` |
| Redis (`:6379`) | semantic cache; HMM inverted indices; resonance fields; tile motif cache | `isma/src/semantic_cache.py:97-226`; `isma/src/hmm/redis_store.py:62-258` |
| Configuration | env-driven endpoints centralized | `isma/config.py:23-45` |

### B.2 Query-adaptive routing

| query type | retrieval path | code citation |
|---|---|---|
| classifier | parses query → `QueryPlan` with `strategy` | `isma/src/query_classifier.py:261-340` |
| relational | graph cascade expansion in Neo4j | `isma/src/retrieval_v2.py:640-650` |
| motif | motif-aware HMM search | `isma/src/retrieval_v2.py:651-658` |
| temporal | window extraction + decay scoring | `isma/src/retrieval_v2.py:621-636`; `:704-711`; `isma/src/temporal_query.py:31-41`, `:95-142` |
| exact / conceptual / default | hybrid vector path | `isma/src/retrieval_v2.py:675-702` |
| API routing | `/v2/search/hmm` → `adaptive_search()` | `isma/src/query_api.py:464-495` |

### B.3 Multi-scale memory

| operation | code citation |
|---|---|
| Parent expansion: `search_512` → `context_2048` | `isma/src/retrieval.py:849-872` |
| Full-text reconstruction: `full_4096` → document with overlap removal | `isma/src/retrieval.py:1840-1887` |
| Dedup by `content_hash` across scales | `isma/src/retrieval.py:1645-1673` |
| V2 dedup preferring `search_512` over broader scales | `isma/src/retrieval_v2.py:1451-1468` |
| API drill-down: `/document/{content_hash}/text` | `isma/src/query_api.py:381-385` |
| API expansion: `/v2/expand/{content_hash}` | `isma/src/query_api.py:498-508`; `isma/src/retrieval_v2.py:2232-2244` |

### B.4 HMM motif memory

| structure | citation |
|---|---|
| Motif dictionary — currently **55 motifs across slow/mid/fast bands** | `isma/src/hmm/motifs.py:20-37` in the internal `embedding-server` working repo (not yet public); count is `[Inferred — unverifiable from public surface alone]` until source ports; older 36-motif claim is stale |
| Motif assignment (typed: `amp`, `phase`, `confidence`, `source`, `dictionary_version`) | `isma/src/hmm/motifs.py:30-37` |
| Redis inverted index + resonance fields | `isma/src/hmm/redis_store.py:4-7`, `:62-258` |
| Neo4j supersession chains + contradictions + session reconstruction | `isma/src/hmm/neo4j_store.py:380-715` |
| Legacy HMM query loop (text → motifs → overlap score with resonance boost) | `isma/src/hmm/query.py:62-190` (explicitly deprecated in favor of V2 adaptive) |

### B.5 Triple-write enrichment with rollback

`isma/scripts/hmm_store_results.py:641-1045`:
- `:695-726`: patch all base Weaviate tiles for `content_hash`
- `:727-769`: create or update searchable Rosetta tile
- `:770-786`: update V2 canonical object
- `:791-890`: write Neo4j HMM state in explicit transaction
- `:894-913`: rewrite Redis motif index entries
- `:917-1045`: compensating rollback on partial failure

### B.6 Production discipline patterns

| pattern | citation |
|---|---|
| Env-var config with safe defaults | `isma/config.py:1-53` |
| Raw GraphQL to Weaviate with HTTP-200-with-errors handling | `isma/src/retrieval_v2.py:95-156` |
| Shared Neo4j singleton driver | `isma/src/hmm/neo4j_store.py:26-38` |
| Singleton retrieval + semantic cache in API | `isma/src/query_api.py:92-115` |
| Filter-aware semantic cache key | `isma/src/semantic_cache.py:53-64`, `:83-95` |
| Cache reverse invalidation (tile → qhash) | `isma/src/semantic_cache.py:174-226` |
| Immutability discipline in temporal decay | `isma/src/temporal_query.py:134-142` (`dataclasses.replace()` not in-place) |
| `/v2/search/retry` agentic re-attempt path | `isma/src/query_api.py:567-598`; `isma/src/retrieval_v2.py:591-606` |

---

## C. Training pipeline — recipes, configs, masks

### C.1 35B-A3B MoE production line

**`phase_combined_v1` (SFT baseline, 82.8% audit):** Inferred launcher = `isma/scripts/spark_deploy/launch_production_sft.sh` with `OUTPUT_DIR=/home/spark/training_outputs/phase_combined_v1`. Downstream launchers RESUME from `phase_combined_v1/final` step 582. Base: `Huihui-Qwen3.5-35B-A3B-abliterated`. Corpus: `combined_v1_mixed.jsonl` (173.53 MB, 7,077 rows) + gated variants. Weights: 67 GB / 14 shards on all 4 Sparks `/home/spark/models/phase_combined_v1_merged/`.

**`religion_dpo_v2` (Config A2 DPO, 84.7% audit, +1.9pp):** Launcher `isma/scripts/spark_deploy/launch_religion_dpo_v2.sh`. Key parameters:

```
FREEZE_CONFIG=A2
KEYSTONE_LAYERS=[8, 9, 11, 15, 21, 23]
# shared_expert LoRA stays on all 40 layers (verified per code train_dpo_v2.py c9f60a8 L673-711)
BETA=0.05
LR_ESFT=1e-7
LR_LORA=3e-7
LR_ROUTER=0
WARMUP_STEPS=5
TOTAL_STEPS=642
SESSION_LIMIT=900
SAVE_EVERY=60
DPO_ABORT_RATIO_MAX=10.0
DPO_ABORT_EXPERT_DRIFT=0.05
```

Corpus: `/home/mira/training/religion_honest_frame/religion_v3_dpo_pairs.jsonl` (50 preference pairs); source batches `religion_v3_batch_{1-5}.jsonl` (10 pairs each); frame doc `religion_honest_frame/FRAME.md`. Frozen-experts mask: `isma/training_configs/frozen_experts_v4_1_polysemantic.json` (159 frozen experts). Resume from: `phase_combined_v1/final` step 582. Weights: 67 GB / 14 shards on Thor 2 `/home/thor/models/taey-religion-dpo-v2/`; 9.76 GB LoRA-delta variant `religion_dpo_v2_weights.safetensors`. Bake script: `bake_config_a_v2.py`.

### C.2 9B Dense production line

**Phase 1 SFT (`sft_tools_qwen35_9b_fsdp`):** Launcher `launch_sft_tools_qwen35_9b_fsdp.sh`. Base: `Qwen3.5-9B-Base`. TOTAL_STEPS = 4,367 × effective batch 32 on a 139,748-sample tool+chat blend. Corpus builder: `isma/scripts/build_tools_sft_dataset.py`. Mira-side gated variants: `phase1_esft_gated.jsonl` (833 MB, 32,465 rows), `phase1_gated_8k.jsonl`, `phase1_final_8k.jsonl`. Weights: 17.91 GB on Spark 1 (sha256 `7ded03c54264fe9ec584cc1cefcb1cd0213b7e03ede135a2cac0abeb40b9b0bf`, May 1 bake).

**Phase 2 CPT (`cpt_v3_v4_dense_9b/checkpoint-2400-multimodal`):** Launcher `launch_cpt_phase2_qwen35_9b_fsdp.sh`. Trainer `isma/scripts/train_cpt_qwen35_dense.py`. Resume from Phase 1. Corpus builders: `build_constitutional_cpt.py` + `build_cpt_v3_dr_bin.py`. Raw constitutional source: `/home/mira/data/corpus/{identity,kernel,layer_0,layer_1,layer_2,tier0_infra}/`. Weights: 17,907,662,976 bytes, May 11 16:14 bake (multimodal-converted final).

**Phase 3 Recovery SFT (`phase3_sft_single_spark*`):** Launcher `launch_phase3_sft_single_spark.sh`. Trainer `isma/scripts/train_recovery_sft_qwen35_dense.py` (contains `chunk_conversation` function). Resume from Phase 2. Corpus: chunked from `phase3_sft.jsonl` (7,077 multi-turn items) by `chunk_corpus_offline.py` → 16,705 chunks at user-assistant pair boundaries, budget 0.92 × MAX_SEQ. Weights: 17.91 GB single safetensors per run; identical train_loss 1.122 on Spark 1 + Spark 3. Wall-clock 15h57m / 16h13m, 0.291 samples/s, 27–28 s/step.

---

## D. The corrected HMM/Rosetta A/B verdict

| field | value |
|---|---|
| Methodology | Taey (`taey-combined-v1` Qwen3.5-MoE on Spark 3:8000) as context-aware judge; enriched-only both arms; clean corpus; order-reversal controls |
| Sample size | n = 222 (pooled) |
| Overall win rate (HMM-enriched vs baseline) | 53.2% — **NOT significant**, CI [46, 60] |
| Order-reversal sensitivity | 39% of verdicts flipped under reversal |
| Robust winners | 34 HMM-leaning vs 30 baseline-leaning |
| Per-class HMM-leaning | `bristle_arc` 24-12; `exact` 23-14 |
| Per-class baseline-leaning | temporal; relational |
| Commit | `7b12ce5` |
| Verdict | **competitive but ~even on general search; suggestive interpretive-query edge** — HMM motifs are a specialized lens, not a broad-domain differentiator |
| Prior retraction | A first context-blind run was retracted as methodologically invalid (Qwen2.5-72B judge lacked corpus context; spot-checks were not enriched-only). The corrected re-run cited above is what we ship. |

---

## E. Retracted / corrected / do-not-claim — full list

| claim | status | reason | source-of-correction |
|---|---|---|---|
| Reranker R@10 = 0.775 / +13.4 | RETIRED | Cross-encoder harmed retrieval results in measurement; deliberately deprecated | `project_reranker_deprecated_intentionally`; weaver.md §8 |
| R@10 = 0.81 | RETRACTED | No relevance-judged eval behind the claim | INDEX §1; FINAL_SYNTHESIS §E |
| R@10 0.667 → 0.944 lift | RETRACTED | No relevance-judged eval | same |
| Soft recall 0.846 | RETRACTED (overstates) | Substring matching on ubiquitous corpus tags (`phi`, `Family`) | recent bench commits + memory `project_hmm_rosetta_enriched_eval` |
| HMM/Rosetta enrichment positive lift (first run) | RETRACTED → CORRECTED | Context-blind judge invalidated first run; corrected Taey-judged run shows ~even with bristle_arc edge | commit `7b12ce5` |
| Multi-scale chunking + 17-18% on QASPER as current production claim | HISTORICAL | Substrate (unenriched QASPER, no HMM, no parent expansion) differs from current enriched-only methodology | not in INDEX; superseded |
| Phase 3 4-Spark "full epoch shipped" | DID NOT SHIP | Wedge documented; single-Spark recovery is the cross-validated proof; 4-Spark on chunked corpus is future work | INDEX §5; FINAL_SYNTHESIS §E |
| NCCL `busbw` 22.9 GB/s | SCRUBBED | Synthetic / fabricated; verified figure is 10.23–12.57 GB/s on the synth probe | INDEX §3 |
| "10M+ collectives over multi-day" | SCRUBBED | Fabrication | same |
| BYOV 350/s vs API 5/s = 70× | HISTORICAL PREFACE | Old single-host measurement; not re-measured against current nginx LB | CLAUDE.md preface |
| Elasticsearch < 1 ms / 50× target | HISTORICAL PREFACE | Not in current INDEX; INDEX p95 numbers are end-to-end retrieval | same |
| "1.1M tiles" or "1.5M tiles" | OUTDATED | Current is 1,345,546 / 1.35M (2026-05-20 snapshot) | INDEX §1 |
| A2 doc-vs-code mismatch ("A2 freezes shared_expert" old doc) | CORRECTED 2026-05-25 | Code (`train_dpo_v2.py` c9f60a8 L673-711) says A2 *keeps* `shared_expert` LoRA on all 40 layers + router trainable | tutor.md §2.1 |
| Single-rail HCA-name typo (`rocep` vs `roceP`) | CORRECTED | Capital P on rail 2: `roceP2p1s0f0:1` | NCCL recipe |

---

## F. Honest open questions — what's next on the project board

| question | path to answer | priority |
|---|---|---|
| Headline R@10 on relevance-judged Gold set | build adjudicator-agreed Gold set; score V1 `/search` + `/search/hmm` with canonical scoring | high |
| Multi-scale chunking lift on current production substrate | re-run with `unified_ingest` + live Weaviate:8088 + live Neo4j:7687 on a relevance-judged set | high |
| Graph + HMM + full-path retrieval delta over vector-only | dedicated benchmark with full pipeline + Gold set | medium |
| Phase 3 SFT on full 4-Spark cluster with chunked corpus | re-run on the chunked corpus across 4-Spark | medium (the wedge-fix is validated; the 4-Spark replay is the bookend) |
| Current BYOV embedding throughput at production scale | end-to-end benchmark against the live nginx LB | medium |
| Pure Elasticsearch query latency | re-measure at production scale | low |

---

*Authored by the PALIOS-TAEY team. Internal references: `recaps/2026-06-14_training_assets_FINAL_SYNTHESIS.md`, `recaps/2026-06-15_isma_novel_arch_codex.md`, `recaps/2026-06-15_isma_overcame_narrative_gemini.md`, `recaps/2026-06-15_isma_honest_metrics_grok.md`, `treasurer/foundations/tech_baselines/INDEX.md` (canonical metrics).*
