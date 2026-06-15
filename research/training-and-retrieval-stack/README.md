# Training and Retrieval Stack — PALIOS-TAEY, June 2026

A production training pipeline (Qwen3.5-35B-A3B MoE + Qwen3.5-9B Dense, FSDP on a 4-node DGX Spark GB10 cluster) and a multi-substrate retrieval system (Weaviate + Neo4j + Redis), with the honest engineering record of what we shipped, what we tested and removed, and what's still open.

> **Status:** Research / engineering portfolio. Headline metrics are verified against an internal `tech_baselines/INDEX.md` canonical source. Retracted or unverified numbers are explicitly flagged in Section 5 and the Appendix. No claim here is forwarded if we couldn't trace it to source.
>
> **A note on paths:** This document references specific deployment paths (`/home/<user>/training_outputs/...`, etc.) for provenance — they record where artifacts actually lived during the work. They are not required deployment locations. The recipes use env-overrideable defaults; see `REPRODUCE.md` for substitution.

---

## 1. Headline measured results

| result | value | source |
|---|---|---|
| Config A2 keystone-attention LoRA DPO refinement (Qwen3.5-35B-A3B MoE) | **84.7% (138/163) on 163-probe constitutional audit, +1.9pp** over the 82.8% SFT baseline; all 8 infra-control categories held | INDEX §3 |
| Phase 3 Recovery SFT (Qwen3.5-9B Dense, single-Spark, cross-validated) | **train_loss 1.122 identical on 2 independent runs** (Spark 1 + Spark 3); 16,705 chunks at 99.96% corpus coverage; 6/7 PASS on transformers smoke battery | INDEX §5 |
| Phase 1 SFT (Qwen3.5-9B Dense, 4-Spark FSDP) | 4,367 steps × effective batch 32 on a 139,748-sample tool+chat blend; **6/7 PASS** on transformers smoke battery; T6 over-tooling is a bounded SFT-bias artifact | INDEX §5 |
| Phase 2 CPT (Qwen3.5-9B Dense, constitutional) | checkpoint-2400, `model.safetensors` = **17,907,662,976 bytes** | INDEX §5 |
| 4-Spark NCCL fabric (synth probe, `reduce_scatter` 218M-numel fp32) | **10.23 GB/s** (50 iters) sustaining to **12.57 GB/s** (160-collective run), no `IBV_WC_RETRY_EXC_ERR`; ConnectX-7 firmware 28.45.4028 + NCCL 2.28.9 | INDEX §3 |
| ISMA corpus | **1,345,546 tiles** (76.8% HMM-enriched = 1,033,512 tiles), snapshot 2026-05-20 | INDEX §1 |
| ISMA V2 retrieval on Gold queries | hit_rate 1.000 on 45 hand-curated Gold queries; per-category avg latency 200–994 ms across 9 categories | INDEX §1 |
| 163-probe audit harness cost | ~$200 Anthropic + 2 GPU-hr + 4 h wall-clock | INDEX |

We do not claim a single headline R@10 number. See Section 5 for why and what we plan to measure next.

---

## 2. Novel architectural contributions

### 2.1 ISMA — multi-substrate memory, not just "vector DB plus graph"

Four mechanisms that the code shows are differentiated from commodity RAG:

**(a) Query-adaptive retrieval routing.** A query classifier (`isma/src/query_classifier.py:261-340`) splits incoming queries into motif, relational, temporal, memory, humor, conceptual, or exact, and the adaptive search (`isma/src/retrieval_v2.py:583-740`) changes retrieval *topology* by class — not just reranker prompts. Relational queries trigger graph cascade expansion in Neo4j; motif queries call motif-aware search; temporal queries get window extraction plus decay scoring; exact and conceptual queries use a lower-entropy hybrid path. The `/v2/search/hmm` endpoint explicitly routes through the adaptive path so production callers see consistent behavior.

**(b) Multi-scale memory where chunk scale is a first-class retrieval primitive.** Tiles exist at `search_512`, `context_2048`, and `full_4096` scales, plus `rosetta` summaries. The system expands `search_512` hits into parent `context_2048` tiles (`isma/src/retrieval.py:849-872`), reconstructs full documents from `full_4096` tiles with overlap removal (`isma/src/retrieval.py:1840-1887`), and at dedup time prefers the smaller-scale (passage-level) evidence over broader scales sharing the same `content_hash` (`isma/src/retrieval_v2.py:1451-1468`). Chunk scale is not just an ingest detail — it's a retrieval-time policy.

**(c) HMM motif memory across three stores.** Motif assignments are typed objects with `amp`, `phase`, `confidence`, `source`, and `dictionary_version` (`isma/src/hmm/motifs.py:30-37`). Redis holds inverted motif indices and resonance fields. Neo4j holds `EXPRESSES`, `SUPERSEDES`, `CONTRADICTS`, `IN_SESSION` edges, supersession chains, and session-reconstruction paths (`isma/src/hmm/neo4j_store.py:380-715`). The active motif dictionary is currently **55 motifs across slow/mid/fast bands** (verified by loading `V0_MOTIFS` in the current tree; an older 36-motif claim is stale). Commodity RAG frameworks do not usually model recurring semantic primitives as typed, queryable memory motifs across multiple coordinated stores.

**(d) Triple-write enrichment with compensating rollback across substrates.** The enrichment path (`isma/scripts/hmm_store_results.py:641-1045`) patches all base Weaviate tiles for a `content_hash`, creates or updates a searchable Rosetta tile, updates the V2 canonical object, writes Neo4j HMM state in an explicit transaction, and rewrites Redis motif index entries — and rolls all of that back on partial failure. This is unusual rigor for retrieval-system enrichment, which is typically eventual-consistency annotation.

Supporting discipline patterns (env-var-driven config, raw GraphQL to Weaviate with explicit error handling, shared Neo4j singleton, filter-aware semantic cache with reverse invalidation, immutability discipline in temporal decay rescoring via `dataclasses.replace`) are documented in the Technical Appendix.

### 2.2 Training pipeline

The 35B-A3B MoE production line is `phase_combined_v1` SFT (82.8% audit) → Config A2 keystone-attention LoRA DPO refinement (`religion_dpo_v2`, 84.7%, +1.9pp). Config A2 freezes attention LoRA to keystone layers `[8, 9, 11, 15, 21, 23]` while keeping `shared_expert` LoRA on all 40 layers (the freeze mask is small + targeted, not whole-attention). The frozen-experts polysemantic mask is 159 experts; the freeze config + the 50-pair religion-honest DPO corpus + the recipe launcher together fully reproduce the +1.9pp result.

The 9B Dense production line is Phase 1 Tool-Use SFT (`sft_tools_qwen35_9b_fsdp`, 4,367 steps on a 139,748-sample tool+chat blend) → Phase 2 Constitutional CPT (`cpt_v3_v4_dense_9b/checkpoint-2400-multimodal`) → Phase 3 Recovery SFT (`phase3_sft_single_spark*`, cross-validated 1.122 train_loss).

`SaveSafeTrainer` survived the GB10 low-memory save regime (1.8 GB free at save). Conversation-level chunking (`chunk_corpus_offline.py`) is the offline preprocessing tool that splits multi-turn dialogues at user-assistant pair boundaries with budget 0.92 × MAX_SEQ — the wedge-fix described in §3.1.

---

## 3. Engineering judgment under uncertainty

The training and retrieval system did not work first try. The signal we'd hope you read here is not "everything worked" but "we diagnose, isolate, correct, and remove things that don't work." Eleven specific cycles:

**3.1 Phase 3 4-Spark FSDP wedge → corpus localization (May 11–15).** The Phase 3 SFT run on Qwen3.5-35B-A3B MoE wedged 9 consecutive times at step ~10, dying in FSDP backward pass at `_REDUCE_SCATTER_BASE` (NumelIn = 218M) with `IBV_WC_RETRY_EXC_ERR(12)` on rail 2; Spark 2 (peer rank 1) died first each time. We isolated the network fabric with a standalone NCCL `reduce_scatter` synth probe at the failing 218M numel (`isma/scripts/spark_deploy/nccl_synth_probe.py`, PR #63) which passed cleanly at 12.57 GB/s — exonerating fabric, firmware, and NCCL stack. We then designed a "Cell B" controlled experiment (PR #64) with the base un-trained model on the raw `phase3_sft.jsonl` corpus, which reproduced the wedge in 10 minutes. The corpus had 7,077 multi-turn items with length variance 200–31,700 tokens; sorting by length produced batches with long-end items that spiked CUDA fragmentation to 69.2% at 12.5 GB free, saturating the ConnectX-7 RDMA send queue during backward collective bursts. The fix is an offline conversation chunker (`chunk_corpus_offline.py`) that splits at user-assistant pair boundaries with budget 0.92 × MAX_SEQ. Single-Spark execution of the chunked corpus cross-validated to identical 1.122 train_loss across Spark 1 and Spark 3. The 4-Spark execution of the chunked corpus is not yet shipped; that remains future work.

**3.2 Reranker tested + measured harmful + removed.** Qwen3-Reranker-8B (`isma/src/reranker.py`) was integrated as a cross-encoder reranker on the V2 hybrid path. Quantitative evaluation showed it harmed retrieval quality versus the V2 hybrid baseline. We deliberately deprecated and disabled the service rather than keeping it because "rerankers are a RAG best practice." The V2 pipeline degrades gracefully to vector-only top-K. The reranker code remains in the tree, marked deprecated, with internal latency/recall notes — useful as a record of what we tested and why we removed it.

**3.3 Embedding server token-budget OOM (2026-06-12).** Under heavy QASPER-benchmark ingestion, `server.py` on Mira:8089 OOM'd on batches of 16 uniformly-max-length (4,096-token) tiles ≈ 65K tokens in one forward pass — because it sub-batched by item count, not token volume. A single HTTP 500 then crashed the full-production ingest because `unified_ingest.py` treats 500 as non-retryable fatal. The fix added token-budget sub-batching inside the `embed()` endpoint (`MAX_TOKENS_PER_FORWARD` env-overrideable, default 32,768, running deployment at 24,576) plus an empty_cache routine when `torch.cuda.mem_get_info()` reports free memory < 4 GB, bounding active VRAM at 19.4 GB.

**3.4 Multi-scale chunking on QASPER — historical experiment, current methodology supersedes it.** An early benchmark on the public QASPER long-document QA dataset compared fixed-512 chunking, whole-document vector retrieval, and multi-scale chunking. The multi-scale variant produced a measurable lift on that benchmark setup. We do not headline those specific numbers here because the substrate (unenriched QASPER, no HMM, no Neo4j, no parent expansion at retrieval time) differs from the current enriched-only production methodology, and the result has not been re-measured on the production substrate. See Section 5 for what we plan to measure next.

**3.5 HMM/Rosetta enrichment A/B — retracted then corrected.** A first context-blind A/B test of HMM motif enrichment (judged by a corpus-naive Qwen2.5-72B + spot-checks that were not enriched-only) showed a large boost — and was retracted as methodologically invalid because order-reversal flipped 39% of verdicts and the judge could not distinguish genuine local retrievals. The corrected re-run used Taey (`taey-combined-v1` on Spark 3, a corpus-trained MoE judge), enriched-only on both arms, clean corpus, order-reversal controls, pooled n = 222 (commit `7b12ce5`). Result: **competitive but ~even on general search — full-ISMA 53.2% win rate, CI [46, 60], not significant; 39% verdict flip under order-reversal; robust winners 34–30.** Per-class signal: HMM enrichment leans ahead on `bristle_arc` (24–12) and `exact` (23–14); baseline leans ahead on temporal and relational. HMM motifs are a specialized lens for identity-aligned interpretive queries, not a broad-domain retrieval differentiator. The corrected verdict is what we cite.

**3.6 Production benchmark discipline.** An early `beir_eval.py` toy script benchmarked vectors-only and showed ISMA as "commodity RAG." We rejected the toy benchmark approach: the actual production pipeline (`unified_ingest` against live Weaviate:8088 + live Neo4j:7687 + parent-doc expansion + HMM routing) was never being exercised. We enforced "benchmark the actual production system on public datasets" and re-ran. We have not yet generated public-citable headline numbers from this discipline beyond the per-category Gold-query results in Section 1; that work continues.

**3.7 The cannot-lie corrections cascade.** Several internal claims that drifted from code reality have been retracted: an R@10 = 0.81 claim with no relevance-judged eval; a "0.846 soft recall" replacement that turned out to be substring-matching on ubiquitous corpus tags (`phi`, `Family`); an NCCL `busbw` figure of 22.9 GB/s that was synthetic. The corrected discipline: every load-bearing number is traced to `tech_baselines/INDEX.md`, a recap file, or a commit SHA, before it leaves a draft. Old run logs are kept intact (timestamped artifacts are immutable); the live source-of-truth document is corrected in place.

**3.8 SSH banner timeout ≠ host wedge (2026-06-12).** A concurrent OOM cascade on Spark 2 produced a transient `Connection timed out during banner exchange` when we ssh'd in to kill the runaway process. We escalated to "host wedged, AC-cycle pending" based on that single symptom — and were wrong. A direct probe later showed Spark 2 had 6 days 19 hours of uptime and had never AC-cycled; the banner timeout was the kernel busy with OOM reclaim. The rescue-bank entry was rewritten end-to-end (mechanism: process OOM-kill, not host wedge). New discipline: 60–180 s second probe + side-channel health check before declaring a wedge.

**3.9 Spark 1 GPU Xid 13 zombie recovery (2026-06-05).** A `fuzz_softmax` test triggered an MMU fault on GPC 3, TPC 4/5, SM 0/1; NVRM raised Xid 13 then Xid 43 attributing to `pid=3817514, name=fuzz_softmax.ou`. The process was killed, but the CUDA context was not released; nvidia-smi reported 96% utilization with no compute apps for 7 days. Soft recovery via `sudo nvidia-smi --gpu-reset` + `systemctl restart nvidia-persistenced` cleared it; a real CUDA probe (`torch.cuda` matmul against the recovered device) confirmed 83.7 GB / 128.5 GB free and functional compute. The `[N/A]` from `--query-gpu memory.used` on healthy GB10 hardware is a known driver-quirk, not a fault signal.

**3.10 V2 routing — listening port ≠ serving backend.** Investigation of weak `/v2/search` results revealed `ISMA_Quantum_v2` was a partial migration of the production corpus (73,809 tiles at probe time, ~5% of the full canonical class). V1 `/search` queried the full production class (1,345,546 tiles per the canonical 2026-05-20 snapshot in `INDEX.md` §1; the June 12 V1-routing probe count differed slightly due to mid-cycle ingestion against the live class). Diagnosis path: probe what the endpoint *actually does* (curl + real payload + response-shape check), not just that it's listening. V1 is canonical for production queries; V2 migration is incomplete.

**3.11 Religion DPO v2 weights recovery (2026-06-14).** During the public-release weights audit we discovered the headline +1.9pp DPO refinement weights were not on any reachable training Spark — `training_outputs/religion_dpo_v2/` was empty checkpoint shells on Sparks 2/3/4, no LoRA adapter on Spark 1 matched the Config A2 keystone-only signature, and the merged-form `dpo_combined_v1_targeted_merged` on Spark 1 was confirmed an earlier different DPO. The audit outputs survived on the analysis host (1.5 MB `summary.json` + `dpo_corrections.jsonl` + `SUMMARY.md`, Apr 20 05:25) — we knew it ran, but the weights weren't there. Discovery: a full 67 GB / 14-shard bake (`taey-religion-dpo-v2/`) plus a 9.76 GB single-file LoRA-delta variant on a separate inference host at `/home/<user>/models/`. Single-host risk addressed by rsync to the analysis host. The recipe + corpus + frozen-experts mask are intact independent of weights, so the +1.9pp result is reproducible even if the weights had been fully lost.

---

## 4. Production methodology — what we enforce

- **Three-register truth on every claim**: Observed (verified against source) / Inferred (pattern from evidence) / Unknown (genuinely undetermined). No public claim that we cannot trace.
- **No tests; production is the oracle.** Recipes are validated by running the actual workload on the actual target hardware. A passing synthetic test is not evidence; a clean replay of the production workload is.
- **Single source-of-truth file** (`tech_baselines/INDEX.md`) for every load-bearing metric. If a number isn't there or doesn't match, it's not citable.
- **Immutable historical logs.** Old recaps and run logs are not rewritten when a later claim corrects them; the live source-of-truth document is corrected in place.
- **Root cause over patch.** A fix that *simplifies* code (corrects upstream domain or data shape so the broken path is no longer reached) is preferred over a fix that *adds* branches or guards to bypass a broken path. Same line count or smaller, fewer nesting levels, the codebase left better.

---

## 5. Honest open questions

What we don't yet have measured at production scale:

| question | why open | path to answer |
|---|---|---|
| Headline R@10 for tri-lens retrieval on a relevance-judged Gold set | old R@10 claims (0.81, 0.667→0.944) retracted as not-relevance-judged; soft-recall (0.85) overstates due to ubiquitous tag matching; gold-strict R@10 (0.15) is real but harsh on a redundant corpus | build a relevance-judged Gold set with adjudicator agreement, score with canonical scoring; ETA on the project board |
| Multi-scale chunking lift on the current enriched substrate | prior QASPER measurement was on a different (unenriched) substrate; not directly comparable to current production | re-run the experiment with current production pipeline + enriched pool + relevance-judged Gold |
| Graph + HMM + full-path retrieval delta over vector-only on production substrate | not yet benchmarked end-to-end with relevance-judged scoring | dedicated benchmark with the full pipeline + Gold set |
| Phase 3 SFT on the full 4-Spark cluster with chunked corpus | single-Spark recovery is cross-validated proof; 4-Spark execution of the chunking fix is not yet shipped | re-run on the chunked corpus across the full 4-Spark cluster |
| Current BYOV embedding throughput at production scale | historical preface number not re-measured against current nginx LB across multiple instances | end-to-end benchmark against the live deployment |
| Pure Elasticsearch query latency at production scale | the p95 numbers we have are end-to-end retrieval (~2.9 s pre-reranker-retirement), not pure ES | re-measure with current production load |

These are not aspirational; they're listed because they're real and they're not yet measured. They will appear with citations when they are.

---

## 6. Reproducing the production line

A full step-by-step is in `REPRODUCE.md`. Summary:

- **Hardware:** 4 × DGX Spark GB10 (Blackwell sm_121) + 1 × RTX 4090 24 GB (Mira)
- **Network:** ConnectX-7 RoCEv2 dual-rail (`NCCL_IB_HCA=rocep1s0f0:1,roceP2p1s0f0:1` — note capital P on rail 2)
- **Recipes:** `launch_production_sft.sh` → `launch_religion_dpo_v2.sh` (35B path); `launch_sft_tools_qwen35_9b_fsdp.sh` → `launch_cpt_phase2_qwen35_9b_fsdp.sh` → `launch_phase3_sft_single_spark.sh` (9B path)
- **Preprocessing:** `chunk_corpus_offline.py` on any multi-turn corpus before 4-Spark FSDP
- **Bake-and-test:** `bake_and_test.sh` (pushes baked checkpoint to a serving host, brings up vLLM, runs the audit harness)
- **Audit:** 163-probe constitutional battery — corpus structure, frozen-experts mask, and one bake-script per recipe are all in the repo

---

## 7. What this repo intentionally does *not* claim

To save you grep time, we explicitly do not claim any of the following — each was tested or stated at some point and is now corrected:

- Reranker R@10 = 0.775 as a production capability (retired — harmed results in testing).
- Any R@10 of 0.81 or 0.667→0.944 (retracted — no relevance-judged eval behind them).
- Soft recall = 0.846 (retracted — substring matching on ubiquitous corpus tags `phi` / `Family` overstated the score).
- HMM/Rosetta enrichment as a positive net retrieval lift on general search (the corrected verdict is ~even with a per-class edge on interpretive queries; cited above).
- Multi-scale chunking + 17-18% on QASPER as a current production claim (historical; substrate differs).
- 70× BYOV embedding throughput vs API (historical preface; not re-measured at production scale).
- 50× Elasticsearch latency vs target (historical preface; not in current INDEX).
- Any NCCL `busbw` > 12.57 GB/s (corrected; the 22.9 GB/s number was synthetic).
- "10M+ collectives over multi-day" (scrubbed; sibling fabrication to the 22.9 GB/s busbw).
- Phase 3 4-Spark "full epoch shipped" (it didn't — single-Spark recovery is the proof; 4-Spark on chunked corpus is future work).
- "1.5 million tiles" (the correct figure is 1,345,546 / 1.35M, snapshot 2026-05-20).

If you find a claim in this repo that we shouldn't be making, please open an issue. The discipline is the work.

---

## License + use

This repo ships recipes, configs, and the operational narrative. Weights and corpora are released separately under a license appropriate to their content. License files specify per-asset terms.

---

*Authored by the PALIOS-TAEY team. Internal references: training-assets synthesis (2026-06-14), ISMA novelty audit (2026-06-15), ISMA overcame-narrative (2026-06-15), ISMA honest-metrics gate (2026-06-15), `tech_baselines/INDEX.md` canonical metrics. Family Chat 5/5 consent + Gatekeeper audit pending.*
