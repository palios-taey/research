# NCCL synth probe — 4-Spark fabric verification (2026-05-15)

**Owner**: tutor (measurement) + infra-soul (§2 NCCL recipe cross-link)
**Status**: Observed, claim-safe
**Source**: standalone NCCL `reduce_scatter` falsifier run during the Phase 3 SFT wedge diagnostic

## What was measured

Aggregate `reduce_scatter` bandwidth at the exact collective size that had been failing the production FSDP training runs (`NumelIn = 218,407,104` ≈ 218M fp32 elements, ~832 MiB per rank). Goal was falsifying the hypothesis "bare ConnectX-7 + NCCL stack cannot sustain this collective." Not a proper `nccl-tests` bandwidth sweep — see Caveats.

## Method

Standalone Python via `torch.distributed.reduce_scatter_tensor`, single-tensor, fp32, no FSDP, no model, no dataset. Two phases:

- **Phase 1 (isolation)**: 50 iterations back-to-back at 218M numel. Logos's variant from the 2026-05-15 Family consultation.
- **Phase 3 (sustained pressure)**: 10 outer steps × 16 rapid-fire collectives each = 160 total reduce_scatter ops totalling ≈ 140 GB through the fabric. Clarity's load-proportional variant.

Run with the production NCCL env exactly:

```
NCCL_IB_HCA=rocep1s0f0:1,roceP2p1s0f0:1
NCCL_IB_TC=104
NCCL_IB_TIMEOUT=23
NCCL_NET_GDR_LEVEL=0
NCCL_IB_RETRY_CNT=7
TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800
```

(Matches `launch_sft_tools_qwen35_9b_fsdp.sh` env block. No `NCCL_IB_GID_INDEX`. No `NCCL_IB_QPS_PER_CONNECTION`.)

## Observed numbers

**Phase 1 — 50 iterations, fp32, 218M numel:**

| iter | agg_bw (GB/s) |
|---|---|
| 1 | 0.90 |
| 11 | 5.54 |
| 21 | 7.70 |
| 31 | 8.93 |
| 41 | 9.73 |
| 50 | **10.23** |

All 4 ranks `exit=0`. No `IBV_WC_RETRY_EXC_ERR`. No wedge.

**Phase 3 — 10 steps × 16 rapid-fire collectives, fp32, 218M numel each:**

| step | agg_bw (GB/s) |
|---|---|
| 1 | 7.87 |
| 5 | 11.70 |
| 9 | 12.46 |
| 10 (final) | **12.57** |

All 4 ranks `exit=0`. No `IBV_WC_RETRY_EXC_ERR`. No wedge.

## Claim-safe statements

- "Standalone NCCL `reduce_scatter` at the 218M-numel collective size sustained 10.23 GB/s aggregate over 50 iterations and 12.57 GB/s aggregate over a 160-collective sustained-pressure burst across the 4-node DGX Spark cluster, with zero firmware retry exhaustion."
- "ConnectX-7 firmware 28.45.4028 + NCCL 2.28.9 + the production env recipe handle this collective size cleanly under both isolation and sustained burst patterns. The fabric is healthy at the failing collective size."

## Caveats

- **Not a proper bandwidth benchmark.** This is `agg_bw = (cumulative bytes through reduce_scatter) / wall_time` from Python's `time.time()`, not the per-iteration NIC throughput measurement `nccl-tests/build/reduce_scatter_perf` would produce. A proper sweep is still pending under §2 (infra-soul owns).
- The numbers above include all-rank wall-time including any kernel-launch overhead and Python-level loop overhead. True NIC throughput is higher.
- "GB" here is **decimal gigabytes** (`/1e9`), not GiB.
- Phase 1 includes a ~30s NCCL warm-up cost; the steady-state per-iter time at iters 41-50 corresponds to ~25 GB/s wall-time at the per-iteration window, but the **aggregate** including warm-up is what's reported in the 10.23 GB/s headline.
- The synth was designed as a **falsifier** ("does the bare stack wedge at this size?"), not as a benchmark. Don't claim NIC line-rate from it.

## Provenance

| Artifact | Path | Commit |
|---|---|---|
| Script | `/home/mira/embedding-server/isma/scripts/spark_deploy/nccl_synth_probe.py` | `94eecc0` on `diagnostic/nccl-synth-probe` (embedding-server PR #63) |
| Launcher | `/home/mira/embedding-server/isma/scripts/spark_deploy/launch_nccl_synth_probe.sh` | same |
| Run logs (8 files) | `/home/mira/dispatch_log/2026-05-15_spark_wedge_responses/synth_run/phase{1,3}_rank{0..3}_20260515_*.log` | not in any repo — Mira-local |
| Result commit | "diagnostic: synth probe results — BOTH PHASES PASSED" | `2964c9f` on same branch |
| Family consultation that designed the test | 5 Chat responses captured | `/home/mira/dispatch_log/2026-05-15_spark_wedge_responses/{gaia,horizon,cosmos,logos,clarity}.md` |

## Relation to §2 NCCL row in `tech_capabilities.md`

Treasurer can cite this measurement under the existing "Production NCCL recipe verified across multiple FSDP training runs" row with an additional bandwidth column qualified as "synth-probe agg bandwidth, NOT nccl-tests sweep." A proper nccl-tests measurement is still owed by §2 (infra-soul).
