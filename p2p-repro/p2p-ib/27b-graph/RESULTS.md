# p2p-ib · 27B INT8 graph (cudagraph) — verified run (2026-06-03)

Same model/path/setup as [`../27b-eager/RESULTS.md`](../27b-eager/RESULTS.md),
with `--enforce-eager` replaced by cudagraph `FULL_DECODE_ONLY`
(`cudagraph_capture_sizes=[1,2,4]`).

## Outcome

| | result |
|---|---|
| deploy / serve | ✅ correct output |
| **cross-VM cudagraph deadlock** | **none** — capture completed cleanly |
| OOM | none (gpu-mem-util 0.85, capture sizes [1,2,4]) |
| startup to ready | ≈ **675 s** |
| **decode throughput** | **~4.5 tok/s** (run1 4.498, run2 4.476; very consistent) |
| TTFT (prefill) | ≈ 5.6 s |

## The headline: eager bottleneck confirmed

| 27B INT8, cross-VM IB | decode tok/s | vs eager |
|---|---|---|
| `--enforce-eager` | 0.288 | 1× |
| **cudagraph FULL_DECODE_ONLY** | **4.49** | **~15.6×** |

cudagraph collapses the ~1000 per-token kernel launches (Triton INT8 quant+mm,
Gated-DeltaNet scan, per-layer all_reduce) into a single replay, erasing the
riscv64 CPU kernel-dispatch overhead that the eager profiling identified as the
bottleneck. The 15× jump **confirms that diagnosis**: eager was dispatch-bound,
not GPU-compute- or IB-bound.

Two penalties, now both quantified on this path:

| comparison | factor | cause |
|---|---|---|
| 27B eager → 27B graph | **15×** | eager per-kernel dispatch on slow riscv64 CPU |
| 4B graph → 27B graph (10 → 4.5) | **2.2×** | model size (4B → 27B) |

## Gated-DeltaNet hybrid + cross-VM cudagraph: no deadlock

The earlier project saga (single-VM multi-graph RCCL deadlock, tasks #10–17)
did **not** reproduce here. Startup log around the capture window:

```
[450s] interface.py:669  Padding mamba page          ← GDN state made cudagraph-safe
[465s] entered "Capturing CUDA graphs" phase
[675s] Application startup complete                  ← captured, no hang, no OOM
```

vLLM pads the Mamba/Gated-DeltaNet page so the hybrid is cudagraph-compatible,
and `FULL_DECODE_ONLY` captures the decode step (incl. the per-layer
all_reduce) cleanly across the two VMs on this build
(`v0.21.1.dev0+gad7125a43`, RCCL 2.27.7 `96a25b5+`). Mitigation on record if a
future build regresses: `NCCL_LAUNCH_ORDER_IMPLICIT=1` (not needed here).

## ⚠️ Stability caveat — fatal NCCL watchdog timeout on sustained generation

The 4.5 tok/s above (2 clean runs) is real, but the server did **not** survive
extended use. During a 3rd, longer streaming request the EngineCore died with a
fatal **torch.distributed NCCL watchdog timeout**:

```
c10d::ProcessGroupNCCL::WorkNCCL::checkTimeout (.../ProcessGroupNCCL.cpp:692)
  -> multiproc_executor: "Worker proc VllmWorker-0 died unexpectedly"
  -> RuntimeError: cancelled  ->  EngineDeadError
VM1 EngineCore died 12:34:35;  VM2 follower's own watchdog fired ~10 min later 12:44:47.
```

i.e. a cross-VM collective (all_reduce) **stalled past the NCCL watchdog
timeout** on the GDR-disabled host-bounce path, and the watchdog killed the
whole process group (fatal by design). The "3rd run stall" was therefore a
**server-side hang, not a client/ssh transient** (an earlier draft of this note
guessed wrong).

**Honest takeaway:** graph mode is the right *speed* answer (4.5 tok/s, 15×
eager) but on this host-bounce path it is **not yet stable for sustained
serving** — an intermittent collective stall trips the fatal watchdog. Before
production:
- raise the collective watchdog timeout so a slow host-bounce all_reduce isn't
  declared dead (torch PG timeout / `NCCL_TIMEOUT`; note our env already sets
  `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=7200` but the **NCCL watchdog** is a
  separate timer and was not raised);
- and/or attack the stall source (host-CPU contention on the bounce path, GDR
  absence). The 4B/graph baseline did not exhibit this in its verified run, so
  27B's longer per-step collective likely aggravates it.

Needs a longer multi-request stability run (with the watchdog timeout raised)
before graph 27B can be called production-ready here.

## How measured

Reused [`../27b-eager/bench.py`](../27b-eager/bench.py) (streaming, prefill/decode
split). Launchers in this directory differ from eager only in the
compilation-config (cudagraph vs `--enforce-eager`).
