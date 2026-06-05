# RESULTS — p2p-shm (single-VM dual-GPU, host SHM transport)

**Verified:** 2026-06-05, host `10.103.11.199`.
**Kernel:** `Image-6.19.5-p2p-all`, guest `iommu.passthrough=1`.
**Model / mode:** `Qwen3.6-27B-Quark-W8A8-INT8`, TP=2, cudagraph `FULL_DECODE_ONLY` `[1,2,4]`, gemv INT8 patch.
**Transport:** `via SHM/direct/direct` — split topo `rccl-topo-split.xml` + `--disable-custom-all-reduce`, async-scheduling on.

## Transport proof (leader log)
```
[sitecustomize] GEMV INT8 override active -> /home/ubuntu/gemv-patch/triton_scaled_mm.py
NCCL INFO Channel 00 : 0[...] -> 1[...] via SHM/direct/direct
```
No `via P2P/IPC`, no `via NET`.

## Throughput — `bench.py` N-sweep (96-tok streams)
| N | aggregate tok/s | per-req decode | TTFT | wall |
|---|---|---|---|---|
| **1** | 7.48 | **15.41** | 6.7s | 12.8s |
| 2 | 12.07 | 12.31 | 7.5s | 15.9s |
| 4 | 23.26 | 13.42 | 8.9s | 16.5s |

`>>> SWEEP CLEAN — no hangs/deadlocks`. vLLM self-reported generation throughput peaked ~22 tok/s.

## vs p2p-direct (identical config; only the transport differs)
| metric | [p2p-direct](../p2p-direct) (`P2P/IPC`) | **p2p-shm** (`SHM`) |
|---|---|---|
| single-stream decode | 16.20 | 15.41 |
| N=2 per-req decode | 11.93 | 12.31 |
| N=4 aggregate | 20.92 | 23.26 |
| transport | `via P2P/IPC` | `via SHM/direct/direct` |

P2P is ~5% faster single-stream (lower Infinity-Fabric latency than SHM), in line
with the "single-VM Infinity Fabric ~6–12 % faster than cross-VM IB" note. The
gap is **small** because single-stream 27B INT8 decode is gemv-compute-bound — the
per-token TP all-reduce is tiny, so the transport barely moves the needle. At N=4
the two are within noise (the aggregate metric is prefill-heavy / jittery on this
emulated platform; SHM's higher number is not a robust "P2P loses" result).

## No deadlock
async-scheduling left **ON** (single-VM); clean startup + N=1/2/4 sweep, same as
p2p-direct. The `async × cudagraph` hang remains specific to p2p-ib's **cross-VM**
distributed-executor path, not single-VM.
