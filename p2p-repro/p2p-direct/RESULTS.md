# RESULTS — p2p-direct (single-VM dual-GPU, P2P/IPC)

**Verified:** 2026-06-05, host `10.103.11.199`.
**Re-tested:** 2026-06-07 on `/home/ubuntu/vllm-venv` (standardized off `/data/vllm0.21-pt2.11`) —
single-stream **16.18 tok/s** (3-run mean; ≈ the 16.20 below), N=4 agg 53.07, `via P2P/IPC` confirmed.
Venv swap is throughput-neutral.
**Kernel:** `Image-6.19.5-p2p-all` (unified; `cpu_supports_p2pdma` hack), guest cmdline `iommu.passthrough=1`.
**Model / mode:** `Qwen3.6-27B-Quark-W8A8-INT8`, TP=2, cudagraph `FULL_DECODE_ONLY` `[1,2,4]`, gemv INT8 patch.
**Transport:** `via P2P/IPC` (+ `GDR 1`), `NCCL_IB_DISABLE=1`, `NCCL_TOPO_FILE=rccl-topo.xml`.
**Knobs vs p2p-ib `27b-graph`:** custom all-reduce **enabled**, async-scheduling **enabled** (both, no hang).

## Transport proof (leader log)
```
[sitecustomize] GEMV INT8 override active -> /home/ubuntu/gemv-patch/triton_scaled_mm.py
NCCL INFO Channel 00/0 : 0[301000] -> 1[201000] via P2P/IPC comm ... nRanks 02
NCCL INFO Connected all rings, use ring PXN 0 GDR 1
```
No `via SHM`, no `via NET`.

## Throughput — `bench.py` N-sweep (96-tok streams)
| N | aggregate tok/s | per-req decode | TTFT | wall |
|---|---|---|---|---|
| **1** | 8.56 | **16.20** | 5.3s | 11.2s |
| 2 | 11.96 | 11.93 | 7.3s | 16.1s |
| 4 | **20.92** | 12.30 | 10.1s | 18.4s |

`>>> SWEEP CLEAN — no hangs/deadlocks` — every request completed at every N;
`GDR 1` stayed engaged. vLLM self-reported generation throughput peaked ~19 tok/s.

**Single-stream ~16.2 tok/s beats p2p-shm's ~15** — consistent with the
single-VM Infinity-Fabric path being a touch faster than SHM/host-bounce.

## What each knob is worth (single-stream decode)
| config | tok/s | what it adds |
|---|---|---|
| no iommu=pt, CA off, async off | ~4.3 | the *wrong* config (early measurement) |
| **+ `iommu.passthrough=1`** | ~11.2 | **~2.6×** — removes the emulated RISC-V IOMMU's per-DMA translation (affects *all* GPU DMA, not just P2P) |
| **+ custom all-reduce + async-scheduling** | **~16.2** | CA over Infinity Fabric + scheduler/GPU-exec overlap |

## async-scheduling: p2p-direct ≠ p2p-ib
p2p-ib's `27b-graph` **must** run `--no-async-scheduling` or it hangs
(`async_copy_ready_event.synchronize()` never returns under cudagraph). p2p-direct
ran **with** async-scheduling and did **not** hang — clean startup, clean N=1/2/4
sweep. So that deadlock is specific to the **cross-VM distributed-executor path**,
not cudagraph itself. Keeping async-scheduling on is part of how p2p-direct
reaches 16 tok/s.

## Reproduce
`start_dual_gpu.sh` + `rccl-topo.xml` here; host kernel build + dual-GPU QEMU
launcher (`iommu.passthrough=1`) in [`../p2p-ib/27b-gdr/host/`](../p2p-ib/27b-gdr/host).
Bench: `python bench.py 1 2 4` on the guest.
