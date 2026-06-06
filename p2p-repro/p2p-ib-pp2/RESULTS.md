# RESULTS — p2p-ib-pp2 (PP=2 across two VMs over RoCE/IB)

**THE headline: on the slow cross-VM IB link, PP beats TP ~2.3–2.5×.** This is the
crossover that the single-VM [p2p-direct-pp2](../p2p-direct-pp2) control predicted —
single-VM TP wins, cross-VM PP wins.

**Verified:** 2026-06-06, host `10.103.11.199`, two guests (VM1 ssh 2224 / VM2 ssh 2225).
**Model / mode:** `Qwen3.6-27B-Quark-W8A8-INT8`, **PP=2** across 2 nodes (1 GPU each),
text-only, max-model-len 2048, max-num-seqs 8, cudagraph `FULL_DECODE_ONLY` `[1,2,4]`,
`--no-async-scheduling`, gemv INT8 patch. Same stack as
[../p2p-ib/27b-graph](../p2p-ib/27b-graph) but **pipeline- instead of tensor-parallel**.
**Launchers:** [`start_vm1_leader.sh`](start_vm1_leader.sh) (node 0) +
[`start_vm2_follower_headless.sh`](start_vm2_follower_headless.sh) (node 1).

## Cross-VM PP works

NCCL 2-node group formed over IB (`Connected all rings`, `via NET/IB/`); vLLM assigned
**PP ranks** (rank 0 = stage 0 on VM1, rank 1 = stage 1 on VM2); output correct
(`The capital of France is` → "Paris"); **no async×cudagraph hang** (`--no-async-scheduling`
held — the failure mode documented in [../p2p-ib/27b-graph](../p2p-ib/27b-graph/RESULTS.md)
did not occur). KV pool 48,810 tok (2048 ctx, 23.83x concurrency).

**Transport:** `DMA_BUF Support is force enabled` (RCCL_FORCE_ENABLE_DMABUF=1) but the ring
reports **`use ring PXN 0 GDR 0`** — i.e. host-bounce, *not* the GDR-1 direct-VRAM path.
Same GDR-0 path as 27b-graph TP, so the PP-vs-TP comparison below is apples-to-apples.

## Throughput — PP vs TP, cross-VM IB (both GDR 0)

Clean streaming decode (TTFT-cancelled, `ignore_eos`, warmed):

| metric | TP=2 ([27b-graph](../p2p-ib/27b-graph)) | **PP=2 (this)** | PP vs TP |
|---|--:|--:|--:|
| single-stream decode | ~3.0 | **7.63** | **2.5× faster** |
| N=4 per-client decode | ~2.8 | **6.24** | ~2.2× |
| N=4 aggregate decode | ~11 | **24.97** | **2.3× faster** |

(PP@GDR0 single-stream 7.63 even beats the [27b-gdr](../p2p-ib/27b-gdr) TP **GDR-1** run at
~5.4–6.2 — PP's comm savings outweigh GDR here.)

## Why PP wins cross-VM (and loses single-VM)

PP trades per-layer **all-reduce** for **one activation handoff per stage boundary**:

- **TP** all-reduces every layer — for 64 layers that's ~64 collective ops per token over
  the wire. On cross-VM IB each all-reduce is ~50 µs (vs ~10 µs Infinity-Fabric), so TP is
  throttled by the link → ~3 tok/s.
- **PP** sends one activation tensor across the stage boundary per token (1 IB transfer),
  so the slow link barely matters → 7.63 tok/s.
- On **single-VM** ([p2p-direct-pp2](../p2p-direct-pp2)) the link is fast (Infinity-Fabric),
  so TP's all-reduce is cheap and PP's batch=1 pipeline bubble makes it ~8–19 % *slower*.

**Rule confirmed: PP for slow interconnects (cross-VM IB), TP for single-VM.**

## Gotchas hit getting here

1. **Multimodal ViT stalled the first attempt.** The 27b-gdr-derived launcher kept the
   vision tower (`max_pixels`, no `TRITON_ATTN`); the dummy-ViT profiling on the follower
   ran the O(N²) `TORCH_SDPA` path and ground for ~hours. Fixed by relaunching **text-only**
   (`--limit-mm-per-prompt '{"image":0,"video":0}'`) — text decode is the right PP-vs-TP
   metric anyway.
2. **First-time GDN recompile is ~1–2 h on TCG** (full-head PP kernels not cached from the
   half-head TP runs); cached after. Distinguish compile-vs-hang via `~/.triton/cache`
   growth + `gpu_busy_percent`.
3. **Boot bring-up** (also see [../p2p-ib/27b-gdr/README.md](../p2p-ib/27b-gdr/README.md)):
   graceful-poweroff the single-VM first (clean rootfs unmount) or the 2 VMs hit a
   models.img lock / port race; mlx5 IB link + IPs (10.99.0.1/.2) come up ~178 s post-boot.
