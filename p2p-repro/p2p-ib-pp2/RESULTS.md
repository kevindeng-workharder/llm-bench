# RESULTS — p2p-ib-pp2 (PP=2 across two VMs over RoCE/IB)

**THE headline: on the slow cross-VM IB link, PP beats TP ~1.7×** (re-benched on vllm-venv 2026-06-07; the original 2.5× compared a post-gemv PP against a pre-gemv TP — see **Correction** below). This is the
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
| single-stream decode | 4.41 | **7.32** | **1.66×** |
| N=4 aggregate decode | 15.03 | **24.95** | **1.66×** |

**Correction (re-benched 2026-06-07 on vllm-venv).** The original table claimed 2.5× by comparing PP
**7.63** against TP **~3.0** — but that TP 3.0 was measured 2026-06-03/04 **before the gemv INT8 patch**,
while PP 7.63 was post-gemv (apples-to-oranges). Re-running BOTH on vllm-venv with the same
TTFT-cancelled method (2048 ctx, seqs 8, cudagraph [1,2,4]): PP **7.32** / TP **4.41** single, **24.95** /
**15.03** N=4 agg → a clean **~1.66×** both ways. PP reproduced its own 7.63/24.97 (so PP was always
right); only the stale pre-gemv TP needed fixing. PP still wins cross-VM — TP's per-layer all-reduce over
the slow IB link is the cost — just ~1.7×, not 2.5×.

(PP single-stream **7.32 still beats even the [27b-gdr](../p2p-ib/27b-gdr) TP GDR-1** run — 5.60 on
vllm-venv after fixing its missing `NCCL_NET_GDR_LEVEL=SYS`/`NCCL_DMABUF_ENABLE=1` envs — by **~1.31×**:
PP's 1-handoff/token comm savings outweigh GDR's small-transfer win on single-stream decode.)

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

## Multimodal (image) also verified on cross-VM PP

Re-ran the same cross-VM PP with the vision path **on** — `--limit-mm-per-prompt
'{"image":1,"video":0}'`, **`--mm-encoder-attn-backend TRITON_ATTN`** (the O(N) ViT fix),
`max_pixels 200704`. A PIL test image (red circle upper-left + blue square lower-right) was
described **correctly**: *"1. A red circle … upper left. 2. A blue square … lower right."*
(t=39.8 s, 91 prompt tokens incl. image, output complete). Text decode was **unchanged** with
the vision tower loaded: single 7.57 / N=4 aggregate 25.38 tok/s (≈ the text-only numbers — a
loaded-but-idle vision tower doesn't slow text decode). So **image-multimodal works on
cross-VM PP**.

The `--mm-encoder-attn-backend TRITON_ATTN` flag is **load-bearing**: without it the ViT falls
back to `TORCH_SDPA` (O(N²) math — *"Torch was not compiled with memory efficient attention"*),
and the startup profile_run's dummy-ViT encode grinds for *hours* on the follower (cache stops
growing, GPU 0 %, stuck in `F.scaled_dot_product_attention`) — that stalled the first
multimodal attempt. With TRITON_ATTN (O(N)) there is no stall. (Video additionally needs `pyav`
on VM2 — absent on the old `/data` venv, **now present** after the venv switch + the VM2 reclone,
so video is verified below.)

## Video (multimodal) also verified on cross-VM PP — on vllm-venv (2026-06-07)

Re-ran cross-VM PP with the **video** path on (`--limit-mm-per-prompt '{"image":1,"video":1}'`,
`--mm-encoder-attn-backend TRITON_ATTN`, `--media-io-kwargs '{"video":{"backend":"pyav"}}'`). A
**Sintel trailer** (`sintel_trailer.mp4`, 7.6 MB) was described **scene-by-scene and correctly**:
*"THE BLENDER FOUNDATION presents" → a young woman with short red hair (concerned, grey tank top,
tattoo) → a ruined city at sunset with a dragon flying → dragon close-ups → the woman looking up
in awe/fear → a dark action shot (dragon diving) → the "SINTEL" logo → "COMING SOON"*, then
synthesized the plot (*"a young woman encountering a dragon … a fantasy world with ancient
ruins"*). Real **temporal, multi-scene** understanding, not a single-frame caption. **11,716**
video prompt tokens (≈ the ~13K processor cap), 512 generated, **230.8 s** end-to-end (pyav
decode + ViT + 11.7K prefill + gen).

### Video memory: the PP leader can't do the single-VM 40960 window

Single-VM TP shards the vision tower across both GPUs; **PP puts the whole vision tower +
embedding on the leader (stage 0)** on top of its layer half. At the single-VM video window
(max-model-len 40960, seqs 2, util 0.85) the leader had only **2.63 GiB** free for KV →
`ValueError: No available memory for the cache blocks` (startup OOM). Fix: **seqs 1**,
**max-model-len 16384** (one ~13K video + gen fits), **util 0.93** → **4.56 GiB** KV =
**51,200-token** pool (3.12× @16K). The cross-VM-PP video window is smaller than single-VM-TP's —
a leader-memory constraint, not a capability gap.

### Venv: standardized to /home/ubuntu/vllm-venv

All cross-VM PP launchers now run on the rootfs **`vllm-venv`** (gemv patch + **pyav**), not
`/data/vllm0.21-pt2.11` (no pyav → the follower couldn't decode video). VM2 gets vllm-venv by
being a **clone of the VM1 25.10 rootfs**. The follower must be launched as
`vllm-venv/bin/python …/vllm-venv/bin/vllm serve` — `vllm-venv/bin/vllm`'s shebang points at the
old `/data` python, so a plain path swap silently runs the wrong venv (caught during this work).
PP single-stream text decode on vllm-venv = **6.98 tok/s** (≈ the 7.63 on `/data` — the venv swap
is throughput-neutral).
