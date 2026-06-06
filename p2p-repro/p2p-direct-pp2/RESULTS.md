# RESULTS — p2p-direct-pp2 (PP=2 on single-VM, default P2P)

The **PP cell** of the [p2p-direct](../p2p-direct) transport. Cross-VM IB × PP (where PP is
expected to win) is the separate [p2p-ib-pp2](../p2p-ib-pp2) cell.

**Verified:** 2026-06-06, host `10.103.11.199`.
**Model / mode:** `Qwen3.6-27B-Quark-W8A8-INT8`, **PP=2** (TP=1), text-only, max-model-len
8192, max-num-seqs 8, cudagraph `FULL_DECODE_ONLY` `[1,2,4,8]`, gemv INT8 patch.
**Transport:** vLLM default NCCL topology (no explicit topo XML — PP needs none), so the
inter-stage send/recv runs over the default single-VM P2P/IPC path (≈ p2p-direct). The
`NCCL_DEBUG=WARN` launcher suppressed the per-channel `via …` line, so this isn't grep-proven
the way the TP transports are, but single-VM default on this kernel resolves to P2P/IPC.
**Launcher:** [`../../servers/vllm/qwen3_6-27b-quark-int8-graph-pp2.sh`](../../servers/vllm/qwen3_6-27b-quark-int8-graph-pp2.sh).

## Model support

`Qwen3_5ForConditionalGeneration` (`qwen3_5.py`) declares `SupportsPP` + `make_layers` +
`is_pp_missing_parameter`, and is `IsHybrid`. No PP hard-block fires (the only `"pipeline
parallel is not supported"` asserts in vLLM are for `phi4mm` / EAGLE3; the sibling
Qwen3-Next GDN model also declares `SupportsPP`). Boots cleanly — workers come up as
**`Worker_PP0` / `Worker_PP1`** (vs `Worker_TP*` for TP), each loading its half of the 64
layers. KV pool 54,418 tok (8192 ctx, 6.64x concurrency). Output correct
(`The capital of France is` → "Paris").

## Throughput — PP=2 vs TP=2

Clean streaming decode (TTFT-cancelled, `ignore_eos`, warmed):

| metric | TP=2 ([p2p-shm](../p2p-shm), 27B) | **PP=2** | PP vs TP |
|---|--:|--:|--:|
| single-stream decode | 15.4 | **14.21** | **−8 %** |
| N=4 per-client decode | 13.42 | **10.90** | **−19 %** |
| N=4 aggregate decode | ~53.7 | **43.62** | **−19 %** |

(Configs aren't perfectly matched — PP here is 8192 ctx vs the [p2p-shm](../p2p-shm/RESULTS.md)
TP run — but the direction and magnitude are unambiguous.)

## Read: PP loses to TP on single-VM

PP is consistently **~8–19 % slower** — exactly as theory predicts. PP's only edge is
**less communication** (one activation handoff per stage boundary vs an all-reduce every
layer), which pays off **only on a slow interconnect**. This is single-VM with fast
Infinity-Fabric / SHM, so TP's per-layer all-reduce is already cheap, and PP can't recover
its **batch=1 pipeline bubble** (only one stage active at a time); at N≥4 TP's batching
pulls further ahead (53.7 vs 43.6).

**Rule of thumb: PP for the cross-VM IB path ([../p2p-ib](../p2p-ib)); TP for single-VM.**

No late hang — cudagraph capture + KV/state-cache setup completed fine. The
`async-scheduling × cudagraph` hang in [../p2p-ib/27b-graph](../p2p-ib/27b-graph/RESULTS.md)
is **cross-VM (distributed-executor) specific** and did **not** reproduce on single-VM PP.

## First-boot GDN recompile (~2 h, one-time)

cudagraph sizes `[1,2,4,8]` (vs the multimodal `[1,2]`) trigger a full GDN-kernel Triton
recompile (`chunk_gated_delta_rule`, `chunk_scaled_dot_kkt`, `recompute_w_u`) for the new
batch shapes — **~2 h, CPU-bound** on TCG: both GPUs at **0 %** the whole time,
`~/.triton/cache` grew to ~294 MB, the engine logged repeated benign *"No available shared
memory broadcast block found in 60 seconds"*. It is **NOT a hang.** Tell compile-vs-hang
apart:

```bash
find ~/.triton/cache -type f -newermt '-120 seconds' | wc -l   # >0 = compiling (cache growing)
cat /sys/class/drm/card{0,1}/device/gpu_busy_percent           # ~0% during compile, not a deadlock
```

Cached after, so a same-config reboot is fast. Any config that adds new batch/cudagraph
shapes pays this once — a property of Triton-JIT-on-TCG, not PP per se (see
[`../../docs/qwen3_6-27b-quant-bench.md`](../../docs/qwen3_6-27b-quant-bench.md) §1).
