# Qwen3.6-27B pipeline parallelism (PP=2) vs tensor parallelism (TP=2)

Splits the model **by depth** (PP) instead of splitting each layer (TP), on the
single-VM dual-7900-XTX setup. Launcher:
[`servers/vllm/qwen3_6-27b-quark-int8-graph-pp2.sh`](../servers/vllm/qwen3_6-27b-quark-int8-graph-pp2.sh).

## Does the hybrid GDN model support PP?

**Yes.** `Qwen3_5ForConditionalGeneration` (`qwen3_5.py`) declares `SupportsPP` +
`make_layers` + `is_pp_missing_parameter`, and is `IsHybrid`. No PP hard-block fires
(the only `"pipeline parallel is not supported"` asserts in vLLM are for `phi4mm` /
EAGLE3, not this model; the sibling Qwen3-Next GDN model also declares `SupportsPP`).
PP=2 boots cleanly — workers come up as **`Worker_PP0` / `Worker_PP1`** (vs `Worker_TP*`
for TP), each loading its half of the 64 layers.

## Results (text-only, 8192 ctx, max-num-seqs 8, cudagraph [1,2,4,8])

Clean streaming decode (TTFT-cancelled, `ignore_eos`, warmed):

| metric | TP=2 (p2p-shm, 27B) | **PP=2** | PP vs TP |
|---|--:|--:|--:|
| single-stream decode | 15.4 | **14.21** | **−8 %** |
| N=4 per-client decode | 13.42 | **10.90** | **−19 %** |
| N=4 aggregate decode | ~53.7 | **43.62** | **−19 %** |

Output correct (`The capital of France is` → "Paris"). KV pool 54,418 tok (8192 ctx,
6.64x concurrency); cudagraph `[1,2,4,8]` all captured. (Configs aren't perfectly
matched — PP here is 8192 ctx vs the [`p2p-shm`](../p2p-repro/p2p-shm/RESULTS.md) TP
run — but the direction and magnitude are unambiguous.)

## Conclusion: PP loses to TP on single-VM

PP is consistently **8–19 % slower** than TP here — exactly as theory predicts:

- PP's advantage is **less communication**: one activation handoff per stage boundary
  instead of an all-reduce every layer. That only pays off on a **slow interconnect**.
- This is single-VM with fast Infinity-Fabric / SHM, so TP's per-layer all-reduce is
  already cheap, and PP can't recover its **batch=1 pipeline bubble** (only one stage
  active at a time). At N≥4, TP's batching pulls further ahead (53.7 vs 43.6).
- **Rule of thumb: PP for slow interconnects (cross-VM IB); TP for single-VM.**

No late hang — cudagraph capture + KV/state-cache setup completed fine. The
`async-scheduling × cudagraph` hang documented in
[`p2p-ib/27b-graph`](../p2p-repro/p2p-ib/27b-graph/RESULTS.md) is **cross-VM
(distributed-executor) specific** and did **not** reproduce on single-VM PP.

## ⚠️ First-boot GDN recompile is brutal on TCG (~2 hours)

The PP launcher uses cudagraph sizes `[1,2,4,8]` (more than the multimodal `[1,2]`), so
the GDN Triton kernels (`chunk_gated_delta_rule`, `chunk_scaled_dot_kkt`,
`recompute_w_u`) **JIT-recompile for the new batch shapes** — **~2 hours, CPU-bound** on
the riscv64/TCG guest (both GPUs idle at 0 % the whole time; `~/.triton/cache` grew to
~294 MB; the engine logs repeated benign *"No available shared memory broadcast block
found in 60 seconds"*). It is **NOT a hang** — the Triton cache grows steadily the
whole time. Cached after, so a **same-config reboot is fast**. Any config change that
introduces new batch/cudagraph shapes pays this cost once (this is a property of
Triton-JIT-on-TCG, not of PP per se — see also `qwen3_6-27b-quant-bench.md` §1).
