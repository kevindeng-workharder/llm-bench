# pp2 — pipeline parallelism (PP=2) instead of tensor parallelism

A **different axis** from the three transport scenarios above. Those are all **TP=2**
(split every layer, all-reduce each layer) varied by *how the two GPUs talk*; this one
keeps the single-VM dual-GPU setup but changes the **parallelism mode** to
**pipeline-parallel (PP=2)** — split the 64 GDN/full-attn layers **by depth**
(`Worker_PP0` = first half, `Worker_PP1` = second half), with **one activation handoff
per stage boundary** instead of an all-reduce every layer.

The point: PP trades per-layer all-reduce for a single per-stage handoff, which only
pays off on a **slow interconnect**. On single-VM (fast Infinity-Fabric) it doesn't — so
this scenario is the control that shows **TP wins on single-VM, and PP is the thing you'd
reach for on the cross-VM IB path** ([../p2p-ib](../p2p-ib)).

- **Status:** ✅ verified 2026-06-06 — PP=2 boots on the hybrid GDN model (`Qwen3_5`
  declares `SupportsPP`; workers `Worker_PP0`/`Worker_PP1`), output correct, **no late
  hang**. Single-stream **14.21 tok/s**, N=4 aggregate **43.62 tok/s**. See
  [RESULTS.md](RESULTS.md).
- **vs TP=2** ([../p2p-shm](../p2p-shm) / [../p2p-direct](../p2p-direct)): PP is
  **~8–19 % slower** on single-VM (single 14.21 vs 15.4; N=4 agg 43.6 vs ~53.7) — exactly
  as theory predicts. PP's win would be the slow cross-VM IB link, not this one.

## Requires

Same single-VM dual-GPU setup as [../p2p-direct](../p2p-direct) (unified kernel, gemv INT8
patch, `vllm-venv`). PP needs **no topology XML** — just `--pipeline-parallel-size 2`.

## Files

| file | role |
|---|---|
| [`../../servers/vllm/qwen3_6-27b-quark-int8-graph-pp2.sh`](../../servers/vllm/qwen3_6-27b-quark-int8-graph-pp2.sh) | launcher (kept with the general launchers — no topo XML needed). Text-only, 8192 ctx, cudagraph `[1,2,4,8]`. |
| `RESULTS.md` | PP=2 vs TP=2 numbers + the single-VM verdict + the ~2 h first-boot recompile gotcha |

## Run

```bash
# deploy the launcher to the dual-GPU guest and run (single process, both GPUs):
scp servers/vllm/qwen3_6-27b-quark-int8-graph-pp2.sh p2p-host:/tmp/ \
  && ssh p2p-host 'scp -P 2224 /tmp/qwen3_6-27b-quark-int8-graph-pp2.sh ubuntu@127.0.0.1:/home/ubuntu/'
ssh -p 2224 ubuntu@127.0.0.1 'cd ~ && setsid bash qwen3_6-27b-quark-int8-graph-pp2.sh </dev/null >/tmp/pp.log 2>&1'

# confirm PP split + no hang (first boot ~2 h — see RESULTS.md "first-boot recompile"):
ssh -p 2224 ubuntu@127.0.0.1 'grep -aoE "Worker_PP[0-9]" /tmp/pp.log | sort -u'   # want PP0 + PP1
```

## ⚠️ First boot ≈ 2 hours (one-time)

The `[1,2,4,8]` cudagraph sizes trigger a full GDN-kernel Triton recompile (shapes not in
`~/.triton/cache`), CPU-bound on the riscv64/TCG guest — ~2 h with **both GPUs at 0 %**.
It is **NOT a hang** (the Triton cache grows steadily). Cached after; same-config reboots
are fast. RESULTS.md shows how to tell compile-vs-hang apart.
