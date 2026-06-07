# p2p-ib · 27B INT8 graph (cudagraph) — verified run + root-cause (2026-06-03/04)

Same model/path/setup as [`../27b-eager/RESULTS.md`](../27b-eager/RESULTS.md),
with `--enforce-eager` replaced by cudagraph `FULL_DECODE_ONLY`
(`cudagraph_capture_sizes=[1,2,4]`) **plus `--no-async-scheduling`** (the fix
explained below).

## TL;DR

cudagraph makes 27B decode **~15× faster than eager** (0.29 → ~4.5 tok/s raw),
confirming the eager bottleneck was CPU kernel-dispatch. But **plain graph hangs
on sustained generation** — and the cause is *not* a cudagraph collective
deadlock and *not* the NCCL watchdog (both were wrong early guesses). The real
cause is **vLLM async-scheduling's sampled-token copy CUDA event never
signalling under cudagraph**. Disabling it (`--no-async-scheduling`) gives a
**stable** server:

| 27B INT8, cross-VM IB | tok/s | stable? |
|---|---|---|
| eager, N=1 | 0.29 | ✅ stable, slow |
| graph + async-scheduling (default), N=1 | ~4.5 | ❌ **hangs** (req fails to complete) |
| **graph + `--no-async-scheduling`, N=1** | **~3.0** | ✅ stable |
| **graph + `--no-async-scheduling`, N=4 concurrent** | **~11 (aggregate)** | ✅ **136 reqs, 0 hang over 20 min** |

> **Re-tested 2026-06-07 on vllm-venv (post-gemv):** single-stream **4.41 tok/s** (3-run mean), N=4
> aggregate **15.03** — higher than the ~3.0/~11 above, which were measured 2026-06-03/04 **before the
> gemv INT8 patch** (the M=1 decode speedup is the difference). This corrected TP baseline is what the
> [p2p-ib-pp2 crossover](../../p2p-ib-pp2/RESULTS.md) now uses: PP 7.32 / TP 4.41 ≈ **1.7×** (not 2.5×).

`--no-async-scheduling` costs some single-stream throughput (4.5→3.0; async
scheduling overlaps GPU gaps) but **concurrency more than recovers it**: N=4 runs
~11 tok/s aggregate and is rock-solid.

## Root cause (located with py-spy + source)

Plain graph (async scheduling on) hangs on the **first** sustained request:
`Running: 1`, `generation throughput 0.0 tok/s`, **both GPUs pegged at 100%**.
py-spy of both workers' MainThread, at the hang:

```
InputBatch.update_async_output_token_ids (gpu_input_batch.py:1042)
  ← GPUModelRunner._sample (gpu_model_runner.py:3406)
  ← sample_tokens → Worker.sample_tokens → worker_busy_loop
WorkerAsyncOutputCopy thread (active): AsyncGPUModelRunnerOutput.get_output (gpu_model_runner.py:274)
EngineCore (waiting): step_with_batch_queue → collective_rpc result
```

Both are stuck **after** the forward, in the **sampling / async-output** path —
**not** in `all_reduce`, not in a collective. The blocking call (source):

```python
# AsyncGPUModelRunnerOutput: sampled ids copied D2H non-blocking, event recorded
self.sampled_token_ids_cpu = self._sampled_token_ids.to("cpu", non_blocking=True)
self.async_copy_ready_event.record()
# get_output() / update_async_output_token_ids():
self.async_copy_ready_event.synchronize()   # <-- never returns under cudagraph
```

vLLM **async scheduling** pipelines steps: step N's sampled tokens are copied D2H
asynchronously and fed into step N+1, gated by `async_copy_ready_event`. Under
cudagraph that event is never signalled on replay (the ROCm/riscv64 cudagraph +
event interaction is broken), so `.synchronize()` spins forever — both GPUs at
100%, the engine waits on the workers, the request hangs.

**Why eager never hung:** eager has no cudagraph, so the event records/signals
normally. That is the real eager-vs-graph difference (not "graph is faster but
deadlocks").

`async_scheduling` defaults to `None` and is **auto-enabled** for the `mp`
distributed-executor backend (`vllm/config/vllm.py`: `else: async_scheduling =
True`), so it is on unless you pass `--no-async-scheduling`
(`BooleanOptionalAction`, `arg_utils.py`).

### Things that were RULED OUT (early wrong guesses)

- **Not a cross-VM cudagraph collective deadlock** (the #10–17 saga). The
  workers are idle/in-sampling, not in `all_reduce`; and N=4 concurrency runs
  136 requests with **zero** hangs (below). The saga does not reproduce here.
- **Not the NCCL watchdog.** Plain graph's "crash on sustained gen" was the
  torch NCCL watchdog *catching* the above hang and tearing down (recover-by-
  crash). Disabling the watchdog (`TORCH_NCCL_ASYNC_ERROR_HANDLING=0`) only
  converted the crash into a permanent hang — it is **not** the fix and is not
  used here.

## Stability — N=4 concurrent soak (the saga's worry, settled)

20-minute continuous N=4 concurrency on the fixed server:

```
34 rounds × 4 concurrent = 136 requests, 0 hang, 0 fail, 1201 s
per-round aggregate steady at 10.3–11.4 tok/s (overall 10.9 tok/s)
>>>>> N=4 SOAK SURVIVED ✅ <<<<<
```

So graph + `--no-async-scheduling` + `--disable-custom-all-reduce` is stable
under concurrency, not just single-stream.

## Also confirmed

- **No cross-VM cudagraph deadlock at capture, no OOM.** The Gated-DeltaNet
  hybrid captures cleanly across two VMs (`Padding mamba page`); `FULL_DECODE_ONLY`
  with sizes [1,2,4] fits at gpu-mem-util 0.85.
- **Startup is slow and variable** (~11–30 min). The long pole is the Qwen3-VL
  multimodal processor load — `from_pretrained` does a recursive `deepcopy` of
  the HF config that is pathologically slow on the riscv64 CPU (py-spy caught the
  APIServer pinned in it). Not a hang; just slow. Worth caching/short-circuiting
  if startup latency matters.

## Recommendation

For 27B on this cross-VM IB path: **graph + `--no-async-scheduling`**, served
with concurrency (N≈4 → ~11 tok/s aggregate, stable). It is ~10× the eager
single-stream path and, unlike default graph, does not hang. Eager remains the
slow-but-simple fallback. The proper upstream fix would be making the
async-scheduling copy event work under cudagraph on ROCm — until then,
`--no-async-scheduling` is the practical answer.

## How measured

`../27b-eager/bench.py` (single-stream) and an N=4 `ThreadPoolExecutor` soak
(136 reqs / 20 min). py-spy (Python stacks; `--native` unsupported on riscv64)
located the `async_copy_ready_event.synchronize()` hang. Software: vLLM
`v0.21.1.dev0+gad7125a43`, RCCL 2.27.7 `96a25b5+`, ROCm 7.2.3, PyTorch 2.11,
kernel `6.19.5-p2p`.
