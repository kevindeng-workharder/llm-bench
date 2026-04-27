# vLLM 0.19 fp16 batched-NaN bug — debug summary

Concise narrative of how we narrowed the bug, what's confirmed, what's
ruled out, and where the investigation stands.

For the full play-by-play (all 11+ instrumentation patches) see
[`vllm-019-batched-bug.md`](vllm-019-batched-bug.md). This doc is the
distilled "what we know" view.

## Symptom

```
Qwen3-4B fp16 graph TP1, --dtype float16, N=4 concurrent requests:
  client 0: "Here's a simple Python function..."     ✅
  client 1: "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"        ❌ token id 0 looped
  client 2: "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"        ❌
  client 3: "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"        ❌
```

- Threshold: graph mode N≥4, eager mode N≥2.
- Which client wins is non-deterministic (depends on engine scheduler
  arrival ordering). Not slot-position. Not prompt content.
- Affects every dense fp16 model tested (Qwen3-0.6B, Qwen3-4B,
  Qwen2.5-14B), every quant (AWQ MoE), eager and graph modes, TP=1 and
  TP=2.
- llama.cpp on the same hardware batches correctly at all N — hardware
  is not at fault.
- **vLLM v0.11.0 with `--dtype float16` on the same VM, same model, same
  flags is correct at all N (0/8 garbage at N=8).** Only vLLM v0.19.0
  reproduces.

## Hard constraints (set the search)

- **Same ROCm install, same C++ paged_attention_rocm kernel binary** is
  used by both vLLM 0.11 and vLLM 0.19. So the kernel itself is not the
  bug — the difference is in **how vLLM 0.19 calls it**.
- vLLM 0.11 source on this VM is at `/home/ubuntu/ai/lib/.../vllm/`,
  vLLM 0.19 at `/home/ubuntu/ai-2.10/lib/.../vllm/`. Both link the same
  `_rocm_C.so`.
- vLLM 0.19 git source at `/data/vllm-0.19` is `HEAD detached at
  v0.19.0` with only 3 modified files — all build-system patches
  (`CMakeLists.txt`, `cmake/utils.cmake`, `csrc/cuda_vec_utils.cuh`).
  No runtime Python is locally modified. `md5sum` of installed
  `attention.py`, `gpu_model_runner.py`, `qwen3.py`, `triton_attn.py`,
  `builtin.py` matches the source.

## Workaround that works

`--dtype bfloat16` instead of `--dtype float16`. Verified end-to-end:

| Config | N=1 | N=2 | N=4 | N=8 garbage |
|---|---|---|---|---|
| Qwen3-4B fp16 graph TP1 | 25 | 55 | 45 ❌ | 209 t/s, 6/8 bad |
| Qwen3-4B **bf16** graph TP1 | 37 | 57 | 114 | **212 t/s, 0/8 ok** ✅ |

bf16 has fp32-level 8-bit exponent (max ≈ 3.4e38) vs fp16's 5-bit
(max ≈ 65 504). bf16 doesn't trigger the overflow that fp16 does inside
the attention kernel.

## Bisection — what we ruled out

| Hypothesis | How tested | Verdict |
|---|---|---|
| MoE expert routing | Dense Qwen3-4B fp16 fails identically | RULED OUT |
| AWQ / compressed-tensors quant | Dense fp16 fails | RULED OUT |
| AITER backend | `VLLM_ROCM_USE_AITER*=0` reproduces | RULED OUT |
| Prefix caching | `--no-enable-prefix-caching` reproduces | RULED OUT |
| Chunked prefill | `--no-enable-chunked-prefill` reproduces | RULED OUT |
| CUDA Graphs | `--enforce-eager` reproduces (lower threshold N≥2) | RULED OUT |
| Async scheduling (PR #27614) | `--no-async-scheduling` reproduces | RULED OUT |
| Model Runner V2 (PR #25266) | `VLLM_USE_V2_MODEL_RUNNER` default off | RULED OUT |
| Bookkeeping vectorization (PR #25801) | Never merged | RULED OUT |
| Sampler bug | Sampler diff vs 0.11 = refactoring only; v9/v10 instrument shows `argmax` reads NaN logits | RULED OUT |
| `compute_logits` / lm_head matmul | v11 instrument shows hidden_states already NaN before lm_head | RULED OUT |
| C++ paged_attention_rocm kernel | Same kernel binary works correctly with vLLM 0.11 | RULED OUT (per user constraint) |
| Attention backend choice | ROCM_ATTN, TRITON_ATTN, no-custom-paged, no-prefill-decode-split — all reproduce | RULED OUT |
| Our 3 build-system patches | None touch runtime Python | RULED OUT |
| `torch.ops.vllm.X` dispatch vs direct call | Both paths reproduce | RULED OUT |

## Bisection — pinpointed location

Step-by-step with in-place `os.write(2, ...)` instrumentation on the
running 0.19 venv. All scripts are at
`scripts/instruments/instrument-019-v{2..11}.py`,
`instrument-qwen3-{layers-v2,attn}.py`, and `probe-attention-zeros.py`.

| Layer | Probe | Finding |
|---|---|---|
| `_prepare_input_ids` | v6/v7 (entry trace) | Called with `num_reqs=4`, `prev_sampled_token_ids` present, fast-opt branch taken |
| Fast-opt scatter | v8 (src trace) | `prev_sampled_token_ids[:4, 0] = [real, 0, real, 0]` — already has zeros |
| Sampler output | v9 (`sampled_token_ids`) | `shape=[4,1]`, `vals=[[real], [0], [0], [0]]` — sampler innocent |
| Sampler input | v10 (logits stats) | `max_per_row=[real, NaN, NaN, NaN]` — argmax(NaN-row) = 0 |
| `compute_logits` | v11 (hidden_states) | rows 1, 3 of `sample_hidden_states` already NaN before lm_head |
| `Qwen3DecoderLayer` | qwen3-layers-v2 | Layer 0 fully OK; layer 1's `self_attn` is the first to introduce NaN |
| `Qwen3Attention` | qwen3-attn | Q/K/V/q_norm/k_norm/RoPE all finite; `self.attn(q,k,v)` output has NaN at rows 1-3 |
| `Attention.forward` | probe-attention-zeros | Pre-fill output with zeros instead of empty: layer 0 writes ALL 4 rows correctly, layer 1+ writes rows 0,1 correctly + NaN to rows 2,3 |

So:
1. The kernel writes to all rows (zeros don't survive — they're
   overwritten with NaN, not left at 0).
2. Layer 0 produces valid output for all 4 rows; layers 1-35 produce
   valid output for rows 0,1 and NaN for rows 2,3.

Special-case for layer 0: at the first decode step its Q/K values flow
straight from input embedding without prior residual accumulation, so
they're smaller. Subsequent layers see larger Q/K (after RMSNorm scale-up)
and the bug triggers.

## Two more hypotheses tested 2026-04-27 — both wrong

**Hypothesis 5: KV-write-vs-attn-read ordering bug.** vLLM 0.19 split
the KV cache write and the attention compute into two separate
`torch.ops` (`unified_kv_cache_update` + `unified_attention_with_output`),
stitched only by a fake dependency tensor `kv_cache_dummy_dep`. Theory
was the fake dep didn't enforce a real GPU-stream wait on RDNA3, so
attention reads stale KV.

Patched `Attention.forward` to add `torch.cuda.synchronize()` (guarded
with `if not torch.cuda.is_current_stream_capturing()` so cudagraph
capture still works) between the two ops. **Both eager (sync runs every
step) and graph mode (sync only between captures) reproduce the bug at
N=4. RULED OUT.**

**Hypothesis 6: C++ paged_attention_rocm kernel itself.** vLLM 0.19
removed the `VLLM_ROCM_CUSTOM_PAGED_ATTN` env-var guard from
`use_rocm_custom_paged_attention`, so 0.19 always runs the C++ kernel
on gfx1100, while 0.11 lets you fall back to triton via env. Maybe the
C++ path has a bug exposed only in 0.19.

Patched `chunked_prefill_paged_decode.py` to force `use_custom = False`,
which sends every attention call through the triton kernel instead.
**Same 2/4 garbage at N=4. RULED OUT — the bug is in BOTH kernel
paths.**

So the bug is **not** in either backend kernel. It's in the data the
kernels are READING — specifically the KV cache contents or the
indexing tensors (`block_table`, `slot_mapping`, `seq_lens`) that tell
the kernel where to look. 0.19 must be filling these incorrectly for
some sequences. The kernel itself is innocent — it just dutifully reads
from wherever the metadata points it to and produces NaN if that data
is uninitialized memory.

This is consistent with the layer-0-OK / layer-1+ broken pattern:
layer 0 attention attends only to the *current* token (no past KV to
worry about), so even a wrong block_table doesn't hurt; layer 1+
attends to all past tokens including the ones layer 0 just wrote
elsewhere via `unified_kv_cache_update`, which is when the wrong-slot
write becomes a wrong-slot read.

## Current best hypothesis (still hunting)

vLLM 0.19 split a single op (KV-cache write + attention compute) into
**two separate ops** that the C++ kernel binary in `_rocm_C.so` was
designed to be called from within one Python-side `forward()`:

- **0.11** (`vllm/v1/attention/backends/rocm_attn.py:forward`) writes K,V
  to the cache (`PagedAttention.write_to_paged_cache(...)` or
  `ops.reshape_and_cache_flash(...)`) **inline before** calling
  `chunked_prefill_paged_decode(...)`. Same Python frame, same CUDA
  stream, no dependency tricks needed.
- **0.19** moved KV-cache write into a separate `do_kv_cache_update`
  method, exposed through a `unified_kv_cache_update` torch op, called
  from `vllm/model_executor/layers/attention/attention.py:Attention.forward`
  *before* `unified_attention_with_output`. The two ops are stitched
  together via a fake `kv_cache_dummy_dep` tensor return value to make
  `torch.ops` dispatch see a data dependency.

On RDNA3 gfx1100, the fake-dep doesn't translate into an actual GPU
stream wait. The attention kernel can read stale KV cache for some
sequences (which is why **the rows that fail vary non-deterministically**
and **layer 0 is always OK** — its KV cache write completes before any
read because there's no prior layer's KV write to race with).

A previous session on this VM found this exact ordering issue (per
`[riscv-patch]` comment in `attention.py`), added an unconditional
`torch.cuda.synchronize()` to fix it, but had to remove it because the
synchronize made cudagraph capture fail with
`hipErrorStreamCaptureInvalidated`. The comment explicitly recommended
the proper guarded form:

```python
if not torch.cuda.is_current_stream_capturing():
    torch.cuda.synchronize()
```

We are now testing that exact fix — applying it between
`unified_kv_cache_update` and `unified_attention_with_output` in both
the direct-call and `torch.ops.vllm` branches of `Attention.forward`.
Patch script: `scripts/instruments/patch-attention-sync.py` (to be
saved if the test confirms the fix).

## Open questions

1. Does the stream-aware sync actually fix the bug at N=4 / N=8 fp16?
   (Test in flight.)
2. If yes, is it specific to ROCm's HIP stream behaviour (and missing
   from the upstream V1 split-op design), or does it manifest on
   NVIDIA too under some conditions?
3. The fix would slow down small batches (sync = full GPU drain).
   Does it cost throughput at higher concurrency?
4. Long-term: should the upstream fake-dep in
   `unified_kv_cache_update` → `unified_attention_with_output` be
   replaced with an explicit cuda event / `torch.cuda.stream.record_event`
   so it works without a global drain?

## Reproducer

From this repo:

```bash
./scripts/sync-launchers.sh

# Broken:
python -m runner.matrix configs/bench-matrix.yaml \
   --only-server vllm-qwen3-4b-fp16-graph-tp1

# Fixed via dtype workaround:
python -m runner.matrix configs/bench-matrix.yaml \
   --only-server vllm-qwen3-4b-bf16-graph-tp1

# Control showing 0.11 fp16 is correct:
python -m runner.matrix configs/bench-matrix.yaml \
   --only-server vllm-qwen3-4b-fp16-graph-tp1-vllm011
```

All instrumentation scripts are in `scripts/instruments/` and can be
re-applied to the live venv to walk the same bisection.
