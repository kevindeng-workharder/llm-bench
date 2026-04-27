# vLLM 0.19 fp16 batched-NaN bug — debug summary

**Status: SOLVED (2026-04-27).** Fix lives in
`scripts/instruments/patch-clamp-v2-zero-degenerate.py`. After applying,
Qwen3-4B fp16 graph TP1 produces 0 garbage streams at N=1/2/4/8. See
[Real fix verified (FULL)](#real-fix-verified-full--zero-degenerate-softmax-rows--nan-safe-clamp)
below for the actual mechanism.

Concise narrative of how we narrowed the bug, what's confirmed, what's
ruled out, and where the investigation stands.

For the full play-by-play (all 14+ instrumentation patches) see
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
  used by both vLLM 0.11 and vLLM 0.19. The kernel itself is not the
  bug — the difference is in **how vLLM 0.19 calls it** (and what
  metadata it passes).
- vLLM 0.11 venv at `/home/ubuntu/ai/lib/.../vllm/`, vLLM 0.19 at
  `/home/ubuntu/ai-2.10/lib/.../vllm/`. Both link the same
  `_rocm_C.so`.
- vLLM 0.19 git source at `/data/vllm-0.19` is `HEAD detached at
  v0.19.0` with only 3 modified files — all build-system patches
  (`CMakeLists.txt`, `cmake/utils.cmake`, `csrc/cuda_vec_utils.cuh`).
  No runtime Python is locally modified. `md5sum` of installed
  `attention.py`, `gpu_model_runner.py`, `qwen3.py`, `triton_attn.py`,
  `builtin.py` matches the source.

## Workaround that works

`--dtype bfloat16` instead of `--dtype float16`. Verified end-to-end
on Qwen3-4B graph TP1:

| Config | N=1 | N=2 | N=4 | N=8 garbage |
|---|---|---|---|---|
| **fp16** graph TP1 | 25 | 55 | 45 ❌ | 209 t/s, 6/8 bad |
| **bf16** graph TP1 | 37 | 57 | 114 | **212 t/s, 0/8 ok** ✅ |

bf16 has fp32-level 8-bit exponent (max ≈ 3.4e38) vs fp16's 5-bit
(max ≈ 65 504). bf16 doesn't trigger the overflow that fp16 does inside
the attention kernel — but the *root* cause of why fp16 overflows
specifically in 0.19 (and not 0.11) is still open.

## What we drilled down to

In-place `os.write(2, ...)` instrumentation on the running 0.19 venv
(scripts in `scripts/instruments/`). Working downward through the
forward pass:

| Layer | Probe | Finding |
|---|---|---|
| `_prepare_input_ids` | v6/v7 (entry trace) | Called with `num_reqs=4`, `prev_sampled_token_ids` present, fast-opt branch taken |
| Fast-opt scatter | v8 (src trace) | `prev_sampled_token_ids[:4, 0] = [real, 0, real, 0]` — already has zeros |
| Sampler output | v9 | `sampled_token_ids` shape=[4,1], rows 1-3 = 0 — sampler innocent |
| Sampler input (logits) | v10 | `max_per_row=[real, NaN, NaN, NaN]` — argmax(NaN) = 0 |
| `compute_logits` input (`sample_hidden_states`) | v11 | rows 1, 3 already NaN before lm_head |
| `Qwen3DecoderLayer` | qwen3-layers-v2 | Layer 0 fully OK; layer 1's `self_attn` is the first to introduce NaN |
| `Qwen3Attention` sub-ops | qwen3-attn | Q/K/V/q_norm/k_norm/RoPE all finite; `self.attn(q,k,v)` output has NaN at rows 1-3 |
| `Attention.forward` (output buffer) | probe-attention-zeros | Pre-fill output with zeros: layer 0 writes ALL 4 rows correctly; layer 1+ writes rows 0,1 correctly + NaN to rows 2,3 |

The kernel writes to all rows (zeros don't survive — they're
overwritten with NaN, not left at 0). Layer 0's attention attends
**only to the current token** (no past KV at decode iter 1) so it can't
hit a wrong-block-table read. Layer 1+ attends to past tokens
including ones layer 0 just wrote, and that's where the failure
appears.

## Hypotheses tested and ruled out

| # | Hypothesis | How tested | Verdict |
|---|---|---|---|
| 1 | MoE expert routing | Dense Qwen3-4B fp16 fails identically | OUT |
| 2 | AWQ / compressed-tensors quant | Dense fp16 fails | OUT |
| 3 | AITER backend | `VLLM_ROCM_USE_AITER*=0` reproduces | OUT |
| 4 | Prefix caching | `--no-enable-prefix-caching` reproduces | OUT |
| 5 | Chunked prefill | `--no-enable-chunked-prefill` reproduces | OUT |
| 6 | CUDA Graphs | `--enforce-eager` reproduces (lower threshold N≥2) | OUT |
| 7 | Async scheduling (PR #27614) | `--no-async-scheduling` reproduces | OUT |
| 8 | Model Runner V2 (PR #25266) | `VLLM_USE_V2_MODEL_RUNNER` default off | OUT |
| 9 | Bookkeeping vectorization (PR #25801) | Never merged | OUT |
| 10 | Sampler bug | Diff vs 0.11 = refactoring; v9/v10 show `argmax` reads NaN logits | OUT |
| 11 | `compute_logits` / lm_head | v11 shows hidden_states already NaN before lm_head | OUT |
| 12 | C++ `paged_attention_rocm` kernel | Same kernel binary works correctly with vLLM 0.11 | OUT (per same-binary constraint) |
| 13 | Attention backend choice | ROCM_ATTN, TRITON_ATTN, no-custom-paged, no-prefill-decode-split — all reproduce | OUT |
| 14 | Our 3 build-system patches | None touch runtime Python | OUT |
| 15 | `torch.ops.vllm.X` dispatch vs direct call | Both paths reproduce | OUT |
| 16 | KV-write/attn-read ordering (0.19 split-op + fake `kv_cache_dummy_dep` + missing CUDA stream sync on RDNA3) | Inserted `if not torch.cuda.is_current_stream_capturing(): torch.cuda.synchronize()` between the two ops in both eager and graph mode | OUT — still 2/4 N=4 garbage |
| 17 | C++ kernel specifically (vs triton) | `use_custom = False` forces Triton path | OUT — Triton path **also** 2/4 garbage |

## What's left — the bug must be in the inputs to the kernels

Both kernel paths (C++ `paged_attention_rocm` AND triton) reproduce, so
the kernels are reading from the right place — they're just reading
wrong data. The remaining suspects are the metadata tensors that
*tell* the kernels where to read:

- `slot_mapping[token_idx] → cache slot` (where to write the new K, V)
- `block_table[seq, block_idx] → physical block` (which physical KV
  blocks belong to each seq)
- `seq_lens[seq] → length so far`
- `query_start_loc[seq] → start offset in query`
- The actual KV cache contents from prior layers / steps

If `slot_mapping` is wrong for seqs 2 and 3 at layer 0, the K, V get
written into the *wrong* physical slots. At layer 1 the kernel reads
from the seq-2/3 block_table-pointed slots and finds garbage (because
the real K, V went elsewhere). Garbage K → softmax(Q @ garbage) → NaN.
This matches the layer-0-OK / layer-1+-broken pattern exactly.

## Notable supporting evidence

- vLLM 0.19 **removed** the `VLLM_ROCM_CUSTOM_PAGED_ATTN` env-var guard
  from `use_rocm_custom_paged_attention` in `platforms/rocm.py`. So 0.19
  always selects the C++ kernel on gfx1100; 0.11 let you fall back via
  env. Earlier "probe A" testing this env had no effect (silently
  ignored).
- vLLM 0.19 split the single inline call site (KV write + attention
  compute) into two separate `torch.ops` (`unified_kv_cache_update` +
  `unified_attention_with_output`). 0.11 did them in one Python frame.
  This split is plausibly where wrong metadata gets baked in, but the
  ordering aspect (H16) was tested and ruled out.
- A previous session's `[riscv-patch]` comment in `attention.py` notes
  an earlier attempt to fix a "0.19 decode regression" with
  `torch.cuda.synchronize()`, removed because cudagraph capture broke.
  We re-tried with the suggested guarded form — didn't fix it. So
  whatever that earlier session was chasing was a different aspect of
  the same bug surface, or a different bug entirely.

## Pinpointed kernel: `context_attention_fwd` (triton prefix-prefill)

Probed `chunked_prefill_paged_decode` itself
(`scripts/instruments/probe-cppd-output-trace.py`): zero-prefill the
output buffer, log per-seq output max BEFORE and AFTER the
`context_attention_fwd(skip_decode=True)` call.

The N=4 batch turns out to be **mixed prefill + decode**:
```
[PROBE #1] num_seqs=4 query_start_loc=[0, 1, 21, 43, 61]
                     seq_lens=[20, 20, 22, 18]  max_query_len=22
```
- seq 0 has 1 token (decode — already prefilled in earlier iter)
- seqs 1, 2, 3 have 22, 20, 18 tokens (this is their initial prefill)

So the call dispatches in two stages inside
`chunked_prefill_paged_decode`:

```python
if max_query_len > 1:
    context_attention_fwd(..., skip_decode=True)   # handles prefill seqs 1-3

if use_custom:
    paged_attention_rocm(...)                      # handles decode seq 0
```

After zero-prefill + after `context_attention_fwd` (PROBE #1, layer 0):
```
seq0 out[0:1]  = [0.0]                     # decode skipped here, kernel didn't write
seq1 out[1:21] = [0.94, 0.93, 0.40, 0.24, inf, inf, 0.56, ...]   # INF at pos 4, 5
seq2 out[21:43]= [0.94, 0.93, 0.40, ..., inf, inf, ..., 391.5, ...]
seq3 out[43:61]= [0.94, 0.93, 0.40, ..., inf, inf, ...]
```

So **layer 0's prefill kernel `context_attention_fwd` already writes
`inf` into the output for every prefill seq**, at specific token
positions. By layer 1+ those `inf`s propagate through residual +
RMSNorm + matmul → **NaN**. The seq-0 decode kernel is innocent.

`context_attention_fwd` lives at:
- `vllm/v1/attention/ops/prefix_prefill.py` in 0.19
- `vllm/attention/ops/prefix_prefill.py` in 0.11

Diffed the two: they're materially the same online-softmax structure
with subtract-max trick (`m_ij = max(qk); p = exp(qk - m_ij)`). Subtle
differences in `m_i` / `l_i` initialization (0.11 starts `l_i=1.0`,
0.19 starts `l_i=0.0`) and the epilogue `acc / (l_i + 1e-10)` — both
algorithmically equivalent in the math but plausibly different at
masked-out boundary tokens or when `l_i` runs out of fp16 range mid-loop.

The exact arithmetic that overflows in fp16 inside the triton kernel
(suspect: a per-block `acc * alpha` rescale that loses fp32 headroom
when stored to fp16 output) needs further triton-internal
instrumentation to nail down. **But the root cause is now confirmed
upstream of attention.py — it's specifically the prefix-prefill triton
kernel, only on the fp16 path, only at the first-layer prefill stage,
and only when prefill and decode are batched together.**

## Real fix verified (FULL) — zero degenerate softmax rows + NaN-safe clamp

**TL;DR**: `acc / (l_i + 1e-10)` blows up to fp16 max **65 504** when
`l_i` is degenerately small (≈ 0 for masked-out / numerically dead
softmax rows). The v1 clamp prevented inf in the kernel output, but
those clamped 65 504 values then went into `o_proj` (a 4096-wide
matmul over fp16 weights) which **summed to a value > 65 504** in fp32
and got cast to fp16 inf. From there it cascaded.

The real fix zeroes out rows whose softmax denominator is degenerately
small — semantically those rows had no valid attention weight anyway —
and clamps with NaN-safe `tl.where`:

```python
acc = acc / (l_i[:, None] + 1e-10)
# v2 fix:
acc = tl.where(l_i[:, None] < 1e-3, 0.0, acc)   # zero degenerate rows
acc = tl.where(acc != acc, 0.0, acc)             # NaN -> 0  (NaN-safe)
acc = tl.where(acc > 65504.0, 65504.0, acc)
acc = tl.where(acc < -65504.0, -65504.0, acc)
tl.store(out_ptrs, acc, ...)
```

Patch script: `scripts/instruments/patch-clamp-v2-zero-degenerate.py`.
Applies on top of (or replacing) the v1 clamp at the same 4 sites.

End-to-end on Qwen3-4B fp16 graph TP1 (multiple runs):

| Config | N=1 | N=2 | N=4 | N=8 |
|---|---|---|---|---|
| **Before any patch** | 0/1 | 0/2 | 2/4 | 6/8 |
| **v1 clamp** (`tl.minimum/maximum`) | 0/1 | 0/2 | 0/4 ✅ | 4/8 |
| **v1 + force-triton** (use_custom=False) | 0/1 | 0/2 | 0/4 ✅ | 2/8 |
| **v2 clamp** (zero-degenerate-rows + tl.where) + force-triton | **0/1** | **0/2** | **0/4** | **0/8 ✅** |
| **v2 clamp ALONE** (native use_custom for C++ paged-attn) | **0/1** | **0/2** | **0/4** | **0/8 ✅** |
| **--dtype bfloat16** (no patch) | 0/1 | 0/2 | 0/4 | 0/8 |

So **v2 clamp fully fixes N=1/2/4/8 on `--dtype float16`**, and the
fix is sufficient on its own — `use_custom=False` (force-triton path)
is NOT required. The C++ `paged_attention_rocm` kernel was always
innocent; the bug was purely in the triton `prefix_prefill._fwd_kernel`
epilogue, and the resulting bad rows were leaking into o_proj which
amplified them to inf.

Throughput with v2 clamp + native C++ paged-attn:
~36 / 58 / 112 / 211 t/s for N=1/2/4/8 — same range as the
original/uncorrupted vLLM 0.19 fp16 path.

**Eager-mode confirmation** (`--enforce-eager`, no CUDA graph capture)
also passes 0/N at every N with v2 clamp applied:
~3 / 4.8 / 13.6 / 25.1 t/s for N=1/2/4/8. This rules out CUDA graphs
as a contributing factor — v2 clamp is genuinely the root-cause fix,
not a graph-replay timing accident.

### How we found it

1. After v1 clamp, ran the v4 NaN/inf probe on every Qwen3 decoder
   layer with `bs >= 8 AND positions != 0` (filters out vLLM's
   `dummy_run` which uses synthetic identical inputs).
2. First inf appeared at **L17 02_after_attn**, rows 27, 28 only.
3. Within 1-2 layers, NaN spread to rows 23-40 via `q · k_cache`.
4. Mechanism: `acc / (l_i + 1e-10)` with `l_i` ≪ 1e-10 produces
   fp32 values like `acc * 1e10`, which the v1 clamp pinned at the
   fp16 ceiling 65 504. Then `o_proj` did `out = attn @ W_o`,
   summing 4096 fp16-multiplied values; the fp32 accumulator grew
   past 65 504 and the output cast to fp16 became inf.
5. v2 fix zeroes the degenerate rows directly, bypassing the
   amplification cascade.

Probe scripts (kept for re-running the bisection):
- `scripts/instruments/probe-find-n8-nan.py` (per-stage NaN/inf
  detector with capture-skip)
- `scripts/instruments/probe-stats-n8-v3.py` / `…v4.py`
  (real-data filter via positions, distinct-row count, NaN-only print)

### Original v1 (insufficient) — kept for context

The triton attention kernels in vLLM 0.19 do their math in fp32 (`acc`,
`l_i`, `m_i` are all `tl.float32`), apply the standard subtract-max
softmax trick, and then in the epilogue do
`acc = acc / (l_i + 1e-10)` and write the result through a fp16
`tl.store` to the output buffer. For some token positions in mixed
prefill+decode batches, `acc / l_i` produces an fp32 value above
`fp16_max ≈ 65 504` — converting that to fp16 yields `inf`. Inf
propagates as NaN one layer later.

Patched the three kernel files in
`vllm/v1/attention/ops/` to clamp the `acc` value into the fp16
representable range RIGHT BEFORE `tl.store`:

```python
acc = acc / (l_i[:, None] + 1e-10)
acc = tl.minimum(tl.maximum(acc, -65504.0), 65504.0)   # llm-bench fp16 fix
tl.store(out_ptrs, acc, ...)
```

Patch sites:
- `prefix_prefill.py` :: `_fwd_kernel` epilogue (line 336)
- `prefix_prefill.py` :: `_fwd_kernel_alibi` epilogue (line 619)
- `triton_prefill_attention.py` :: `_fwd_kernel` epilogue (line 168)
- `chunked_prefill_paged_decode.py` :: `kernel_paged_attention_2d`
  epilogue (line 233)
- `triton_decode_attention.py` :: 3 inline `acc / e_sum` sites
  (lines 178, 413, 574) — wrapped in `tl.minimum/tl.maximum`

Patch scripts:
- `scripts/instruments/patch-prefix-prefill-clamp.py` (the 2-site
  prefix_prefill version)
- `scripts/instruments/patch-all-triton-clamp.py` (the comprehensive
  version covering all 4 files / 7 sites)

Verified end-to-end on Qwen3-4B fp16 graph TP1 with
`scripts/instruments/patch-all-triton-clamp.py` applied:

| Config | N=1 | N=2 | N=4 | N=8 |
|---|---|---|---|---|
| **Before patch** | 0/1 | 0/2 | **2/4 bad** | **6/8 bad** |
| **Triton clamp only** (use_custom on) | 0/1 | 0/2 | **0/4 ✅** | **4/8 bad** |
| **Triton clamp + force-triton** (use_custom=False) | 0/1 | 0/2 | **0/4 ✅** | **2/8 bad** |
| **`--dtype bfloat16`** (no patch) | 0/1 | 0/2 | **0/4 ✅** | **0/8 ✅** |

So **N=1, 2, 4 are FULLY FIXED** by the triton-side clamp alone.

The 省事 path — combining the triton clamp with `use_custom = False`
(see `scripts/instruments/probe-force-triton.py`) — routes every
attention dispatch through the patched triton kernels and never calls
the un-patched C++ `paged_attention_rocm`. That improves N=8 from 4/8
bad to 2/8 bad but does NOT fully fix it.

The residual 2/8 at N=8 is therefore NOT the C++ kernel — it's a
second fp16 overflow path inside the triton kernels themselves,
upstream of the epilogue clamp we added. Most likely candidates:
- `qk = sm_scale * tl.dot(q, k)` producing values that, when later
  cast to fp16 inside `tl.dot(p, v)` via the `p.to(V.dtype)` cast,
  push some intermediate outside fp16 range, OR
- a residual / RMSNorm overflow in a layer activation buffer that's
  shape-dependent on the batch (mixed prefill+decode shapes at N=8
  expose a different chunk size that nudges activations past 65504).

For now `--dtype bfloat16` is the practical fix; the kernel-side
hunt for the N=8 overflow is documented as a follow-up.

## Practical fix paths

1. **Apply the v2 clamp** (`scripts/instruments/patch-clamp-v2-zero-degenerate.py`):
   stays on `--dtype float16`, full N=1/2/4/8 correctness, no
   throughput penalty. **Recommended.** The earlier `use_custom=False`
   force-triton patch is NOT needed — the v2 clamp on the triton
   prefix-prefill kernel is sufficient on its own; the C++
   `paged_attention_rocm` decode kernel runs as upstream intends.
2. **Use `--dtype bfloat16`** if you don't want to patch vLLM.
   Same end result; bf16's wider exponent absorbs the overflow
   naturally.
3. **Long-term upstream fix:** the v2 clamp pattern (zero degenerate
   softmax rows + NaN-safe `tl.where` clamp) belongs in every
   `acc / (l_i + eps)` epilogue in vLLM's triton attention kernels
   when the output dtype is fp16. No C++ kernel change needed.

### Reverting all debug patches

`scripts/instruments/revert-all-debug-patches.py` rolls back every
patch we applied during this investigation:
- restores `qwen3.py` from `.before-instrument`
- restores `prefix_prefill.py` from `.before-clamp`
- restores `chunked_prefill_paged_decode.py` from
  `.bak-before-fix-20260424`
- strips the v1 / v2 clamp blocks in place from
  `triton_prefill_attention.py` and `triton_decode_attention.py`
  (those have no useful pre-clamp backup)

Importantly, the script leaves `.pre-riscv-patch` files alone — those
are upstream-vanilla and reverting to them would re-break the
riscv64 cross-compile environment.

## Reproducer

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
