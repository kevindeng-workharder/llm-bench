# vLLM 0.19 fp16 batched-NaN bug — debug summary

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

Practical fix paths:
1. **Use `--dtype bfloat16`.** Already verified: 0/8 garbage at N=8.
   Same throughput.
2. **Disable mixed prefill+decode batching** so prefill never lives
   alongside decode. There's no clean knob for this in 0.19; would
   need a scheduler change.
3. **Patch `context_attention_fwd`** to keep the running accumulator
   in fp32 and only cast to fp16 at `tl.store` time — should avoid
   the mid-loop overflow.

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
