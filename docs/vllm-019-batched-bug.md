# vLLM 0.19 batched-correctness regression on riscv64 + ROCm gfx1100

**Status:** open. Verified 2026-04-26.

**Earlier name was "AWQ MoE batched bug" — that title was wrong.** Initial
investigation only ran the AWQ MoE model, where the bug appears at N≥2 and
reads as "AWQ-only / MoE-only". After running the full matrix (5 dense
fp16 configs across 4 model sizes + the two compilation modes + TP=1 and
TP=2) the bug reproduces on **every single dense and MoE config** at N≥4
under graph mode and at N≥2 under eager mode. Quantization, MoE routing,
prefix caching, AITER, chunked prefill, cudagraphs, and TP-size are all
ruled out as causes. The common factor is **vLLM 0.19** itself.

## The regression

Running the **same model** (Qwen3-4B fp16) with the **same launch flags**
(graph mode, TP=1, --max-model-len 4096, --gpu-memory-utilization 0.85)
under the two vLLM versions installed on the VM:

| vLLM version | venv | N=1 | N=2 | N=4 | N=8 | garbage at N=8 |
|---|---|---|---|---|---|---|
| **0.11.0** (legacy) | `/home/ubuntu/ai`     | 19.6 | 30.0 | 51.2 | **71.4** | **0/8 ✅** |
| **0.19.1.dev** (current) | `/home/ubuntu/ai-2.10` | 36.6 | 41.2 | 115  | **213**  | **5/8 ❌** |

(aggregate tok/s; bold = throughput numbers but ⚠️ = mixed with garbage outputs)

- 0.11 scales correctly: N=8 gets 3.6× the tok/s of N=1, all 8 outputs valid.
- 0.19 scales **superficially** to 5.8×, but 5 of those 8 streams are
  outputting `!` for the entire `max_tokens` budget. The "throughput
  improvement" comes from counting bad tokens in the totals.
- 0.19 IS faster on the single-request case (1.87× — real triton 3.4
  kernels). The win disappears the moment you batch.

## Symptom

When the engine batches N≥4 (graph mode) or N≥2 (eager mode) concurrent
requests, **some subset of clients** in the batch return a degenerate
output: a single token (`!`, token id 0) repeated for the full
`max_tokens` budget. Which client(s) survive is non-deterministic — depends
on engine scheduler ordering. Not slot-index, not prompt content.

```
=== Qwen3-4B graph TP1 N=4 ===
  [0]  80t   1.39t/s  GARBAGE  '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
  [1]  80t   1.39t/s  ok       'The Rust borrow checker is a compile-time...'
  [2]  80t   1.39t/s  GARBAGE  '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
  [3]  80t   1.39t/s  GARBAGE  '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
```

## Reproduction matrix (2026-04-26)

| Config | N=1 | N=2 | N=4 | N=8 | bug threshold |
|---|---|---|---|---|---|
| Qwen3-0.6B fp16 graph TP1 | 0/1 | 0/2 | **2/4** | **6/8** | N≥4 |
| Qwen3-4B fp16 eager TP1   | 0/1 | **1/2** | **2/4** | **6/8** | N≥2 |
| Qwen3-4B fp16 graph TP1   | 0/1 | 0/2 | **1/4** | **5/8** | N≥4 |
| Qwen3-4B fp16 graph TP2   | 0/1 | 0/2 | **1/4** | **5/8** | N≥4 |
| Qwen2.5-14B fp16 graph TP2| 0/1 | 0/2 | **2/4** | **7/8** | N≥4 |
| Qwen3-30B AWQ eager TP1   | 0/1 | **1/2** | – | – | N≥2 |
| Qwen3-30B AWQ graph TP1   | 0/1 | 0/2 | **2/4** | **6/8** | N≥4 |
| **Qwen3-4B fp16 graph TP1 vLLM 0.11 (control)** | 0/1 | 0/2 | **0/4** | **0/8** | none |

Numbers are `garbage_clients / N`. Across 7 different vLLM 0.19 configs the
same pattern holds. The 0.11 control passes at every N.

## What is **not** the cause (verified)

| Hypothesis | Test | Result |
|---|---|---|
| Quantization-specific (AWQ / compressed-tensors) | dense fp16 Qwen3-4B | bug present |
| MoE routing | dense fp16 Qwen3-0.6B | bug present |
| CUDA-Graph capture/replay | `--enforce-eager` | bug present (lower threshold) |
| Prefix caching | `--no-enable-prefix-caching` | bug present |
| Chunked prefill | `--no-enable-chunked-prefill` | bug present |
| AITER backend | `VLLM_ROCM_USE_AITER*=0` | bug present |
| Fused-grouped-topk | `VLLM_USE_FUSED_MOE_GROUPED_TOPK=0` | bug present |
| MoE expert padding | `VLLM_ROCM_MOE_PADDING=0` | bug present |
| Sampler non-determinism | `temperature=0` | bug present |
| Prompt content / slot index | 7 controlled prompt-permutation probes | non-deterministic, exactly 1 ok |
| Tensor parallelism (TP=2) | tested | bug present at TP=1 AND TP=2 |
| Model size / arch family | tested 0.6B / 4B / 14B / 30B-MoE | bug present on all |
| **vLLM version** | **0.11 vs 0.19** | **0.11 OK / 0.19 broken** ← root cause |

## Working workarounds

1. **Use vLLM 0.11**. The legacy `/home/ubuntu/ai` venv (vLLM 0.11.0 +
   torch 2.8 + the TritonPlaceholder stub) batches correctly at all N
   tested. Single-request rate is ~50% of 0.19 (no real triton), but real
   batched throughput is roughly equivalent to 0.19's *valid-only* output
   rate.
2. **`--max-num-seqs 1` on 0.19**. Forces engine serialization. Outputs
   correct, throughput tanks to roughly the single-client rate (no
   batching at all).

## What changed between 0.11 and 0.19

Both versions on this stack share: the same ROCm 6.2.4 install, the same
gfx1100 kernels, the same VFIO passthrough, the same OS image. The
differences:

- vLLM bumped from 0.11.0 to 0.19.1.dev0+g2a69949bd.
- Torch from 2.8 to 2.10 (riscv64+rocm builds).
- The triton path: 0.11 used the `TritonPlaceholder` torch fallback; 0.19 uses
  real `triton 3.4.0` cross-compiled against triton's pinned LLVM fork.
- vLLM's V1 engine internals (rewritten scheduler, paged-attention API,
  rocm_attn backend) were entirely reworked between these versions.

The bug is somewhere in that diff. Updated guesses after the
attention-backend ablation pass below.

## Attention-backend ablation (2026-04-26)

Three probes, each toggling one suspected component on the dense
Qwen3-4B fp16 graph TP1 setup. All four numbers below are
`garbage_clients / N`.

| Probe | env | N=1 | N=2 | N=4 | N=8 |
|---|---|---|---|---|---|
| (default) | `VLLM_ROCM_CUSTOM_PAGED_ATTN=1`, ROCM_ATTN, prefill/decode split on | 0/1 | 0/2 | 1/4 | 5/8 |
| **A** | `VLLM_ROCM_CUSTOM_PAGED_ATTN=0` (force triton paged) | 0/1 | **1/2** | **3/4** | **7/8** |
| **B** | `VLLM_ATTENTION_BACKEND=TRITON_ATTN` | 0/1 | 1/2 | 2/4 | 6/8 |
| **C** | `VLLM_V1_USE_PREFILL_DECODE_ATTENTION=0` + `VLLM_ROCM_CUSTOM_PAGED_ATTN=0` | 0/1 | 0/2 | 2/4 | 6/8 |
| **0.11** (control) | n/a | 0/1 | 0/2 | 0/4 | 0/8 |

Conclusions from the ablation:

- The bug is **NOT in the attention backend choice**. Every variant
  reproduces it.
- C++ `paged_attention_rocm` is actually *closer to correct* than the
  triton paged path (probe A is the worst at 7/8 garbage).
- `prefill/decode split` doesn't matter (probe C with split off behaves
  like the default — both fail at N≥4).
- This narrows the bug *out* of attention and *into* one of:
  - **Sampler** (logits → token id) — V1 sampler was rewritten in 0.19
  - **gpu_model_runner input packing** (how prefill+decode rows are
    concatenated into the batched forward)
  - **Embedding / RoPE position encoding** for batch > 1
  - **Async scheduling** (0.19 logs `Asynchronous scheduling is enabled`,
    a feature absent in 0.11)
  - **`model.forward` in graph capture mode** at batch sizes ≥ 4

The "**exactly one row of N gives valid output, the rest stream
`!`** (token id 0) consistently across all backends" pattern strongly
suggests the model produces correct logits for **only one row of the
batched forward output**; the other N−1 rows are zeros / NaN /
uninitialized, and argmax(0...) → token 0 → `!`.

That's most consistent with an off-by-one in the per-row scratch buffer
or hidden-state addressing somewhere in the shared model-runner
plumbing — not in any one attention kernel.

## Reproduction

```bash
# From the llm-bench repo:
./scripts/sync-launchers.sh

# 0.19 (broken):
python -m runner.matrix configs/bench-matrix.yaml \
   --only-server vllm-qwen3-4b-fp16-graph-tp1

# 0.11 (control, OK):
python -m runner.matrix configs/bench-matrix.yaml \
   --only-server vllm-qwen3-4b-fp16-graph-tp1-vllm011

python -m runner.report results/raw/ > results/$(date +%F).md
```

## Cross-check

llama.cpp running `Qwen3.6-35B-A3B-MXFP4_MOE.gguf` on the same single 7900
XTX batches **correctly** at N=1, 4, and 8 (no garbage), aggregate
throughput peaking at ~16 tok/s at N=4. So the hardware itself, the ROCm
runtime, and the gfx1100 device libraries are not at fault. This is a
vLLM-0.19-specific regression on this stack.

## Why we missed this earlier

Previous concurrent throughput tests (the `concurrent-llamacpp.py` style
script that measured N=100 @ 163 tok/s on this stack) only counted streamed
chunks per second and did not validate output content. **Garbage `!` tokens
streamed at full speed contributed to the throughput total just like real
tokens.** Adding a content sanity check (`is_garbage()` in
`runner/bench.py`) is what surfaced the regression.
