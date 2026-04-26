# vLLM 0.19 batched-correctness regression on riscv64 + ROCm gfx1100

**Status:** **ROOT CAUSE FOUND 2026-04-27 — fp16 NaN overflow inside the
model forward.** Workaround: use `--dtype bfloat16`.

## TL;DR

Running vLLM v0.19.0 with `--dtype float16` on this stack causes
**NaN to appear in the model's `hidden_states` output** for some rows
of the batched forward when batch ≥ 2. Sample in-flight
instrumentation:

```
[V11-HS] shape=[4, 2560]
  max_per_row=[17.656, NaN, 38.5, NaN]
  first3=[[0.017, -1.6, 3.55], [nan, nan, nan],
          [-0.044, 1.05, -0.91], [nan, nan, nan]]

[V10-PRE-SAMP] shape=[4, 151936]
  max_per_row=[29.031, NaN, NaN, NaN]
  argmax_per_row=[198, 0, 0, 0]
```

The NaN propagates through `compute_logits` → argmax(NaN-row) returns
token id 0 → streamed as `!`. With `--dtype bfloat16`, `garbage` drops
from 2/4 → **0/4** at N=4 (and 6/8 → 0/8 at N=8). Same model, same
flags, same hardware.

bfloat16 has the same exponent range as fp32 (8-bit), while fp16 is 5-bit
(max ≈ 65 504). Some operation in the model's forward (most likely the
attention softmax pre-scale, or an RMSNorm with very large activations)
overflows fp16 → ±inf → propagates to NaN — but only for some batch rows
because the broken matmul kernel only writes correct output for one row
when batch ≥ 2 on this gfx1100 build.

vLLM 0.11 with the same `--dtype float16` is correct on this stack — so
either a numerical-stability subtract-max trick was removed in 0.19, or a
new fp16 kernel was wired in for ROCm that doesn't guard against
overflow.

## How we got here (instrumentation chain)

Pinpointed step-by-step by patching `gpu_model_runner.py` with
`os.write(2, ...)` debug prints (raw fd write to bypass logger
buffering):

1. `_prepare_input_ids` fast-opt branch: `prev_sampled_token_ids[:4, 0]`
   already had stride-2 zeros → bug is upstream of `_prepare_input_ids`.
2. `_sample` output: `sampled_token_ids` has only row 0 with valid
   token, rows 1–3 = 0 → bug is upstream of sampler.
3. `_sample` input (logits): `max_per_row=[real, NaN, NaN, NaN]` →
   logits already NaN. Bug is upstream of `compute_logits` (lm_head).
4. `sample_hidden_states[logits_indices]`: rows 1, 3 are NaN. Bug is
   inside `model.forward` itself.
5. Switching `--dtype float16` → `--dtype bfloat16` makes garbage = 0/4
   without any other code change. → fp16 overflow inside model forward.

The instrumentation patches live in `scripts/instrument-019-v*.py` so
the chain is reproducible.

**Earlier write-ups in this file claimed the bug was MoE-specific, then
sampler-specific, then attention-specific — all wrong. The real cause
is fp16 numerical overflow inside the model forward, which the broken
matmul / norm kernels on this gfx1100 build don't recover from for
batch > 1.** The MoE / quant / attention symptoms were just the bug
showing up in different downstream consumers of the corrupt
hidden_states.

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

## Diff against upstream — what we changed (2026-04-27)

Asked: did our local patches to vLLM 0.19 cause this regression?

**No.** Hard-checked the installed venv against the source tree:

```
$ cd /data/vllm-0.19 && git status
HEAD detached at v0.19.0
Changes not staged for commit:
        modified:   CMakeLists.txt
        modified:   cmake/utils.cmake
        modified:   csrc/cuda_vec_utils.cuh
```

Three modifications, **all build-system / cross-compile fixes**:

1. `CMakeLists.txt` + `cmake/utils.cmake` (`get_torch_gpu_compiler_flags`):
   add `-D_GLIBCXX_NO_ASSERTIONS -U_GLIBCXX_HAVE_IS_CONSTANT_EVALUATED`
   to the GPU compile flags (libstdc++ HIP-mode workaround documented as
   Patch 1 in MEMORY.md).
2. `cmake/utils.cmake` (`define_extension_target`): add
   `-DUSE_ROCM -D__HIP_PLATFORM_AMD__ -D__HIP_PLATFORM_HCC__` and the
   ROCm include path for HIP-language targets (Patch 7).
   `run_python` tolerates non-zero exit when stdout is non-empty.
3. `csrc/cuda_vec_utils.cuh`: include `<hip/hip_bf16.h>` and
   `<hip/hip_fp16.h>` under `USE_ROCM`.

Site-packages content matches the source tree exactly (verified via
`md5sum` on `vllm/v1/sample/logits_processor/builtin.py` and
`vllm/v1/attention/backends/triton_attn.py` — same hash as
`/data/vllm-0.19/...`). The `.bak` files in site-packages turned out to
be stale snapshots from a previous wheel install, not patches.

**No runtime-Python files are modified.** The bug is in upstream
v0.19.0 itself — our build-system patches just make it compile on
riscv64 + ROCm.

Verified by reverting all three build-system patches and rebuilding
would not change anything observable here, since they're either
(a) compile flags that gate which kernels build (without them the build
fails entirely) or (b) header includes (without them, missing types).
The runtime path that's broken (sampler / model_runner / scheduler) is
all pure Python and is unmodified.

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
| **D** | `--no-async-scheduling` | 0/1 | 0/2 | **2/4** | **6/8** |

Conclusions from the ablation:

- The bug is **NOT in the attention backend choice**. Every variant
  reproduces it.
- C++ `paged_attention_rocm` is actually *closer to correct* than the
  triton paged path (probe A is the worst at 7/8 garbage).
- `prefill/decode split` doesn't matter (probe C with split off behaves
  like the default — both fail at N≥4).
- **Async scheduling is NOT the cause**. `--no-async-scheduling`
  (probe D) keeps `'async_scheduling': False` and logs "Asynchronous
  scheduling is disabled." — yet N=4 still 2/4 garbage and N=8 still
  6/8. Strong signal that PR #27614 (which flipped async-sched to
  default-on) is not our regression vector.
- This narrows the bug *out* of attention AND scheduling; the most
  likely remaining surface is the V1 input-batch / model-runner row
  packing itself (`gpu_model_runner.py`'s `_prepare_inputs` /
  `execute_model` family), or a graph-capture static-input rebind bug
  for batch-size ≥ 4.

The "**exactly one row of N gives valid output, the rest stream
`!`** (token id 0) consistently across all backends" pattern strongly
suggests the model produces correct logits for **only one row of the
batched forward output**; the other N−1 rows are zeros / NaN /
uninitialized, and argmax(0...) → token 0 → `!`.

That's most consistent with an off-by-one in the per-row scratch buffer
or hidden-state addressing somewhere in the shared model-runner
plumbing — not in any one attention kernel.

## Upstream issues with the same shape (2026-04-27)

The `!`-token degenerate-output failure mode at concurrent batch is a
recurring vLLM bug pattern, NOT specific to riscv64 + ROCm. None of the
following have a merged fix:

| # | Title | Model | Hardware | vLLM | State |
|---|---|---|---|---|---|
| [#13035](https://github.com/vllm-project/vllm/issues/13035) | "Llama-3.1-405B-Instruct-FP8 only generates exclamation marks" | Llama 3.1 405B FP8 | NVIDIA | 0.6+ | "downgrade to 0.6.6 works" |
| [#17652](https://github.com/vllm-project/vllm/issues/17652) | "Degradation of Qwen/Qwen3-30B-A3B performance depending on batch size" | Qwen3-30B-A3B | A100 | 0.8.5 | closed not-planned, repeating-token garbage at batch=50 |
| [#18252](https://github.com/vllm-project/vllm/issues/18252) | "Qwen3 uses vllm automatic batch inference to abnormal output" | Qwen3-4B | A800 | 0.8.5 | closed stale, batched garbage / single OK |
| [#27364](https://github.com/vllm-project/vllm/issues/27364) | "Qwen3-VL {4B,8B} FP8 returns only exclamation marks (`!!!!!...`)" | Qwen3-VL-{4,8}B-FP8 | Jetson Thor | 0.11 | open, FP8 KV cache angle |
| [#36010](https://github.com/vllm-project/vllm/issues/36010) | "Qwen3.5-27B Batch Inference very slow / not working" | Qwen3.5-27B | NVIDIA | recent | open |
| [#38527](https://github.com/vllm-project/vllm/issues/38527) | "Qwen3.5-35B-A3B-FP8 outputs all exclamation points" | Qwen3.5-35B-A3B-FP8 | RTX Pro 6000 Blackwell | 0.18.0 | open, no diagnosis yet |
| [vllm-ascend #5313](https://github.com/vllm-project/vllm-ascend/issues/5313) | "Qwen3-VL-32B exclamation marks for video inference" | Qwen3-VL-32B | Ascend | 0.11.0 | open |

Across these reports, the common thread is:

- The target model is mid-large-ish (4B – 405B).
- Batch-size 1 / single-request is fine.
- Concurrent batched inference produces token-id-0 (`!`) loops or
  near-degenerate single-token loops.
- The bug spans NVIDIA (A100, A800, Blackwell, Jetson Thor), AMD ROCm
  (us, gfx1100), and Ascend.
- It spans dense fp16/bf16, FP8, GPTQ, AWQ — not quant-specific.
- No fix has shipped. The matching issues are typically closed as
  "stale" / "not planned" by the bot.

Our setup adds one more data point to this pile: dense fp16 Qwen3-4B on
v0.19.0 with riscv64 + ROCm gfx1100. The cross-section of (vLLM 0.19,
plain dense, no spec-decoding, no MoE, no FP8/AWQ, no Mamba/GDN) is the
cleanest minimal repro any of these reports has — so it's worth filing
upstream as a fresh issue with the bench harness as the reproducer.

## Suspects ruled in / out (final)

| Suspect | Ruled out by | Status |
|---|---|---|
| MoE expert routing | dense Qwen3-4B fails | OUT |
| Quantization (AWQ, GPTQ, FP8) | dense fp16 fails | OUT |
| Triton attention kernel | TRITON_ATTN backend probe | OUT |
| C++ rocm_paged_attention | `VLLM_ROCM_CUSTOM_PAGED_ATTN=0` probe | OUT |
| Prefix caching | `--no-enable-prefix-caching` | OUT |
| Chunked prefill | `--no-enable-chunked-prefill` | OUT |
| AITER backends | `VLLM_ROCM_USE_AITER*=0` | OUT |
| CUDA Graphs | `--enforce-eager` reproduces | OUT |
| Sampler.py | diff vs 0.11 is refactoring only | OUT |
| Async scheduling (PR #27614) | `--no-async-scheduling` reproduces | OUT |
| Model Runner V2 (PR #25266) | env default-off in v0.19 | OUT |
| Bookkeeping vectorization (PR #25801) | never merged | OUT |
| Our 3 build-system patches | no runtime Python changed | OUT |
| **V1 input-batch packing / persistent batch / `_prepare_inputs`** | not yet probed | **prime suspect** |
| **CUDA-graph capture per batch-size shape** | partial-rule-out (eager also fails, lower threshold N≥2) | secondary |

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
