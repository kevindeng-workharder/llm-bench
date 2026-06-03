# p2p-ib · 27B INT8 eager — verified run + bottleneck diagnosis (2026-06-03)

## Setup

- Model `/data/Qwen3.6-27B-Quark-W8A8-INT8` (29 GB on disk) — **Qwen3-Next**
  architecture: Gated-DeltaNet linear-attention hybrid, VL (vision-language).
- `--quantization quark --dtype bfloat16 --enforce-eager --max-model-len 2048`,
  TP=2 across two QEMU/riscv64 guests over CX-7 RoCE (`NCCL NET=IB`),
  GDR disabled (host-bounce). Leader = api_server (node 0); follower =
  `vllm serve --headless` (node 1).
- Stack: vLLM `v0.21.1.dev0+gad7125a43`, RCCL 2.27.7 (`96a25b5+`), ROCm 7.2.3,
  PyTorch 2.11, Python 3.13, kernel `6.19.5-p2p`.

## Outcome

| | result |
|---|---|
| deploy / serve | ✅ correct output (`The capital of France is` → ` Paris.`) |
| startup to ready | ≈ **690 s** (27B weight load ~13.5 GB/rank + VL encoder init; eager has **no** cudagraph-capture phase) |
| OOM | none at gpu-mem-util 0.85 (27B INT8 fits, ~13.5 GB/rank) |
| **decode throughput** | **0.288 tok/s** (≈ 3.5 s/token) |
| TTFT (prefill) | ≈ 5 s |

Throughput (bench.py, streaming, N=1, 32 tok, warm):

```
run1: TTFT 5.2s  decode 0.290 tok/s (3.4 s/tok)
run2: TTFT 4.4s  decode 0.279 tok/s (3.6 s/tok)
run3: TTFT 5.5s  decode 0.295 tok/s (3.4 s/tok)
DECODE avg=0.288  min=0.279  max=0.295 tok/s   (very tight → steady state, not JIT noise)
```

For contrast the 4B/graph baseline on the same path is ~10 tok/s — eager 27B is
~35× slower. The two penalties (eager vs graph, 4B vs 27B) are separable; this
test isolates the *eager* one below.

## Bottleneck: eager kernel-dispatch bound on the riscv64 host CPU

**Not** GPU-compute-bound, **not** primarily IB-bound. Evidence from `profile.sh`:

### A) GPU busy% is bimodal and oscillating

Two independent 20-sample windows:

```
window 1:   VM1 ~15-22%      VM2 ~100%
window 2:   VM1 ~100%        VM2 ~15-17%
```

One rank's GPU sits at ~20% (starved — waiting for the CPU to launch the next
kernel); the *other* sits at ~100% (its NCCL all_reduce kernel **spin-waiting**
for the dispatch-laggard peer over the host-bounce path). Which rank is the
laggard flips between windows — consistent with the two guests contending for
host CPU (the host also drives every GPU↔NIC DMA, since GDR is off).

### B) py-spy: both Workers' MainThread are busy LAUNCHING kernels

Both ranks, MainThread 8/10 `active+gil`, in the same frames:

```
top frames (10 dumps each rank):
  QuarkW8A8Int8.apply_weights          triton_scaled_mm
  TritonInt8ScaledMMLinearKernel...    scaled_int8_quant
  GatedDeltaNetAttention.forward_hip   RowParallelLinear.forward
  ... all_reduce frames appear only ~1/10 samples ...
```

The instantaneous top-of-stack (VM2):

```
current_device (torch/cuda/__init__.py:1149)         ← per-launch overhead
JITFunction.run (triton/runtime/jit.py:551)          ← Triton kernel launch
triton_scaled_mm (.../triton_scaled_mm.py:200)
TritonInt8ScaledMMLinearKernel.apply_weights
QuarkW8A8Int8.apply_weights
QuarkLinearMethod.apply
RowParallelLinear.forward
GatedDeltaNetAttention._output_projection (.../gdn_linear_attn.py:683)
GatedDeltaNetAttention.forward_hip
Qwen3NextDecoderLayer.forward
...
```

The MainThread spends its time **inside the Triton/CUDA kernel-launch machinery**
(`JITFunction.run`, `current_device`), not blocked in `all_reduce`. So
communication is mostly *hidden behind / waited-on by* the dispatch bottleneck,
not the primary cost.

### Why — and the amplifiers

Per decode token, Qwen3-Next (~48 layers, mostly Gated-DeltaNet) issues on the
order of **~1000 individual kernel launches**, each paying full Python+Triton
launch cost on a slow, contended riscv64 host CPU:

- **Quark INT8 via Triton fallback.** The log warns `AITER is not found or
  QuarkOCP_MX is not...`, so the W8A8 path uses the **Triton** `scaled_mm` route:
  every linear = a `scaled_int8_quant` kernel **plus** a `triton_scaled_mm`
  kernel (2+ launches instead of 1 fused INT8 GEMM).
- **Qwen3-Next / Gated-DeltaNet.** Many layers; the linear-attention scan
  (`gdn_attention_core`, `forward_hip`) adds its own multi-kernel sequence.

~1000 launches × ~3 ms riscv64 launch latency ≈ **3 s/token** — matches the
measured 3.5 s/token.

## Why graph mode would fix it (and why we're in eager anyway)

cudagraph captures the whole ~1000-kernel decode step into **one** replay,
erasing the per-launch CPU cost — exactly why the 4B/graph path reaches
~10 tok/s. But cudagraph for the **Gated-DeltaNet hybrid across two VMs** is the
very thing the project's cross-VM cudagraph-deadlock saga was fighting. **Eager
is the deadlock-safe fallback; ~0.29 tok/s is the price of that safety.**

## Levers (eager-compatible, untested here)

1. **Get AITER working** — removes the Triton 2-kernel INT8 fallback (the #1
   hotspot) in favour of a fused INT8 GEMM → fewer launches/token. Most direct.
2. **Fewer host-CPU contenders** — the two guests + host-bounce DMA all share
   host cores; pinning/isolating would reduce the dispatch stall.
3. **cudagraph** — the real fix, gated on solving the hybrid-model cross-VM
   deadlock (e.g. the single-stream avoidance from the project history).

## How this was measured

`bench.py` (streaming, prefill/decode split) and `profile.sh` (GPU busy% +
py-spy MainThread sampling on both ranks) in this directory. py-spy on riscv64
is Python-stack-only (`--native` unsupported) — sufficient to localise the
launch-path hotspot.
