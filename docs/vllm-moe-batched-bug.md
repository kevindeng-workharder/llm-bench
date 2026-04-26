# vLLM Qwen3-MoE AWQ batched-fused-MoE corruption (riscv64 + ROCm gfx1100)

**Status:** open. Verified 2026-04-26 with vLLM `0.19.1.dev0+g2a69949bd`,
torch `2.10.0+riscv64.rocm`, triton `3.4.0`, AMD Radeon RX 7900 XTX (gfx1100,
RDNA3, wave32), running inside QEMU riscv64 with VFIO PCI passthrough.

## Symptom

When the engine batches **N ≥ 2** concurrent requests against
`Qwen3MoeForCausalLM` with **compressed-tensors W4A16** (a.k.a. "AWQ" with
`pack-quantized` format), exactly **one** request in the batch returns valid
output. Every other request streams a single token (`!`, token id 0) for the
full `max_tokens` budget.

```
=== eager N=4 unique temp=0 ===
  [0]  60t   1.39t/s  ttft=2.81s  GARBAGE  '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
  [1]  60t   1.39t/s  ttft=2.14s  ok       'The Rust borrow checker is a compile-time...'
  [2]  60t   1.39t/s  ttft=2.81s  GARBAGE  '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
  [3]  60t   1.39t/s  ttft=2.81s  GARBAGE  '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
```

The "lucky" slot is non-deterministic — depends on the order in which
requests land in the engine scheduler. Slot index, prompt content, and
prompt length do not predict it (verified across 7 controlled probes; see
`scripts/seed-2026-04-26.py` for the raw observations).

## Reproduction

1. Bring up the riscv64 VM with `start_ubuntu_vfio_dual.sh` (or the
   single-GPU variant). Confirm `torch.cuda.device_count()` returns 2 and
   `nvidia-smi`-equivalent reports the 7900 XTX(s).
2. Launch:
   ```
   bash servers/vllm/qwen3-30b-awq-eager-tp1.sh
   ```
3. From the host (with the SSH `-L 8000:localhost:8000` tunnel up):
   ```
   python -m runner.bench -n 2 -t 60 -T 0 --unique
   ```

## What is **not** the cause (verified)

| Hypothesis | Test | Result |
|---|---|---|
| CUDA-Graph capture/replay corrupts batched inputs | rerun in `--enforce-eager` | Same garbage |
| Prefix caching cross-contaminates KV blocks | `--no-enable-prefix-caching` | Same garbage |
| Chunked prefill mixes prefill+decode incorrectly | `--no-enable-chunked-prefill` | Same garbage |
| AITER MoE backend bug | `VLLM_ROCM_USE_AITER_MOE=0` (and full AITER off) | Same garbage |
| Fused-grouped-topk routing kernel bug | `VLLM_USE_FUSED_MOE_GROUPED_TOPK=0` | Same garbage |
| MoE expert padding alignment | `VLLM_ROCM_MOE_PADDING=0` | Same garbage |
| Sampler nondeterminism | `temperature=0` | Same garbage |
| Prompt content / slot index | 7 controlled prompt-permutation probes | Always exactly 1 ok, position varies |

## Working workaround

```
--max-num-seqs 1   # in addition to graph/eager mode
```

Forces the engine scheduler to process one request at a time. All outputs
become correct (`4/4 ok`) but throughput drops to ~0.9 tok/s aggregate at
N=4 — strictly worse than the single-request baseline because each request
now waits for the previous one to finish.

## Where the bug lives (best guess)

- `vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.CompressedTensorsWNA16MoEMethod.apply`
  → `vllm.model_executor.layers.fused_moe.fused_experts`
- The Triton kernel `fused_moe_kernel` (or its W4A16 specialization) called
  via `invoke_fused_moe_kernel`. The "Using default MoE config" warning at
  startup
  ```
  Config file not found at .../E=128,N=768,
    device_name=AMD_Radeon_RX7900_XTX,dtype=int4_w4a16.json
  ```
  means the kernel runs with auto-defaulted block sizes. Default block sizes
  are normally only a *performance* concern — the fact that we get
  *correctness* failures suggests one of:

  1. The default block sizes interact badly with `moe_align_block_size` to
     produce malformed `sorted_token_ids` / `expert_ids` for `M ∈ [2, ...]`.
  2. The `naive_block_assignment` short-circuit (skipped here because
     `block_shape[1] > 0` for int4) is the *only* code path that handles
     batches correctly, and the regular path has an indexing bug for
     `top_k > 1` and `M > 1`.
  3. RDNA3 wave32 vs CDNA wave64 sizing — Triton's autotuned defaults may
     assume wave64.

## Next steps to fully diagnose

1. Dump the inputs/outputs of `fused_moe_kernel` for `M=1` vs `M=2` and diff
   the resulting `topk_output` row-by-row.
2. Try copying `E=128,N=768,device_name=Radeon_8060S_Graphics,dtype=int4_w4a16.json`
   (Strix Halo iGPU, also RDNA3) to `device_name=AMD_Radeon_RX_7900_XTX` —
   if it fixes correctness, the bug is in the auto-default block sizing.
3. Bisect vLLM commits against the `0.11` baseline that was previously
   verified to batch correctly (with Qwen2.5-14B dense fp16 — but never
   exercised with MoE).
4. Run the same model on an x86 ROCm host to confirm the bug is RDNA3- /
   riscv-specific or upstream.

## Comparison to llama.cpp on the same hardware

llama.cpp running `Qwen3.6-35B-A3B-MXFP4_MOE.gguf` on the same single 7900 XTX
batches **correctly** at N=1, 4, and 8 (no garbage), with aggregate
throughput peaking at ~16 tok/s at N=4. So the hardware itself, the ROCm
runtime, and the gfx1100 device libraries are not at fault — this is a
vLLM-specific kernel bug.
