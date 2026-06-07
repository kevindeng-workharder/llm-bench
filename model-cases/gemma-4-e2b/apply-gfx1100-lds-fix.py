#!/usr/bin/env python3
"""Apply the gfx1100 (RDNA3) 64 KB-LDS fix to vLLM's Triton unified-attention.

Problem
-------
Gemma's attention uses head_dim = 256 (Qwen etc. use 128). vLLM's V1 ROCm
TRITON_ATTN kernel (`triton_unified_attention.py`) stages its K and V tiles
(`TILE_SIZE x HEAD_SIZE_PADDED`) plus a small per-block buffer in LDS. With
head_dim 256 the K+V tiles alone need ~64 KB at TILE_SIZE=32, so the kernel
asks for 66,560 B of shared memory while gfx1100 (RDNA3) has only 65,536 B
(64 KB) per CU -> `triton.runtime.errors.OutOfResources` -> EngineDeadError ->
first inference 500s. (V1 ROCm has no working TORCH_SDPA fallback; it ignores
VLLM_ATTENTION_BACKEND and always uses TRITON_ATTN.)

Two scoped, correctness-safe changes, both gated on head_size >= 256 so nothing
else (Qwen 128-dim) is affected:

  1. _get_tile_size: return TILE_SIZE 16 (not 32) for head_dim>=256 bf16.
     Halving the KV tile ~halves the dominant LDS consumer -> fits with margin.
     (Only bf16; fp8 element_size==1 needs TILE_SIZE>=32, so it is excluded.)
     This change ALONE clears the limit.

  2. BLOCK_M: cap at max(8, num_queries_per_kv) for head_dim>=256. Belt-and-
     suspenders -- shaves the small BLOCK_M-scaled buffer too. (On its own this
     is NOT enough: the BLOCK_M-independent K+V part is already ~64 KB, which is
     why change #1 is the real fix.)

Idempotent and self-verifying: run twice safely; asserts each hunk matches
exactly once before writing. Smaller tiles/blocks are always numerically
correct -- only finer-grained, marginally slower.

Usage:  python3 apply-gfx1100-lds-fix.py [path-to-triton_unified_attention.py]
Default path is the rootfs vllm-venv location.
"""
import sys

F = (sys.argv[1] if len(sys.argv) > 1 else
     "/home/ubuntu/vllm-venv/lib/python3.13/site-packages/"
     "vllm/v1/attention/ops/triton_unified_attention.py")

s = open(F).read()
orig = s

# ---- Hunk 1: TILE_SIZE 32 -> 16 for head_dim>=256 bf16 (the real fix) ----
h1_old = ('    """Select tile size with Gemma3-specific optimization."""\n'
          '    if _is_gemma3_attention(head_size, sliding_window):')
h1_new = ('    """Select tile size with Gemma3-specific optimization."""\n'
          '    # gfx1100 (RDNA3) 64KB LDS: head_dim>=256 (gemma) overflows the unified-attn\n'
          '    # kernel -- K+V tiles (TILE_SIZE x HEAD_SIZE_PADDED) alone ~64KB at TILE_SIZE=32,\n'
          '    # so no BLOCK_M fits. Halve KV tile to 16 (bf16 only; fp8 needs >=32). Correct (finer tiling).\n'
          '    if head_size >= 256 and element_size >= 2:\n'
          '        return 16\n'
          '    if _is_gemma3_attention(head_size, sliding_window):')
if 'if head_size >= 256 and element_size >= 2:' in s:
    print("hunk 1 (TILE_SIZE): already applied")
else:
    assert s.count(h1_old) == 1, "hunk 1 anchor matched %d times (expected 1)" % s.count(h1_old)
    s = s.replace(h1_old, h1_new, 1)
    print("hunk 1 (TILE_SIZE -> 16): applied")

# ---- Hunk 2: BLOCK_M cap for head_dim>=256 (belt-and-suspenders) ----
h2_old = ('        16 if num_queries_per_kv <= 16 else triton.next_power_of_2(num_queries_per_kv)\n'
          '    )\n'
          '    BLOCK_Q = BLOCK_M // num_queries_per_kv')
h2_new = ('        16 if num_queries_per_kv <= 16 else triton.next_power_of_2(num_queries_per_kv)\n'
          '    )\n'
          '    if head_size >= 256:  # gfx1100 64KB LDS: head_dim>=256 (gemma) overflows TRITON_ATTN at BLOCK_M=16\n'
          '        BLOCK_M = max(8, num_queries_per_kv)\n'
          '    BLOCK_Q = BLOCK_M // num_queries_per_kv')
if 'if head_size >= 256:  # gfx1100 64KB LDS' in s:
    print("hunk 2 (BLOCK_M): already applied")
else:
    assert s.count(h2_old) == 1, "hunk 2 anchor matched %d times (expected 1)" % s.count(h2_old)
    s = s.replace(h2_old, h2_new, 1)
    print("hunk 2 (BLOCK_M cap): applied")

if s != orig:
    open(F, "w").write(s)
    print("wrote", F)
else:
    print("no change (already fully patched)")
