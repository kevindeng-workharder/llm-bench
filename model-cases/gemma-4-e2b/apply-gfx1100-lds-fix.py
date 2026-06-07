#!/usr/bin/env python3
"""Apply the gfx1100 (RDNA3) 64 KB-LDS fix to vLLM's Triton unified-attention.

Problem
-------
Gemma's attention uses head_dim = 256 (Qwen etc. use 128). vLLM's V1 ROCm
TRITON_ATTN kernel (`triton_unified_attention.py`) stages its K and V tiles
(`TILE_SIZE x HEAD_SIZE_PADDED`) in LDS. With head_dim 256 the K+V tiles alone
need ~64 KB at TILE_SIZE=32, so the kernel asks for 66,560 B of shared memory
while gfx1100 (RDNA3) has only 65,536 B (64 KB) per CU ->
`triton.runtime.errors.OutOfResources` -> EngineDeadError -> first inference
500s. (V1 ROCm has no working TORCH_SDPA fallback; it ignores
VLLM_ATTENTION_BACKEND and always uses TRITON_ATTN, so the kernel itself must
change.)

The fix: TILE_SIZE 32 -> 16 for head_dim>=256 bf16
--------------------------------------------------
One scoped, correctness-safe change in `_get_tile_size`, gated on
head_size >= 256 so nothing else (Qwen 128-dim) is affected, and on bf16
(element_size>=2) since fp8 (element_size==1) needs TILE_SIZE>=32. Halving the
KV tile ~halves the dominant LDS consumer -> requirement clears 64 KB with
~16 KB margin. Smaller tiles are always numerically correct -- only finer.

Why TILE_SIZE and not BLOCK_M (the diagnostic that pinned it)
------------------------------------------------------------
A first attempt capped BLOCK_M 16 -> 8. It dropped the requirement only
66,560 -> 66,048 B (512 B). That tiny delta proved the BLOCK_M-INDEPENDENT
LDS -- the K+V tiles -- is already ~65,536 B by itself, so NO BLOCK_M value can
ever fit. TILE_SIZE is the only lever. (The BLOCK_M cap was therefore reverted;
upstream BLOCK_M=16 is kept for slightly better prefill throughput.)

Idempotent and self-verifying: run twice safely; asserts the hunk matches
exactly once before writing.

Usage:  python3 apply-gfx1100-lds-fix.py [path-to-triton_unified_attention.py]
Default path is the rootfs vllm-venv location.
"""
import sys

F = (sys.argv[1] if len(sys.argv) > 1 else
     "/home/ubuntu/vllm-venv/lib/python3.13/site-packages/"
     "vllm/v1/attention/ops/triton_unified_attention.py")

s = open(F).read()

# ---- TILE_SIZE 32 -> 16 for head_dim>=256 bf16 (the fix) ----
old = ('    """Select tile size with Gemma3-specific optimization."""\n'
       '    if _is_gemma3_attention(head_size, sliding_window):')
new = ('    """Select tile size with Gemma3-specific optimization."""\n'
       '    # gfx1100 (RDNA3) 64KB LDS: head_dim>=256 (gemma) overflows the unified-attn\n'
       '    # kernel -- K+V tiles (TILE_SIZE x HEAD_SIZE_PADDED) alone ~64KB at TILE_SIZE=32,\n'
       '    # so no BLOCK_M fits. Halve KV tile to 16 (bf16 only; fp8 needs >=32). Correct (finer tiling).\n'
       '    if head_size >= 256 and element_size >= 2:\n'
       '        return 16\n'
       '    if _is_gemma3_attention(head_size, sliding_window):')

if 'if head_size >= 256 and element_size >= 2:' in s:
    print("TILE_SIZE fix: already applied")
else:
    assert s.count(old) == 1, "anchor matched %d times (expected 1)" % s.count(old)
    open(F, "w").write(s.replace(old, new, 1))
    print("TILE_SIZE fix (32 -> 16 for head_dim>=256 bf16): applied ->", F)
