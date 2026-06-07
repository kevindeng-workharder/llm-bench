# RESULTS — gemma-4-E2B-it on vllm-venv ✅

**Verified:** 2026-06-07, host `10.103.11.199`, single guest, TP=1 (one gfx1100, GPU1).
**Launcher:** [`../../servers/vllm/gemma4-e2b-card1-dual.sh`](../../servers/vllm/gemma4-e2b-card1-dual.sh)
on `/home/ubuntu/vllm-venv` (vLLM 0.21), with the RDNA3 LDS fix
([`apply-gfx1100-lds-fix.py`](apply-gfx1100-lds-fix.py)) applied to its Triton attention.

## Serves — correct output

- vLLM V1 engine, `model=/data/gemma-4-E2B-it`, `dtype=torch.float16`, max-model-len
  4096, cudagraph `FULL_DECODE_ONLY`, served as `gemma4-e2b` on port 8002.
- **First request: 48.4 s** — the prefill attention kernel (now TILE_SIZE=16) JIT-compiles
  on this slow host; cached after.
- **Output correct:** "What is the capital of France?" → **"The capital of France is Paris."**
- **Decode: 12.36 tok/s** (51 tok, TTFT 1.7 s on the warm 2nd request). Slower than the
  dense Qwen3-4B's 36 tok/s on the same GPU — expected: gemma's **head_dim 256 is 2×**
  Qwen's 128 (heavier attention), and the fix **halves the KV tile** (TILE_SIZE 16 vs 32),
  trading attention throughput for fitting the 64 KB LDS. Correctness/serving was the goal.

## The fix — and how the root cause was pinned

gemma loaded and reached `Application startup complete`, but the **first inference 500'd**:

```
triton.runtime.errors.OutOfResources: shared memory, Required: 66560, Hardware limit: 65536
→ vllm.v1.engine.exceptions.EngineDeadError → POST /v1/chat/completions 500
```

head_dim 256 (vs Qwen 128) makes vLLM's V1 ROCm TRITON_ATTN kernel ask for **66,560 B of
LDS**, over gfx1100 (RDNA3)'s **65,536 B (64 KB) per CU**. (V1 ROCm has no working
TORCH_SDPA — it ignores `VLLM_ATTENTION_BACKEND` and always uses TRITON_ATTN, so the
backend can't be swapped; the kernel itself had to change.)

**Diagnostic chain (the data point that mattered):** a first patch capping `BLOCK_M`
16→8 dropped the requirement only **66560 → 66048 (512 B)**. That tiny delta proved the
**BLOCK_M-independent LDS is ~65,536 B — already at the limit by itself.** No BLOCK_M
value could ever fit. That independent part is the **K+V tiles** (`TILE_SIZE × head_dim`,
head_dim fixed at 256), so the only real lever is **TILE_SIZE**. Halving it 32→16 (via
`_get_tile_size`) ~halves the K+V tiles → requirement clears 64 KB with ~16 KB margin.

Two scoped edits, both gated on `head_size >= 256` (Qwen 128-dim untouched), bf16-only
(fp8 needs TILE_SIZE≥32). Smaller tiles/blocks are always numerically correct — just finer.

## No `ai-2.10` needed

gemma "ran before" inside a docker container on an `ai-2.10` = **vLLM 0.19** venv (whose
older attention kernel predates this LDS-heavy unified-attention). That venv was a docker
**payload tarball** (`docker/Dockerfile` → `payload/home-ubuntu-ai-2.10.tar.gz`), **not on
any rootfs backup** and now lost. This fix makes gemma serve on the **current `vllm-venv`
(0.21)** — the same venv as 27B-Quark / AWQ / 4B — so the lost 0.19 venv is irrelevant.

**Verdict:** ✅ serves on gfx1100 with the LDS fix; output correct; launcher correct.
