# RESULTS — gemma-4-E2B-it on vllm-venv ✅

**Verified:** 2026-06-07, host `10.103.11.199`, single guest, TP=1 (one gfx1100, GPU1).
**Launcher:** [`../../servers/vllm/gemma4-e2b-card1-dual.sh`](../../servers/vllm/gemma4-e2b-card1-dual.sh)
on `/home/ubuntu/vllm-venv` (vLLM 0.21), with the RDNA3 LDS fix
([`apply-gfx1100-lds-fix.py`](apply-gfx1100-lds-fix.py)) applied to its Triton attention.
Config: **upstream BLOCK_M=16 + patched TILE_SIZE=16**.

## Serves — correct output, ~21.7 tok/s steady-state

- vLLM V1 engine, `model=/data/gemma-4-E2B-it`, `dtype=torch.float16`, max-model-len
  4096, cudagraph `FULL_DECODE_ONLY`, served as `gemma4-e2b` on port 8002.
- **Output correct:** "What is the capital of France?" → **"The capital of France is Paris."**
- **Long decode (512 tokens, `ignore_eos`):**

  | metric | value |
  |---|---|
  | decode (overall) | **21.66 tok/s** |
  | decode 1st half (tok 1–256) | 21.51 tok/s |
  | decode 2nd half (tok 256–512) | 21.81 tok/s |
  | TTFT (warm) | 1.17 s |
  | end-to-end | 24.8 s / 512 tok |

  Decode is **flat across the run** — no degradation as the KV context grows to 512.
- First request after a cold launch is ~50–63 s (JIT-compiles the attention kernel); cached after.

### vs the historical number
The repo's only prior gemma figure is `scripts/chat/` → **~20 tok/s on Gemma-4-E4B** (in-process
`LLMEngine` + cudagraph **replay**, ai-2.10 = vLLM 0.19, "TP=2 setup"). This run's **21.7 tok/s** is
in the same band — and it's the *smaller* E2B at **TP=1** over the **OpenAI HTTP server** with the
**halved** TILE_SIZE. So **the LDS fix does not meaningfully cost decode throughput.** (Not a
clean apples-to-apples: E2B<E4B, but TP=1<TP=2 and HTTP<graph-replay roughly cancel out.)

> An earlier short test reported 12.4 tok/s — that was an **80-token** measurement at BLOCK_M=8
> with the cold cudagraph/TCG warmup folded into the rate. Over 512 steady-state tokens the real
> figure is **21.7 tok/s**; superseded.

## The fix — and how the root cause was pinned

gemma loaded and reached `Application startup complete`, but the **first inference 500'd**:

```
triton.runtime.errors.OutOfResources: shared memory, Required: 66560, Hardware limit: 65536
→ vllm.v1.engine.exceptions.EngineDeadError → POST /v1/chat/completions 500
```

head_dim 256 (vs Qwen 128) makes vLLM's V1 ROCm TRITON_ATTN kernel ask for **66,560 B of LDS**,
over gfx1100 (RDNA3)'s **65,536 B (64 KB) per CU**. (V1 ROCm has no working TORCH_SDPA — it
ignores `VLLM_ATTENTION_BACKEND` and always uses TRITON_ATTN, so the kernel itself had to change.)

**Diagnostic chain (the data point that mattered):** a first patch capping `BLOCK_M` 16→8 dropped
the requirement only **66560 → 66048 (512 B)**. That tiny delta proved the **BLOCK_M-independent
LDS is ~65,536 B — already at the limit by itself.** No BLOCK_M value could ever fit. That
independent part is the **K+V tiles** (`TILE_SIZE × head_dim`, head_dim fixed at 256), so the only
real lever is **TILE_SIZE**. Halving it 32→16 (in `_get_tile_size`) ~halves the K+V tiles → clears
64 KB with ~16 KB margin.

**The final fix is TILE_SIZE-only.** The BLOCK_M cap was a diagnostic dead-end (couldn't fit alone)
and was **reverted** — `BLOCK_M=16 + TILE_SIZE=16` was re-verified serving (the 21.7 tok/s run
above), so upstream BLOCK_M=16 is kept for slightly better prefill throughput. The one change is
gated on `head_size >= 256` (Qwen 128-dim untouched), bf16-only (fp8 needs TILE_SIZE≥32). Smaller
tiles are always numerically correct — just finer.

## No `ai-2.10` needed

gemma "ran before" on `ai-2.10` = **vLLM 0.19** (whose older attention predates this LDS-heavy
unified-attention) — a docker **payload tarball** (`docker/Dockerfile` →
`payload/home-ubuntu-ai-2.10.tar.gz`), **not on any rootfs backup** and now lost. This fix makes
gemma serve on the **current `vllm-venv` (0.21)** — the same venv as 27B-Quark / AWQ / 4B — so the
lost 0.19 venv is irrelevant.

**Verdict:** ✅ serves on gfx1100 with the one-line LDS fix; output correct; ~21.7 tok/s steady.
