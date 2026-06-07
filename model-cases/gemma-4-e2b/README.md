# gemma-4-E2B-it on vllm-venv (single-VM, TP=1, one GPU)

Google **Gemma-4 E2B** (served fp16), single-GPU — a small non-Qwen model, useful
as a fast sanity case for the rootfs. **Serves ✅** after a one-line RDNA3 LDS fix
to vLLM's Triton attention (see below) — on the **same `vllm-venv` (0.21)** as every
other case, needing nothing from the old `ai-2.10` venv.

- **Launcher:** [`../../servers/vllm/gemma4-e2b-card1-dual.sh`](../../servers/vllm/gemma4-e2b-card1-dual.sh)
  — TP=1, `ROCR_VISIBLE_DEVICES=1` (pins to one physical GPU; `HIP_VISIBLE_DEVICES`
  alone makes torch see zero devices here), `--model /data/gemma-4-E2B-it`,
  max-model-len 4096, served as `gemma4-e2b` on port 8002.
- **Launcher fixes (2026-06-07):** (1) exec venv `/home/ubuntu/ai-2.10` (missing) →
  `/home/ubuntu/vllm-venv`; (2) it sourced the missing `vllm-serve/server-env.sh` —
  re-sourced the present `/home/ubuntu/vllm-serve-env.sh` instead (same pattern as the
  4B fix). No other change.
- **The real blocker — gfx1100 64 KB LDS:** gemma's attention uses **head_dim 256**
  (Qwen uses 128). vLLM's V1 ROCm TRITON_ATTN kernel staged its K+V tiles in **66,560 B
  of LDS > the 65,536 B (64 KB) gfx1100 limit** → `OutOfResources` → first inference 500.
  (V1 ROCm has no working TORCH_SDPA fallback; it ignores `VLLM_ATTENTION_BACKEND`, so
  the kernel itself had to change.)
- **Fix:** [`apply-gfx1100-lds-fix.py`](apply-gfx1100-lds-fix.py) — **one** scoped, idempotent
  edit to `triton_unified_attention.py`, gated on `head_size >= 256` (Qwen 128-dim untouched)
  and bf16 (fp8 needs TILE≥32): **`_get_tile_size` returns TILE_SIZE 16 instead of 32**.
  This ~halves the K+V tiles (which alone hit ~64 KB at TILE_SIZE=32) → clears the limit with
  ~16 KB margin. *(A BLOCK_M cap was tried first but only shaved 512 B — proving TILE_SIZE is
  the only real lever; it was reverted, see RESULTS.)*
- **Status:** ✅ **verified serving** on gfx1100 at upstream BLOCK_M=16 + TILE_SIZE=16. Output
  correct ("capital of France" → "Paris"); **steady-state decode 21.7 tok/s** (512 tokens, flat
  across the run, TTFT 1.2 s warm) — in line with the historical ~20 tok/s for Gemma-4-E4B.
  See [RESULTS.md](RESULTS.md).
