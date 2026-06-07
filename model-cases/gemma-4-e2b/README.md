# gemma-4-E2B-it on vllm-venv (single-VM, TP=1, one GPU)

Google **Gemma-4 E2B** (bf16), single-GPU — a small non-Qwen model, useful as a
fast sanity case for the rootfs.

- **Launcher:** [`../../servers/vllm/gemma4-e2b-card1-dual.sh`](../../servers/vllm/gemma4-e2b-card1-dual.sh)
  — TP=1, `ROCR_VISIBLE_DEVICES=1` (pins to one physical GPU; `HIP_VISIBLE_DEVICES`
  alone makes torch see zero devices here), `--model /data/gemma-4-E2B-it`,
  max-model-len 4096, served as `gemma4-e2b` on port 8002.
- **Two fixes (2026-06-07):** (1) exec venv `/home/ubuntu/ai-2.10` (missing) →
  `/home/ubuntu/vllm-venv`; (2) it sourced the missing `vllm-serve/server-env.sh` —
  re-sourced the present `/home/ubuntu/vllm-serve-env.sh` instead (same pattern as the
  4B fix). No other change.
- **Status:** ❌ **loads but does not serve on gfx1100.** The launcher fix is correct — gemma
  compiles and reaches `Application startup complete` — but the **first inference 500s**: a gemma
  Triton kernel needs **66,560 B of LDS, over the 64 KB (65,536 B) gfx1100 limit**
  (`triton … OutOfResources: shared memory` → `EngineDeadError`). A hardware-kernel mismatch,
  **gemma-specific** (awq / 4B / 27B-Quark all serve on the same GPU); needs RDNA3 kernel
  re-tuning (smaller blocks). See [RESULTS.md](RESULTS.md).
