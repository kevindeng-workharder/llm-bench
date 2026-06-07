# Qwen3-4B (dense fp16) on vllm-venv (single-VM, TP=1, one GPU)

Qwen3-4B dense fp16 — the canonical small control config (fits one GPU).

- **Launcher:** [`../../servers/vllm/qwen3-4b-fp16-graph-tp1.sh`](../../servers/vllm/qwen3-4b-fp16-graph-tp1.sh)
  — TP=1, `ROCR_VISIBLE_DEVICES=0`, `--model /data/Qwen3-4B`, dtype float16,
  max-model-len 4096, graph (cudagraph), served as `qwen3-4b` on port 8000.
- **Rewritten self-contained (2026-06-07):** the original was a *thin wrapper* — it
  `exec env MODEL=… VENV_PREFIX=… bash /home/ubuntu/vllm-serve/launch-server.sh`. That
  `launch-server.sh` generic launcher is **absent from both this rootfs and the repo**
  (never archived), and `VENV_PREFIX=/home/ubuntu/ai-2.10` was missing too. So the
  launch is now **inlined** (source `vllm-serve-env.sh` + direct
  `vllm-venv/bin/python -m vllm …`), same approach as gemma. The other 4B variants
  (`-019-*`, `-vllm011`, `-clamp`, `eager-tp1`, `graph-tp2`) still reference the missing
  wrapper / are intentionally pinned to old vLLM versions — not retrofitted here.
- **Status:** ✅ verified on vllm-venv — see [RESULTS.md](RESULTS.md).
