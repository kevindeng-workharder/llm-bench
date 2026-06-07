# RESULTS — Qwen3-4B (dense fp16) on vllm-venv

**Verified:** 2026-06-07, host `10.103.11.199`, single guest, TP=1 (one gfx1100, GPU0).
**Launcher:** [`../../servers/vllm/qwen3-4b-fp16-graph-tp1.sh`](../../servers/vllm/qwen3-4b-fp16-graph-tp1.sh)
— **self-contained rewrite** on `/home/ubuntu/vllm-venv` (replaces the missing
`launch-server.sh` wrapper).

## Works

- vLLM V1 engine init `model=/data/Qwen3-4B`, `dtype=torch.float16`, cudagraph
  `FULL_DECODE_ONLY [1,2,4,8]`, served as `qwen3-4b` on port 8000.
- Slow TCG CPU-startup (~4 min front-end → EngineCore — the recursive HF-config
  deepcopy), then GPU alloc + cudagraph capture → `Application startup complete`.
- **Output correct:** "capital of France" → the model reasons (`<think>… France is a
  country in Europe. The capital is …`) toward Paris — coherent, on-track.
- **Decode: 36.28 tok/s** (127 tok, TTFT 5.3 s) — fast, as expected for a 4B dense
  model on a single GPU (no quant, no all-reduce).

## Takeaway

A dense fp16 model runs cleanly on this rootfs via a **self-contained launcher**. The
fix that mattered was replacing the missing `vllm-serve/launch-server.sh` wrapper with an
inline `source vllm-serve-env.sh` + direct `vllm-venv/bin/python -m vllm …` — the model
and venv were both already present.
