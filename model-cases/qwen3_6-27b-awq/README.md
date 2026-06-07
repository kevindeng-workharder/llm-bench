# Qwen3.6-27B-AWQ on vllm-venv (single-VM, TP=2)

Qwen3.6-27B in **AWQ** (4-bit) — the AWQ sibling of the flagship Quark-INT8.

- **Launcher:** [`../../servers/vllm/qwen3_6-27b-awq-graph-tp2.sh`](../../servers/vllm/qwen3_6-27b-awq-graph-tp2.sh)
  — TP=2, `--quantization awq`, `--model /data/Qwen3.6-27B-AWQ`, max-model-len 2048,
  graph (cudagraph), served as `qwen3_6-27b-awq` on port 8000.
- **Venv:** standardized 2026-06-07 from the missing `/data/ai-2.11` (vLLM 0.19) →
  `/home/ubuntu/vllm-venv` (vLLM 0.21). Self-contained (sources the present
  `vllm-serve-env.sh`), so it needed no other change.
- **Status:** ✅ verified — see [RESULTS.md](RESULTS.md).
