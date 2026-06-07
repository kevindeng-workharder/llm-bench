# Qwen3.6-27B-FP8 on vllm-venv — DEFERRED

Qwen3.6-27B in **FP8**. Model is present (`/data/Qwen3.6-27B-FP8`), launchers
re-pointed, but **not tested** — deferred on purpose.

- **Launchers:** [`qwen3_6-27b-fp8-graph-tp2.sh`](../../servers/vllm/qwen3_6-27b-fp8-graph-tp2.sh),
  [`…-graph-tp1.sh`](../../servers/vllm/qwen3_6-27b-fp8-graph-tp1.sh),
  [`…-eager-tp2.sh`](../../servers/vllm/qwen3_6-27b-fp8-eager-tp2.sh) — venv standardized
  2026-06-07 from the missing `/data/ai-2.11` (vLLM 0.19) → `/home/ubuntu/vllm-venv` (0.21).
- **Why deferred:** FP8 on **RDNA3 / gfx1100** is the one genuinely-uncertain quant —
  gfx1100 has no native FP8 (it's a CDNA/MI300 feature), and FP8 was *crashing* on the
  old 0.19 venv. On 0.21 it may work, may emulate slowly, or may be unsupported; and it
  would still pay the multi-hour cold compile. The re-point is in place so it *can* be
  tried, but it was skipped this round.
- **Status:** ⏸️ re-pointed, **not run**. Test before claiming it works.
