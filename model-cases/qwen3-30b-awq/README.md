# Qwen3-30B-A3B-Instruct-2507-AWQ — BLOCKED (model not on this image)

Qwen3-30B-A3B (MoE) in AWQ. **Cannot run here — the model weights are not on this
image.**

- **Launchers (unchanged):** [`qwen3-30b-awq-graph-tp2.sh`](../../servers/vllm/qwen3-30b-awq-graph-tp2.sh),
  `…-graph-tp1.sh`, `…-eager-tp1.sh`, `…-eager-tp1-serial.sh` — still reference the
  missing `/home/ubuntu/ai-2.10` venv and `--model /data/Qwen3-30B-A3B-Instruct-2507-AWQ`.
- **Why blocked:** searched the whole guest (`/data`, `/opt`, `/home`, HF cache) on
  2026-06-07 — `Qwen3-30B-A3B-Instruct-2507-AWQ` is **absent** (the only "A3B" hits are
  unrelated `~/.triton/cache` hash dirs). The model image (`models.img`) only carries
  `Qwen3.6-27B-Quark/AWQ/FP8`, `Qwen3-4B`, and `gemma-4-E2B-it`.
- **To unblock:** copy the 30B-AWQ weights to `/data`, then retrofit the launchers like
  the others (venv `ai-2.10` → `vllm-venv`; inline if they use the missing
  `vllm-serve/` wrapper). Until then it is documented here only.
- **Status:** ❌ blocked — needs the model.
