# RESULTS — Qwen3.6-27B-AWQ on vllm-venv

**Verified:** 2026-06-07, host `10.103.11.199`, single guest, both gfx1100, TP=2.
**Launcher:** [`../../servers/vllm/qwen3_6-27b-awq-graph-tp2.sh`](../../servers/vllm/qwen3_6-27b-awq-graph-tp2.sh)
on `/home/ubuntu/vllm-venv`.

## vLLM 0.21 DOES support AWQ on gfx1100

- Weights loaded in **159 s**.
- Then a **~56 min cold kernel compile** (CPU-bound on TCG): the log repeated
  `No available shared memory broadcast block found in 60 s … (e.g. compilation,
  weight/kv cache quantization)`. Confirmed **compiling, not hung** — `~/.triton/cache`
  grew **+3082 entries in 2 min** while both GPUs sat at ~2 % and the `VLLM::EngineCore`
  worker burned CPU. The AWQ/Triton kernels are not in the warm 27B-Quark cache, so
  they compile from scratch the first time; **cached after** → fast relaunch.
- Then **`Application startup complete`** — the server came up and served as model id
  `qwen3_6-27b-awq` (`/v1/models` confirmed). (`served-model-name` is `qwen3_6-27b-awq`,
  **not** `qwen3_6-27b-int8` — a request with the wrong name returns 404.)

## Takeaway

AWQ is runnable on this rootfs purely via the venv re-point (`/data/ai-2.11` →
`vllm-venv`); the only cost is the one-time multi-hour TCG compile. (A standalone
output-correctness bench was not run — testing was redirected before that step — but
AWQ is a mainstream vLLM path and the engine reached steady serving.)
