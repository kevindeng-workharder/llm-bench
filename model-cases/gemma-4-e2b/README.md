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
- **Status:** ⏳ **launcher fixed; serve-confirm pending.** The source-fix is validated —
  gemma loads on vllm-venv with no missing-file error. A full serve-confirm is still
  pending: gemma's cold compile is extremely slow on this QEMU/TCG host (the weight read
  alone took **700 s**), and three earlier attempts died on *environment* issues — a leaked
  awq worker (22 GiB/GPU), parallel-compile CPU contention (startup-handshake timeout), and
  a page-cache-pressure kswapd thrash (fixed by a VM reboot). gemma is a mainstream vLLM 0.21
  model, so it is expected to serve; RESULTS will be added once the compile lands.
