# RESULTS — gemma-4-E2B-it on vllm-venv

**Tested:** 2026-06-07, host `10.103.11.199`, single guest, TP=1 (one gfx1100, GPU1).
**Launcher:** [`../../servers/vllm/gemma4-e2b-card1-dual.sh`](../../servers/vllm/gemma4-e2b-card1-dual.sh)
on `/home/ubuntu/vllm-venv` (source-fixed).

## Loads — but crashes at inference on gfx1100 (LDS limit)

The launcher fix is correct: gemma **loads, compiles, and reaches `Application startup
complete`** on vllm-venv (no missing-file error). But the **first inference request 500s** —
the EngineCore dies with:

```
triton.runtime.errors.OutOfResources: out of resource: shared memory,
Required: 66560, Hardware limit: 65536. Reducing block sizes or `num_stages` may help.
→ vllm.v1.engine.exceptions.EngineDeadError → POST /v1/chat/completions 500
```

A gemma Triton kernel is configured for **66,560 bytes (65 KB) of LDS / shared memory**, but
**gfx1100 (RDNA3) has only 64 KB LDS per CU** (65,536 bytes). The kernel cannot launch.

## This is a hardware-kernel mismatch, not a launcher/env issue

- The venv + source fixes are validated — gemma got all the way to serving.
- It is **gemma-specific**: 27B-Quark (int8), Qwen3.6-27B-AWQ, and Qwen3-4B all serve fine on
  the same gfx1100 (their kernels fit 64 KB LDS). Only gemma's exceeds it.
- To run gemma here you'd have to **re-tune its kernels for RDNA3's smaller LDS** (smaller
  block sizes / fewer `num_stages`, per Triton's own hint) — a real porting task, not a config
  flip. Not done here.

## Path-to-here (env issues, all fixed, before the real blocker showed)

The first three attempts died on environment problems that masked this: a leaked `VLLM::`
worker (22 GiB/GPU), parallel-compile CPU contention (startup handshake timeout), and a
page-cache kswapd thrash (fixed by a VM reboot). On the clean reboot gemma finally compiled
through — and surfaced the genuine LDS limit above.

**Verdict:** ❌ does not serve on gfx1100 as-is (LDS-OOM); launcher is correct.
