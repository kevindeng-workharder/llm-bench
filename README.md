# llm-bench

Side-by-side concurrent throughput benchmark for **llama.cpp** and **vLLM**
running on the **riscv64 + ROCm + AMD 7900 XTX** stack (single or dual GPU,
via QEMU + VFIO passthrough).

Each combination of (engine, model, quant, compilation mode, GPU count, request
concurrency) is captured as a JSON record and rolled up into a markdown table.

## What this measures

For each `(server config × N concurrent clients)`:

- **TTFT** — time to first streamed token
- **per-request tok/s** — single-client steady-state decode rate
- **aggregate tok/s** — sum across clients (the multi-tenant throughput)
- **garbage rate** — fraction of requests whose output is degenerate (e.g.
  `!!!!!` repeating, common failure mode under broken kernels)

A run produces `results/raw/<config-name>.<timestamp>.json` and a per-day
markdown digest in `results/<YYYY-MM-DD>.md`.

## Layout

```
configs/            YAML test plans (server × workload matrix)
servers/
  vllm/             vLLM launcher scripts, one per (model × mode × tp)
  llamacpp/         llama-server launcher scripts
runner/             Python harness — server lifecycle + concurrent client
workloads/          Prompt banks
scripts/            Glue for SSH-into-VM, port forwarding, VRAM cleanup
results/            JSON + markdown
docs/               Bug write-ups, environment notes
```

## Quick start

```bash
# From host (assumes VM is up; SSH `localhost:2222` reaches it):
./scripts/run-one.sh vllm-qwen3-30b-awq-graph-tp1 8   # config × N=8

# Or run the whole matrix:
./scripts/run-matrix.sh configs/bench-matrix.yaml

# Aggregate the day's runs into a markdown report:
python3 -m runner.report results/raw/ > results/$(date +%F).md
```

## Current status (2026-04-27)

**ROOT CAUSE FOUND — fp16 NaN overflow inside vLLM 0.19's model forward
on this stack.** Workaround: use `--dtype bfloat16`.

```
Qwen3-4B graph TP1 N=8:
  --dtype float16  → 209 t/s aggregate, but 6/8 streams output `!!!!!`  ❌
  --dtype bfloat16 → 212 t/s aggregate,  0/8 streams output garbage     ✅
```

Same model, same flags, same hardware, same vLLM 0.19 build. Bisected via
in-place instrumentation of `gpu_model_runner.py`
([`scripts/instruments/`](scripts/instruments/)) — pinpointed the NaN to
`hidden_states[1, :]` and `hidden_states[3, :]` in the model output of
the batched forward, before `compute_logits`. fp16 has 5-bit exponent
(max ≈ 65 504); bfloat16 has the same 8-bit exponent as fp32 and doesn't
overflow. vLLM 0.11 on the same stack with `--dtype float16` is correct,
so vLLM 0.19 introduced (or wired in) a kernel that doesn't handle the
fp16 numerical edge case.

Full debugging chain in [`docs/vllm-019-batched-bug.md`](docs/vllm-019-batched-bug.md).

- vLLM 0.19 graph mode is fast on **single-request** decode (~21× eager,
  ~1.87× the 0.11 baseline). At N≥2 / N≥4 some clients stream garbage
  (token `!` repeating).
- llama.cpp is correct at all N, peak aggregate ~16 tok/s at N=4.
- The previous "163 tok/s @ N=100" number from the 0.19 era was inflated
  by garbage tokens being counted in the throughput total — earlier bench
  scripts measured tok/s only and never inspected output content.

| Engine + config | N=1 | N=2 | N=4 | N=8 | garbage at N=8 |
|---|---|---|---|---|---|
| llama.cpp Qwen3.6-35B MXFP4 | 12.0 | – | 16.0 | 9.9 | 0/8 ✅ |
| **vLLM 0.11** Qwen3-4B fp16 graph TP1 (control) | 19.6 | 30.0 | 51.2 | **71.4** | **0/8 ✅** |
| vLLM 0.19 Qwen3-0.6B fp16 graph TP1 | 36.1 | 58.9 | 117 | 216 | 6/8 ❌ |
| vLLM 0.19 Qwen3-4B fp16 eager TP1 | 3.3 | 7.1 | 14.1 | 25.4 | 6/8 ❌ |
| vLLM 0.19 Qwen3-4B fp16 graph TP1 | 36.6 | 41.2 | 115 | 213 | 5/8 ❌ |
| vLLM 0.19 Qwen3-4B fp16 graph TP2 | 13.7 | 20.2 | 37.5 | 88.8 | 5/8 ❌ |
| vLLM 0.19 Qwen2.5-14B fp16 graph TP2 | 7.6 | 14.1 | 28.0 | 37.7 | 7/8 ❌ |
| vLLM 0.19 Qwen3-30B AWQ graph TP1 | 23.2 | 33.0 | 64.6 | 9.5 | 6/8 ❌ |
| vLLM 0.19 Qwen3-30B AWQ + max-num-seqs=1 (workaround) | – | – | 0.9 | – | 0/4 ✅ |

(aggregate tok/s; ❌ rows have inflated numbers — see garbage column.)
