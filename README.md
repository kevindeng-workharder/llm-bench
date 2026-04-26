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

## Current status (2026-04-26)

- vLLM **graph** mode is dramatically faster than eager (~21× single-request
  decode on Qwen3-30B AWQ).
- vLLM **batched fused-MoE** is broken on this stack: at any N≥2, only one
  request in the batch produces correct output; others stream `!` forever.
  See `docs/vllm-moe-batched-bug.md`.
- llama.cpp is correct at all N, peak aggregate ~16 tok/s at N=4.

| Engine + config | N=1 | N=2 | N=4 | N=8 | garbage |
|---|---|---|---|---|---|
| llama.cpp Qwen3.6-35B MXFP4 | 12.0 | – | 16.0 | 9.9 | 0/N |
| vLLM Qwen3-30B AWQ eager | 1.07 | 1.42 | – | – | 1/2, etc. |
| vLLM Qwen3-30B AWQ graph | 23.2 | 33.0 | 64.6 | 9.5 | 2/4, 6/8 |
| vLLM Qwen3-30B AWQ graph + max-num-seqs=1 | – | – | 0.9 | – | **0/4** |

(numbers are aggregate tok/s)
