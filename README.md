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

## Current status (2026-04-28)

**SOLVED — vLLM 0.19 fp16 multi-concurrent NaN bug fixed by the v2 clamp
patch.** All 11 server configs in the matrix now produce 0/N garbage at
every N from 1 to 8 concurrent.

Mechanism: in mixed prefill+decode batches, certain prefill query rows
hit a degenerate softmax denominator (`l_i` ≪ 1e-3 in fp32). The fp32
accumulation drift then makes `acc / (l_i + 1e-10)` violate the
mathematical "convex combination of v" bound — overshoots fp16 max
65 504, casts to fp16 inf, propagates to NaN through `o_proj`. Affected
streams output token id 0 (`!`) garbage.

Fix is a NaN-safe `tl.where` clamp + zero-degenerate-rows guard at every
triton attention epilogue (7 sites across 4 files). See
[`docs/debug-summary.md`](docs/debug-summary.md) for the full bisection.

### vLLM patches now live in a separate repo

The riscv64 + ROCm gfx1100 cross-compile patches AND the v2 clamp fix are
maintained as proper git commits on a fork:

🔗 **https://github.com/kevindeng-workharder/vllm-riscv**
(branch `riscv-rocm-gfx1100`, based on upstream `v0.19.0`)

To install on a fresh machine, replacing the patch-script chain:

```bash
git clone https://github.com/kevindeng-workharder/vllm-riscv.git
cd vllm-riscv
git checkout riscv-rocm-gfx1100
pip install . --no-deps    # 6 commits land cleanly into the venv
```

The old `scripts/instruments/patch-*.py` scripts and `revert-all-debug-patches.py`
in this repo are kept for historical reproducibility but are no longer the
recommended path.

### Verified bench matrix (post-fix)

| Engine + config | N=1 | N=2 | N=4 | N=8 agg | bad@N=8 |
|---|---|---|---|---|---|
| llama.cpp Qwen3.6-35B-A3B-MXFP4 | 7.0 | 10.8 | 7.0 | 8.6 | 0/8 ✅ |
| vLLM 0.19 Qwen3-0.6B fp16 graph TP1 | 36.1 | 44.1 | 107 | **188** | 0/8 ✅ |
| vLLM 0.19 Qwen3-4B fp16 graph TP1 | 35.8 | 58.3 | 112 | **211** | 0/8 ✅ |
| vLLM 0.19 Qwen3-4B fp16 eager TP1 | 3.0 | 4.8 | 13.6 | 25.1 | 0/8 ✅ |
| vLLM 0.19 Qwen3-4B fp16 graph TP2 | 13.4 | 21.9 | 42.7 | 78.3 | 0/8 ✅ |
| vLLM 0.19 Qwen3-4B bf16 graph TP1 | 36.9 | 57.4 | 114 | 212 | 0/8 ✅ |
| vLLM 0.11 Qwen3-4B fp16 graph TP1 (control) | 19.5 | 28.0 | 50.6 | 77.5 | 0/8 ✅ |
| vLLM 0.19 Qwen2.5-14B fp16 graph TP2 | 8.7 | 14.5 | 31.5 | 31.2 | 0/8 ✅ |
| vLLM 0.19 Qwen3-30B-A3B AWQ graph TP1 | 19.5 | 28.6 | 66.9 | **101** | 0/8 ✅ |
| vLLM 0.19 Qwen3-30B-A3B AWQ eager TP1 | 4.9 | 7.6 | 8.8 | 9.5 | 0/8 ✅ |
| vLLM 0.19 Qwen3-30B-A3B AWQ graph TP2 | 4.4 | 6.5 | 8.7 | 9.7 | 0/8 ✅ |

(aggregate tok/s; all 0/N garbage at every column.)

The earlier "163 tok/s @ N=100" type numbers from the pre-fix era were
inflated by garbage tokens being counted in the throughput total — old
bench scripts measured tok/s only and never inspected output content.
