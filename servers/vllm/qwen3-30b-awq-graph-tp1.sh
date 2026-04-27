#!/bin/bash
# vLLM graph mode, single 7900 XTX, Qwen3-30B-A3B-Instruct-2507-AWQ.
# COMPILATION: graph (CUDAGraphMode.FULL_DECODE_ONLY) — ~21x faster than eager.
#
# OLD note (now obsolete):
#   "KNOWN ISSUE: fused-MoE batched kernel produces garbage logits at N>=2."
# That misdiagnosed the bug. The actual cause was a degenerate-row
# overflow in the triton prefix_prefill attention kernel (NOT in the MoE
# kernel). Fixed by scripts/instruments/patch-clamp-v2-zero-degenerate.py.
# Verified 2026-04-27: N=1/2/4/8 all 0/N garbage, ~19/29/67/101 t/s.
# See docs/debug-summary.md.
set -e
source /home/ubuntu/vllm-serve/server-env.sh
exec /home/ubuntu/ai-2.10/bin/python3 -m vllm.entrypoints.openai.api_server \
    --model /data/Qwen3-30B-A3B-Instruct-2507-AWQ \
    --served-model-name qwen3-30b-a3b \
    --dtype float16 \
    --max-model-len 4096 \
    --max-num-seqs 8 \
    --gpu-memory-utilization 0.85 \
    --tensor-parallel-size 1 \
    --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --host 0.0.0.0 --port 8000
