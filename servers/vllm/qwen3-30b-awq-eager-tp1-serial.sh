#!/bin/bash
# vLLM serial mode (max-num-seqs=1) — workaround for the fused-MoE batched
# bug. Engine processes one request at a time, so concurrent clients are
# queued. Outputs are correct, throughput is single-request bound.
set -e
source /home/ubuntu/vllm-serve/server-env.sh
exec /home/ubuntu/ai-2.10/bin/python3 -m vllm.entrypoints.openai.api_server \
    --model /data/Qwen3-30B-A3B-Instruct-2507-AWQ \
    --served-model-name qwen3-30b-a3b \
    --dtype float16 \
    --max-model-len 4096 \
    --max-num-seqs 1 \
    --gpu-memory-utilization 0.85 \
    --tensor-parallel-size 1 \
    --no-enable-prefix-caching \
    --no-enable-chunked-prefill \
    --enforce-eager \
    --host 0.0.0.0 --port 8000
