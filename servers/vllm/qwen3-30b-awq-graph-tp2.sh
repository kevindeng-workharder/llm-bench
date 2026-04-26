#!/bin/bash
# vLLM graph mode, dual 7900 XTX (TP=2), Qwen3-30B-A3B-Instruct-2507-AWQ.
# Will not improve throughput unless the fused-MoE batched bug is fixed first
# (TP just shards the experts; per-expert kernel call is still buggy at
# batch>1). Kept for future re-test.
set -e
source /home/ubuntu/vllm-serve/server-env.sh
exec /home/ubuntu/ai-2.10/bin/python3 -m vllm.entrypoints.openai.api_server \
    --model /data/Qwen3-30B-A3B-Instruct-2507-AWQ \
    --served-model-name qwen3-30b-a3b \
    --dtype float16 \
    --max-model-len 4096 \
    --max-num-seqs 8 \
    --gpu-memory-utilization 0.85 \
    --tensor-parallel-size 2 \
    --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --host 0.0.0.0 --port 8000
