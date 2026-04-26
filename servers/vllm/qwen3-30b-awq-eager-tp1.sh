#!/bin/bash
# vLLM eager mode, single 7900 XTX, Qwen3-30B-A3B-Instruct-2507-AWQ.
# COMPILATION: --enforce-eager (no torch.compile, no cudagraphs).
# Same fused-MoE batched bug as graph mode (N>=2 garbage), but slower.
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
    --enforce-eager \
    --host 0.0.0.0 --port 8000
