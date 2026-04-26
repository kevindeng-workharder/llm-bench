#!/bin/bash
# vLLM graph mode, single 7900 XTX, Qwen3-4B fp16 (DENSE — not MoE).
# This is our control config to verify vLLM concurrent batching works
# correctly on this stack when MoE is not in the picture. Same env, same
# binaries, same launch flow as the MoE configs.
set -e
source /home/ubuntu/vllm-serve/server-env.sh
exec /home/ubuntu/ai-2.10/bin/python3 -m vllm.entrypoints.openai.api_server \
    --model /data/Qwen3-4B \
    --served-model-name qwen3-4b \
    --dtype float16 \
    --max-model-len 4096 \
    --max-num-seqs 8 \
    --gpu-memory-utilization 0.85 \
    --tensor-parallel-size 1 \
    --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --host 0.0.0.0 --port 8000
