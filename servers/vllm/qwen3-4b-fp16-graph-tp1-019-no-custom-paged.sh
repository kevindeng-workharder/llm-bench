#!/bin/bash
# Probe: vLLM 0.19 + Qwen3-4B fp16 graph TP1 with VLLM_ROCM_CUSTOM_PAGED_ATTN=0.
# Disables the C++ paged_attention_rocm kernel and forces vLLM to use the
# triton-based paged-attention path. If batched correctness comes back,
# the bug is in paged_attention_rocm.
set -e
source /home/ubuntu/vllm-serve/server-env.sh
export VLLM_ROCM_CUSTOM_PAGED_ATTN=0
exec /home/ubuntu/ai-2.10/bin/python3 -m vllm.entrypoints.openai.api_server \
    --model /data/Qwen3-4B \
    --served-model-name qwen3-4b \
    --dtype float16 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.85 \
    --tensor-parallel-size 1 \
    --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --host 0.0.0.0 --port 8000
