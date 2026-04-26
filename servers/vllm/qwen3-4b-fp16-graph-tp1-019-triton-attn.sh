#!/bin/bash
# Probe B: vLLM 0.19 + Qwen3-4B fp16 graph TP1 with VLLM_ATTENTION_BACKEND=TRITON_ATTN.
# Forces the pure-triton attention backend (vs the default ROCM_ATTN which
# uses C++ paged_attention_rocm). If batched correctness comes back here,
# the bug is in the C++ rocm paged-attention kernel.
set -e
source /home/ubuntu/vllm-serve/server-env.sh
export VLLM_ATTENTION_BACKEND=TRITON_ATTN
exec /home/ubuntu/ai-2.10/bin/python3 -m vllm.entrypoints.openai.api_server \
    --model /data/Qwen3-4B \
    --served-model-name qwen3-4b \
    --dtype float16 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.85 \
    --tensor-parallel-size 1 \
    --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --host 0.0.0.0 --port 8000
