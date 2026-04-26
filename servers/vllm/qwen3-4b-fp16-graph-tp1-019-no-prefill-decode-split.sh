#!/bin/bash
# Probe C: vLLM 0.19 + Qwen3-4B fp16 graph TP1 with both:
#   - VLLM_V1_USE_PREFILL_DECODE_ATTENTION=0   (don't split prefill/decode)
#   - VLLM_ROCM_CUSTOM_PAGED_ATTN=0            (don't use C++ rocm_paged)
# So we use the unified vLLM 0.19 default attention path with no rocm-
# specific specializations. If batched correctness comes back here, the
# bug is in the rocm-specific code (either the split scheduler or the
# C++ paged kernel — A vs B will narrow it).
set -e
source /home/ubuntu/vllm-serve/server-env.sh
export VLLM_V1_USE_PREFILL_DECODE_ATTENTION=0
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
