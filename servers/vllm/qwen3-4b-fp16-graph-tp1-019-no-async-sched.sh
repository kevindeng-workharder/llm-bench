#!/bin/bash
# Probe D: vLLM 0.19 + Qwen3-4B fp16 graph TP1 with --no-async-scheduling.
# Async scheduling was flipped to default-on in PR #27614 (merged 2025-12-29,
# in the v0.11→v0.19 window). Our startup log says
# "Asynchronous scheduling is enabled." If batched correctness comes back
# with this flag, async scheduling is the regression vector.
set -e
source /home/ubuntu/vllm-serve/server-env.sh
exec /home/ubuntu/ai-2.10/bin/python3 -m vllm.entrypoints.openai.api_server \
    --model /data/Qwen3-4B \
    --served-model-name qwen3-4b \
    --dtype float16 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.85 \
    --tensor-parallel-size 1 \
    --no-async-scheduling \
    --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --host 0.0.0.0 --port 8000
