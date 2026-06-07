#!/bin/bash
# vLLM graph mode, single 7900 XTX, Qwen3-4B (DENSE fp16). The canonical 4B
# control config — verified previously to batch correctly.
#
# SELF-CONTAINED 2026-06-07: the original was a thin wrapper around
# /home/ubuntu/vllm-serve/launch-server.sh, which is absent from BOTH this rootfs
# and the repo (only the launcher was archived, not the wrapper). So the launch is
# inlined here — same approach as gemma4-e2b-card1-dual.sh — sourcing the present
# /home/ubuntu/vllm-serve-env.sh and the rootfs vllm-venv. Was VENV_PREFIX=ai-2.10
# (also missing); now /home/ubuntu/vllm-venv (vLLM 0.21).
set -eu
source /home/ubuntu/vllm-serve-env.sh
export ROCR_VISIBLE_DEVICES=0          # pin to one physical GPU (TP=1)
unset HIP_VISIBLE_DEVICES 2>/dev/null || true
export VLLM_RPC_TIMEOUT=7200000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=7200
exec /home/ubuntu/vllm-venv/bin/python -m vllm.entrypoints.openai.api_server \
    --model /data/Qwen3-4B \
    --served-model-name qwen3-4b \
    --dtype float16 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.85 \
    --tensor-parallel-size 1 \
    --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8],"max_cudagraph_capture_size":8,"cudagraph_num_of_warmups":0}' \
    --host 0.0.0.0 --port 8000
