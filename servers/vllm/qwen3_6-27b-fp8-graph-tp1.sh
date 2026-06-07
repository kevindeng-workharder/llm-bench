#!/bin/bash
# vLLM graph mode, SINGLE 7900 XTX (TP=1), Qwen3.6-27B-FP8 (dense fp8 quant).
#
# WARNING: 27B at FP8 is ~27 GB of weights — it does NOT fit a single 24 GB
# card. This will almost certainly OOM at load; kept only as a reference /
# completeness sibling. Use the tp2 variant for real runs.
#
# Self-contained: sources the on-VM ROCm 7.2.3 + PyTorch 2.11 runtime env, then
# execs the OpenAI api_server directly.
set -e
source /home/ubuntu/vllm-serve-env.sh
# 27B fp8: a batch>1 step makes the worker JIT-compile new kernel shapes; vLLM's
# default ~60s RPC timeout then kills EngineCore mid-compile (shm_broadcast
# TimeoutError -> "RPC call to sample_tokens timed out" -> EngineDeadError).
# The in-process bench set these long timeouts; serve needs them too so the
# first concurrent batch can warm up the kernels.
export VLLM_RPC_TIMEOUT=7200000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=7200
# Enable SHM transport for RCCL (socket deadlocks concurrent allgather; SHM
# verified on the AWQ sibling). TP=1 barely uses it, kept for consistency.
export NCCL_SHM_DISABLE=0
exec /home/ubuntu/vllm-venv/bin/python -m vllm.entrypoints.openai.api_server \
    --model /data/Qwen3.6-27B-FP8 \
    --served-model-name qwen3_6-27b-fp8 \
    --dtype bfloat16 \
    --quantization fp8 \
    --max-model-len 2048 \
    --max-num-seqs 8 \
    --max-num-batched-tokens 512 \
    --gpu-memory-utilization 0.92 \
    --tensor-parallel-size 1 \
    --trust-remote-code \
    --no-enable-prefix-caching \
    --limit-mm-per-prompt '{"image":0,"video":0}' \
    --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8],"max_cudagraph_capture_size":8,"cudagraph_num_of_warmups":0}' \
    --host 0.0.0.0 --port 8000
