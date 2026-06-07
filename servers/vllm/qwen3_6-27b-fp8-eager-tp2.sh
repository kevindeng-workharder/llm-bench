#!/bin/bash
# vLLM EAGER mode, DUAL 7900 XTX (TP=2), Qwen3.6-27B-FP8 (dense fp8 block quant).
#
# Eager path: skips CUDAGraph capture, so startup avoids compiling a kernel per
# capture size — but per-step decode is slower (no graph replay). Mirrors the
# in-process eager benchmark on the 199 box (gpu_mem 0.88, enforce_eager).
#
# Self-contained: sources the on-VM ROCm 7.2.3 + PyTorch 2.11 runtime env, then
# execs the OpenAI api_server directly. See the graph-tp2 sibling for notes on
# the slow first-compile and the multimodal text-only override.
set -e
source /home/ubuntu/vllm-serve-env.sh
# 27B fp8: a batch>1 step makes the worker JIT-compile new kernel shapes; vLLM's
# default ~60s RPC timeout then kills EngineCore mid-compile (shm_broadcast
# TimeoutError -> "RPC call to sample_tokens timed out" -> EngineDeadError).
# The in-process bench set these long timeouts; serve needs them too so the
# first concurrent batch can warm up the kernels.
export VLLM_RPC_TIMEOUT=7200000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=7200
# Enable SHM transport for RCCL (socket deadlocks concurrent allgather at N>=2;
# SHM verified working on the AWQ sibling). See graph-tp2 for details.
export NCCL_SHM_DISABLE=0
exec /home/ubuntu/vllm-venv/bin/python -m vllm.entrypoints.openai.api_server \
    --model /data/Qwen3.6-27B-FP8 \
    --served-model-name qwen3_6-27b-fp8 \
    --dtype bfloat16 \
    --quantization fp8 \
    --max-model-len 2048 \
    --max-num-seqs 8 \
    --max-num-batched-tokens 512 \
    --gpu-memory-utilization 0.88 \
    --tensor-parallel-size 2 \
    --trust-remote-code \
    --no-enable-prefix-caching \
    --limit-mm-per-prompt '{"image":0,"video":0}' \
    --enforce-eager \
    --host 0.0.0.0 --port 8000
