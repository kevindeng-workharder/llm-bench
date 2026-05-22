#!/bin/bash
# vLLM graph mode, DUAL 7900 XTX (TP=2), Qwen3.6-27B-FP8 (dense fp8 block quant).
#
# Self-contained: sources the on-VM ROCm 7.2.3 + PyTorch 2.11 runtime env, then
# execs the OpenAI api_server directly (the env-driven launch-server.sh shim is
# not present on this VM). Params mirror the in-process benchmark verified on
# the 199 dual-GPU box (graph / FULL_DECODE_ONLY).
#
# NOTE: first start is SLOW — 27B load (~100s) plus first-time FP8 W8A8 block
# triton kernel compile (~5 min/shape on this riscv Triton). Give the runner a
# large ready_timeout_s (see configs/bench-matrix.yaml).
#
# 27B is a multimodal (qwen3_5) checkpoint; --limit-mm-per-prompt 0/0 forces
# text-only so vLLM skips the multimodal profile_run (which needs torchvision).
set -e
source /home/ubuntu/vllm-serve-env.sh
# 27B fp8: a batch>1 step makes the worker JIT-compile new kernel shapes; vLLM's
# default ~60s RPC timeout then kills EngineCore mid-compile (shm_broadcast
# TimeoutError -> "RPC call to sample_tokens timed out" -> EngineDeadError).
# The in-process bench set these long timeouts; serve needs them too so the
# first concurrent batch can warm up the kernels.
export VLLM_RPC_TIMEOUT=7200000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=7200
# Enable SHM transport for RCCL. Socket (vllm-serve-env.sh default) deadlocks
# under concurrent allgather at N>=2 (600s NCCL watchdog -> EngineDeadError);
# SHM is verified working on the AWQ sibling (N=8 = 34 t/s vs socket crash).
export NCCL_SHM_DISABLE=0
exec /data/ai-2.11/bin/python -m vllm.entrypoints.openai.api_server \
    --model /data/Qwen3.6-27B-FP8 \
    --served-model-name qwen3_6-27b-fp8 \
    --dtype bfloat16 \
    --quantization fp8 \
    --max-model-len 2048 \
    --max-num-seqs 8 \
    --max-num-batched-tokens 512 \
    --gpu-memory-utilization 0.85 \
    --tensor-parallel-size 2 \
    --trust-remote-code \
    --no-enable-prefix-caching \
    --limit-mm-per-prompt '{"image":0,"video":0}' \
    --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8],"max_cudagraph_capture_size":8,"cudagraph_num_of_warmups":0}' \
    --host 0.0.0.0 --port 8000
