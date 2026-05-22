#!/bin/bash
# vLLM graph mode, DUAL 7900 XTX (TP=2), Qwen3.6-27B Quark W8A8-INT8.
#
# Quark = AMD's quantizer; this is W8A8 INT8 (per-tensor int8 weights + int8
# activations, pack_method=reorder). Same qwen3_5 multimodal checkpoint, so
# --limit-mm-per-prompt 0/0 forces text-only.
#
# Self-contained: sources the on-VM ROCm 7.2.3 + PyTorch 2.11 runtime env, then
# execs the OpenAI api_server directly.
set -e
source /home/ubuntu/vllm-serve-env.sh
# First batch>1 step JIT-compiles new kernel shapes; long RPC timeout so the
# default ~60s window doesn't kill EngineCore mid-compile.
export VLLM_RPC_TIMEOUT=7200000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=7200
# Enable SHM transport for RCCL (host-RAM ferry between TP workers). Socket
# (vllm-serve-env.sh default) deadlocks under concurrent allgather at N>=2
# (600s NCCL watchdog -> EngineDeadError). SHM is verified working + faster.
export NCCL_SHM_DISABLE=0
exec /data/ai-2.11/bin/python -m vllm.entrypoints.openai.api_server \
    --model /data/Qwen3.6-27B-Quark-W8A8-INT8 \
    --served-model-name qwen3_6-27b-quark-int8 \
    --dtype bfloat16 \
    --quantization quark \
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
