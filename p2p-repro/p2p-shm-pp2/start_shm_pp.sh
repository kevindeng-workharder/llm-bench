#!/bin/bash
# vLLM PIPELINE-PARALLEL (PP=2), DUAL 7900 XTX, Qwen3.6-27B Quark W8A8-INT8 (text-only).
#
# Splits the 64 GDN/full-attn layers across the two GPUs BY DEPTH (Worker_PP0 = first
# half of layers, Worker_PP1 = second half), with ONE activation handoff per stage
# boundary — vs qwen3_6-27b-quark-int8-graph-tp2.sh which splits each layer (TP) and
# does an all-reduce every layer.
#
# Result (see docs/qwen3_6-27b-pipeline-parallel.md): PP works on this hybrid GDN model
# (it declares SupportsPP) but is ~8-19% SLOWER than TP on single-VM — fast Infinity-
# Fabric/SHM makes TP's all-reduce cheap, and PP can't recover its batch=1 pipeline
# bubble. PP's win is slow interconnects (cross-VM IB), not single-VM.
#
# WARNING: the first boot with cudagraph sizes [1,2,4,8] triggers a ~2-hour GDN-kernel
# JIT recompile on the riscv64/TCG guest (shapes not yet in ~/.triton/cache; GPUs idle,
# CPU-bound). It is NOT a hang (the Triton cache grows steadily). Cached after — a
# same-config reboot is fast.
#
# Runs on /home/ubuntu/vllm-venv (gemv INT8 patch baked in), same as the other launchers' siblings.
set -eu
source /home/ubuntu/vllm-serve-env.sh
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
export CC=/opt/rocm/llvm/bin/clang
export VLLM_NCCL_SO_PATH=/home/ubuntu/librccl-rebuilt.so.1.0
export LD_PRELOAD=/home/ubuntu/librccl-rebuilt.so.1.0
export VLLM_RPC_TIMEOUT=7200000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=7200
export NCCL_TOPO_FILE=/home/ubuntu/rccl-topo-split.xml   # split topo -> SHM transport (p2p-shm)
exec /home/ubuntu/vllm-venv/bin/python -m vllm.entrypoints.openai.api_server \
    --model /data/Qwen3.6-27B-Quark-W8A8-INT8 \
    --served-model-name qwen3_6-27b-int8 \
    --quantization quark --dtype bfloat16 \
    --max-model-len 8192 --max-num-seqs 8 --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.85 \
    --pipeline-parallel-size 2 --distributed-executor-backend mp \
    --disable-custom-all-reduce \
    --trust-remote-code --no-enable-prefix-caching \
    --limit-mm-per-prompt '{"image":0,"video":0}' \
    --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8],"max_cudagraph_capture_size":8,"cudagraph_num_of_warmups":0}' \
    --host 0.0.0.0 --port 8000
