#!/bin/bash
# p2p-shm: single-VM dual-GPU TP=2 forced onto NCCL's host-SHM transport.
# = the p2p-direct launcher, but:
#   - NCCL_TOPO_FILE = rccl-topo-split.xml (GPUs faked onto different NUMA -> RCCL picks SHM, not P2P/IPC)
#   - custom all-reduce DISABLED so the TP all-reduce goes NCCL->SHM (not vLLM CA over Infinity Fabric)
#   - everything else identical to p2p-direct (2048 ctx, text) for an apples-to-apples transport comparison
# async-scheduling left ON (single-VM, no cross-VM hang). Deploy to /home/ubuntu/p2p-shm-2gpu.sh.
set -eu
source /home/ubuntu/vllm-serve-env.sh
export NCCL_TOPO_FILE=/home/ubuntu/rccl-topo-split.xml   # <-- the split topo forces SHM (overrides serve-env default)
export NCCL_SHM_DISABLE=0
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
export CC=/opt/rocm/llvm/bin/clang
export VLLM_NCCL_SO_PATH=/home/ubuntu/librccl-rebuilt.so.1.0
export LD_PRELOAD=/home/ubuntu/librccl-rebuilt.so.1.0
export PYTHONPATH=/home/ubuntu:${PYTHONPATH:-}
export VLLM_RPC_TIMEOUT=7200000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=7200
exec /data/vllm0.21-pt2.11/bin/python -m vllm.entrypoints.openai.api_server \
    --model /data/Qwen3.6-27B-Quark-W8A8-INT8 \
    --served-model-name qwen3_6-27b-int8 \
    --quantization quark \
    --dtype bfloat16 \
    --max-model-len 2048 \
    --max-num-seqs 8 \
    --max-num-batched-tokens 2048 \
    --gpu-memory-utilization 0.85 \
    --tensor-parallel-size 2 \
    --distributed-executor-backend mp \
    --trust-remote-code \
    --no-enable-prefix-caching \
    --limit-mm-per-prompt '{"image":0,"video":0}' \
    --disable-custom-all-reduce \
    --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4],"max_cudagraph_capture_size":4,"cudagraph_num_of_warmups":0}' \
    --host 0.0.0.0 --port 8000
