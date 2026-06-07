#!/bin/bash
# p2p-direct: single-VM dual-GPU TP=2 over Infinity Fabric (NCCL via P2P/IPC).
# Derived from p2p-ib 27b-graph's leader, with the two single-VM speedups:
#   - custom all-reduce ENABLED  (no --disable-custom-all-reduce): CA over Infinity Fabric
#   - async-scheduling  ENABLED  (no --no-async-scheduling): single-VM doesn't hit the
#                                  cross-VM "async x cudagraph" hang (verified clean)
#   - single node (no --nnodes/--node-rank/--master-*), NCCL_IB_DISABLE=1
#   - keeps NCCL_TOPO_FILE=rccl-topo.xml (fake same-root -> RCCL selects P2P/IPC)
# Requires: unified kernel Image-6.19.5-p2p-all (cpu_supports_p2pdma hack) + guest iommu.passthrough=1
#           + the gemv INT8 patch (via sitecustomize + PYTHONPATH). Deploy to /home/ubuntu/p2p-direct-2gpu.sh.
set -eu
source /home/ubuntu/vllm-serve-env.sh   # default NCCL_TOPO_FILE=rccl-topo.xml + HSA_FORCE_FINE_GRAIN_PCIE=1 + NCCL_IB_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
export CC=/opt/rocm/llvm/bin/clang
export VLLM_NCCL_SO_PATH=/home/ubuntu/librccl-rebuilt.so.1.0
export LD_PRELOAD=/home/ubuntu/librccl-rebuilt.so.1.0
export PYTHONPATH=/home/ubuntu:${PYTHONPATH:-}
export VLLM_RPC_TIMEOUT=7200000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=7200
exec /home/ubuntu/vllm-venv/bin/python -m vllm.entrypoints.openai.api_server \
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
    --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4],"max_cudagraph_capture_size":4,"cudagraph_num_of_warmups":0}' \
    --host 0.0.0.0 --port 8000
