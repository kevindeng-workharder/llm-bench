#!/bin/bash
# p2p-ib 27B GRAPH test — FOLLOWER (node 1, VM2) — HEADLESS
# Engine args (incl. compilation-config) MUST match the leader.
set -eu
source /home/ubuntu/vllm-serve-env.sh
unset NCCL_TOPO_FILE
unset NCCL_P2P_DISABLE NCCL_SHM_DISABLE
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=roceP3p1s0
export NCCL_NET=IB
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=enP3p1s0np1
export GLOO_SOCKET_IFNAME=enP3p1s0np1
export TP_SOCKET_IFNAME=enP3p1s0np1
export VLLM_HOST_IP=10.99.0.2
export NCCL_DEBUG=INFO
export VLLM_RPC_TIMEOUT=7200000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=7200
export VLLM_NCCL_SO_PATH=/home/ubuntu/librccl-rebuilt.so.1.0
export LD_PRELOAD=/home/ubuntu/librccl-rebuilt.so.1.0
export PYTHONPATH=/home/ubuntu:${PYTHONPATH:-}
exec /data/vllm0.21-pt2.11/bin/vllm serve /data/Qwen3.6-27B-Quark-W8A8-INT8 \
    --served-model-name qwen3_6-27b-int8 \
    --quantization quark \
    --dtype bfloat16 \
    --max-model-len 2048 \
    --max-num-seqs 8 \
    --max-num-batched-tokens 2048 \
    --gpu-memory-utilization 0.85 \
    --tensor-parallel-size 2 \
    --nnodes 2 \
    --node-rank 1 \
    --master-addr 10.99.0.1 \
    --master-port 29500 \
    --distributed-executor-backend mp \
    --trust-remote-code \
    --no-enable-prefix-caching \
    --mm-processor-kwargs '{"max_pixels":451584,"min_pixels":3136}' \
    --disable-custom-all-reduce \
    --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4],"max_cudagraph_capture_size":4,"cudagraph_num_of_warmups":0}' \
    --headless
