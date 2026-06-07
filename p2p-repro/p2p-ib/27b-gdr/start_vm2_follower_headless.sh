#!/bin/bash
# p2p-ib 27B GRAPH + FIX — FOLLOWER (node 1, VM2) — HEADLESS
# --no-async-scheduling (see leader); engine args must match leader.
set -eu
source /home/ubuntu/vllm-serve-env.sh
export RCCL_FORCE_ENABLE_DMABUF=1   # GDR: bypass RCCL gzip /proc/config.gz dmabuf check (rocmwrap.cc); needs the unified kernel for the real pci_p2pdma registration
export NCCL_NET_GDR_LEVEL=SYS       # GDR: allow GPUDirect over the cross-root PCIe path (else NCCL stays GDR 0); matches tools/run_nccl_test.sh
export NCCL_DMABUF_ENABLE=1
# riscv64: use ROCm clang for Triton's C launcher compile. Stock gcc (cc1)
# heap-corrupts (free(): invalid size -> SIGABRT) building scaled_mm_kernel's
# __triton_launcher.c in the loaded-model worker. Triton's build.py honors $CC.
export CC=/opt/rocm/llvm/bin/clang
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
exec /home/ubuntu/vllm-venv/bin/python /home/ubuntu/vllm-venv/bin/vllm serve /data/Qwen3.6-27B-Quark-W8A8-INT8 \
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
    --no-async-scheduling \
    --mm-processor-kwargs '{"max_pixels":451584,"min_pixels":3136}' \
    --disable-custom-all-reduce \
    --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4],"max_cudagraph_capture_size":4,"cudagraph_num_of_warmups":0}' \
    --headless
