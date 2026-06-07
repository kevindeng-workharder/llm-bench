#!/bin/bash
# p2p-ib 27B PP (pipeline-parallel) + GDR — LEADER (node 0, VM1)
# Root cause of the sustained-gen hang: vLLM async-scheduling's
# async_copy_ready_event.synchronize() never returns under cudagraph (the
# non-blocking D2H sampled-token copy's event isn't signalled on replay).
# Fix: --no-async-scheduling (keeps cudagraph speed, uses the synchronous
# Scheduler / no async copy event). Watchdog left at default (no ASYNC=0 hack).
set -eu
source /home/ubuntu/vllm-serve-env.sh
export RCCL_FORCE_ENABLE_DMABUF=1   # GDR: bypass RCCL gzip /proc/config.gz dmabuf check (rocmwrap.cc); needs the unified kernel for the real pci_p2pdma registration
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
export NCCL_SOCKET_IFNAME=enP3p1s0np0
export GLOO_SOCKET_IFNAME=enP3p1s0np0
export TP_SOCKET_IFNAME=enP3p1s0np0
export VLLM_HOST_IP=10.99.0.1
export NCCL_DEBUG=INFO
export VLLM_RPC_TIMEOUT=7200000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=7200
export VLLM_NCCL_SO_PATH=/home/ubuntu/librccl-rebuilt.so.1.0
export LD_PRELOAD=/home/ubuntu/librccl-rebuilt.so.1.0
export PYTHONPATH=/home/ubuntu:${PYTHONPATH:-}
exec /home/ubuntu/vllm-venv/bin/python -m vllm.entrypoints.openai.api_server \
    --model /data/Qwen3.6-27B-Quark-W8A8-INT8 \
    --served-model-name qwen3_6-27b-int8 \
    --quantization quark \
    --dtype bfloat16 \
    --max-model-len 16384 \
    --max-num-seqs 1 \
    --max-num-batched-tokens 16384 \
    --gpu-memory-utilization 0.93 \
    --pipeline-parallel-size 2 \
    --nnodes 2 \
    --node-rank 0 \
    --master-addr 10.99.0.1 \
    --master-port 29500 \
    --distributed-executor-backend mp \
    --trust-remote-code \
    --no-enable-prefix-caching \
    --no-async-scheduling \
    --limit-mm-per-prompt '{"image":1,"video":1}' \
    --mm-processor-kwargs '{"max_pixels":200704}' \
    --mm-encoder-attn-backend TRITON_ATTN \
    --media-io-kwargs '{"video":{"backend":"pyav"}}' \
    --disable-custom-all-reduce \
    --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1],"max_cudagraph_capture_size":1,"cudagraph_num_of_warmups":0}' \
    --host 0.0.0.0 --port 8000
