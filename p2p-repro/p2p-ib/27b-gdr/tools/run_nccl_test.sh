#!/bin/bash
# Minimal 2-node RCCL all-reduce over IB to check DMA_BUF/GDR fast (no model load).
# usage: run_nccl_test.sh <rank 0|1> <socket_ifname enP3p1s0np0|np1>
set -u
source /home/ubuntu/vllm-serve-env.sh
unset NCCL_TOPO_FILE
export NCCL_IB_DISABLE=0 NCCL_NET=IB NCCL_IB_HCA=roceP3p1s0 NCCL_IB_GID_INDEX=3
export NCCL_DMABUF_ENABLE=1 NCCL_NET_GDR_LEVEL=SYS NCCL_DEBUG=INFO
# RCCL_FORCE_ENABLE_DMABUF no longer needed (de-bypassed 2026-06-11): librccl's
# rocmwrap.cc gzip fix reads /boot/config and self-determines DMA_BUF support.
export NCCL_SOCKET_IFNAME=$2 GLOO_SOCKET_IFNAME=$2 TP_SOCKET_IFNAME=$2
export VLLM_NCCL_SO_PATH=/home/ubuntu/librccl-rebuilt.so.1.0 LD_PRELOAD=/home/ubuntu/librccl-rebuilt.so.1.0
export CC=/opt/rocm/llvm/bin/clang
export MASTER_ADDR=10.99.0.1 MASTER_PORT=29555 RANK=$1 WORLD_SIZE=2
exec /data/vllm0.21-pt2.11/bin/python /home/ubuntu/nccl_alltest.py
