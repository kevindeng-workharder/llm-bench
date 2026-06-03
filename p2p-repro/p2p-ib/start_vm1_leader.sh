#!/bin/bash
# ============================================================================
# p2p-ib scenario — LEADER (node 0, VM1)
# Cross-VM vLLM TP=2 over Mellanox CX-7 RoCE (NCCL NET=IB).
# Qwen3-4B FP16, graph mode (FULL_DECODE_ONLY).
#
# VERIFIED WORKING 2026-06-03: reaches "Application startup complete" on
# 0.0.0.0:8000 in ~420 s; serves /v1/completions; ~10 tok/s (peak 12.5)
# single-stream decode. See RESULTS.md.
#
# DEPLOY TO: VM1 (guest reachable via `ssh -p 2224`) at /home/ubuntu/graph_vm.sh
# LAUNCH:    ssh -p 2224 ubuntu@127.0.0.1 \
#              'cd /home/ubuntu && setsid bash graph_vm.sh </dev/null >/tmp/vllm_vm1.log 2>&1'
#
# The follower (VM2) MUST be launched with start_vm2_follower_headless.sh —
# NOT a second api_server. See README.md "Blocker: follower headless".
# ============================================================================
set -eu
source /home/ubuntu/vllm-serve-env.sh

# --- overrides for cross-VM IB (undo the single-host-2-GPU defaults) ---
unset NCCL_TOPO_FILE                  # the 2-GPU-same-host XML does not apply cross-VM
unset NCCL_P2P_DISABLE NCCL_SHM_DISABLE
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=roceP3p1s0         # MUST be roceP3p1s0, NOT mlx5_0 (guest udev name)
export NCCL_NET=IB
export NCCL_IB_GID_INDEX=3            # RoCEv2 GID
export NCCL_SOCKET_IFNAME=enP3p1s0np0 # VM1's RoCE iface (bootstrap/OOB)
export GLOO_SOCKET_IFNAME=enP3p1s0np0
export TP_SOCKET_IFNAME=enP3p1s0np0
export VLLM_HOST_IP=10.99.0.1         # CRITICAL: else MQ picks QEMU NAT 10.0.2.15 and deadlocks
export NCCL_DEBUG=INFO

export VLLM_RPC_TIMEOUT=7200000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=7200

# librccl with -g debug symbols (RCCL 2.27.7-HEAD:96a25b5+). The canonical
# /opt/rocm-riscv-7.2.3/lib/librccl.so.1 is the SAME RCCL; the LD_PRELOAD here
# is only to get nicer stacks if it ever hangs. Drop both lines to use the
# installed librccl (VLLM_NCCL_SO_PATH from vllm-serve-env.sh already points there).
export VLLM_NCCL_SO_PATH=/home/ubuntu/librccl-rebuilt.so.1.0
export LD_PRELOAD=/home/ubuntu/librccl-rebuilt.so.1.0

# sitecustomize.py registers faulthandler on SIGUSR1 (dump all Python stacks
# without killing the process). Optional; useful for debugging a hang.
export PYTHONPATH=/home/ubuntu:${PYTHONPATH:-}

exec /data/vllm0.21-pt2.11/bin/python -m vllm.entrypoints.openai.api_server \
    --model /data/Qwen3-4B \
    --served-model-name qwen3-4b \
    --dtype float16 \
    --max-model-len 4096 \
    --max-num-seqs 4 \
    --max-num-batched-tokens 2048 \
    --gpu-memory-utilization 0.85 \
    --tensor-parallel-size 2 \
    --nnodes 2 \
    --node-rank 0 \
    --master-addr 10.99.0.1 \
    --master-port 29500 \
    --distributed-executor-backend mp \
    --trust-remote-code \
    --no-enable-prefix-caching \
    --disable-custom-all-reduce \
    --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8],"max_cudagraph_capture_size":8,"cudagraph_num_of_warmups":0}' \
    --host 0.0.0.0 --port 8000
