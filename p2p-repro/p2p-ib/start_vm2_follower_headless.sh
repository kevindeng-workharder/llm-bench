#!/bin/bash
# ============================================================================
# p2p-ib scenario — FOLLOWER (node 1, VM2) — HEADLESS
# Cross-VM vLLM TP=2 over Mellanox CX-7 RoCE (NCCL NET=IB).
#
# *** THIS IS THE FIX that took the longest to find (2026-06-03). ***
# On this vLLM dev build (v0.21.1.dev0+gad7125a43, built d20260522) the
# follower node must run `vllm serve --headless`, NOT
# `python -m vllm.entrypoints.openai.api_server`.
#
# Why: the api_server entrypoint has no headless branch, so on the follower it
# spins up a full EngineCore which calls _initialize_kv_caches() ->
# collective_rpc("get_kv_cache_spec") and trips
#     assert self.rpc_broadcast_mq is not None
#     "collective_rpc should not be called on follower node"
# (multiproc_executor.py). The EngineCore dies and takes the follower's whole
# process tree with it — including the Worker that had already joined NCCL.
# The leader's Worker then waits forever for rank 1 at the first big collective
# (profile_run logits all_gather), which LOOKS like an all_gather hang but is
# really "the peer rank is dead".
#
# `vllm serve --headless` -> cli/serve.py:run_headless() -> when
# node_rank_within_dp>0 it builds ONLY MultiprocExecutor(monitor_workers=False)
# + start_worker_monitor(inline=True): a worker-only process, no EngineCore,
# no assert. It stays resident and serves RPCs broadcast by the leader.
#
# VERIFIED WORKING 2026-06-03: follower process tree is `vllm serve` +
# VLLM::Worker, with NO VLLM::EngineCore. See RESULTS.md.
#
# DEPLOY TO: VM2 (guest reachable via `ssh -p 2225`) at /home/ubuntu/graph_vm.sh
# LAUNCH:    ssh -p 2225 ubuntu@127.0.0.1 \
#              'cd /home/ubuntu && setsid bash graph_vm.sh </dev/null >/tmp/vllm_vm2.log 2>&1'
# ============================================================================
set -eu
source /home/ubuntu/vllm-serve-env.sh

# --- overrides for cross-VM IB (mirror the leader, VM2's iface/IP) ---
unset NCCL_TOPO_FILE
unset NCCL_P2P_DISABLE NCCL_SHM_DISABLE
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=roceP3p1s0
export NCCL_NET=IB
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=enP3p1s0np1  # VM2's RoCE iface
export GLOO_SOCKET_IFNAME=enP3p1s0np1
export TP_SOCKET_IFNAME=enP3p1s0np1
export VLLM_HOST_IP=10.99.0.2          # VM2's IB IP
export NCCL_DEBUG=INFO

export VLLM_RPC_TIMEOUT=7200000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=7200

export VLLM_NCCL_SO_PATH=/home/ubuntu/librccl-rebuilt.so.1.0
export LD_PRELOAD=/home/ubuntu/librccl-rebuilt.so.1.0
export PYTHONPATH=/home/ubuntu:${PYTHONPATH:-}

# NOTE: `vllm serve <model> --headless` (NOT api_server). --headless sets
# api_server_count=0 -> run_headless(). With DP=1 (no --data-parallel-* args)
# data_parallel_size_local resolves to 1, so the node_rank_within_dp>0 worker
# path is taken. --host/--port are omitted: a headless node starts no server.
# Engine args (model, TP, compilation-config, ...) MUST match the leader.
exec /data/vllm0.21-pt2.11/bin/vllm serve /data/Qwen3-4B \
    --served-model-name qwen3-4b \
    --dtype float16 \
    --max-model-len 4096 \
    --max-num-seqs 4 \
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
    --disable-custom-all-reduce \
    --headless \
    --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8],"max_cudagraph_capture_size":8,"cudagraph_num_of_warmups":0}'
