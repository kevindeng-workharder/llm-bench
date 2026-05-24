#!/bin/bash
# vLLM graph mode, DUAL 7900 XTX (TP=2), Qwen3.6-27B Quark W8A8-INT8.
#
# Variant of qwen3_6-27b-quark-int8-graph-tp2.sh with:
#   - --max-model-len 49152            (48K context, vs 2048 default)
#   - --max-num-seqs 4                  (KV-cache budget for 48K context)
#   - --max-num-batched-tokens 2048     (faster prefill of long inputs)
#   - --reasoning-parser qwen3          (split <think> blocks into a
#                                         separate `reasoning` field so
#                                         Aider / function-call clients
#                                         see clean `content`)
#   - SHM RCCL transport via the topology-XML workaround documented in
#     kevindeng-workharder/rocm-riscv-build:
#       runtime-patches/rccl-topo-fix/   (branch p2p-deadlock-fix-2026-05-23)
#
# KV cache headroom at this config: ~183K tokens total (well above
# 4 * 49152 = 196K worst-case; head_dim 256 with GQA 4 keeps KV tiny).
#
# Boot time: ~13 min cold (incl. cudagraph capture); warm ~3 min.
set -e
source /home/ubuntu/vllm-serve-env.sh

# Force SHM via topology XML (NCCL picks via SHM/direct/direct instead of P2P/IPC)
export NCCL_TOPO_FILE=/home/ubuntu/rccl-topo-split.xml

# Carry over the vLLM RPC timeouts (long, for JIT)
export VLLM_RPC_TIMEOUT=7200000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=7200

# SHM transport (RCCL chooses via SHM/direct/direct now that topology says cross-CPU)
# (these are no-ops with the split topo but kept for clarity / safety)
export NCCL_SHM_DISABLE=0

# Logging
export NCCL_DEBUG=INFO

exec /data/vllm0.21-pt2.11/bin/python -m vllm.entrypoints.openai.api_server \
    --model /data/Qwen3.6-27B-Quark-W8A8-INT8 \
    --served-model-name qwen3_6-27b-quark-int8 \
    --dtype bfloat16 \
    --quantization quark \
    --max-model-len 49152 \
    --max-num-seqs 4 \
    --max-num-batched-tokens 2048 \
    --gpu-memory-utilization 0.85 \
    --tensor-parallel-size 2 \
    --trust-remote-code \
    --no-enable-prefix-caching \
    --limit-mm-per-prompt '{"image":0,"video":0}' \
    --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4],"max_cudagraph_capture_size":4,"cudagraph_num_of_warmups":0}' \
    --reasoning-parser qwen3 \
    --host 0.0.0.0 --port 8000
