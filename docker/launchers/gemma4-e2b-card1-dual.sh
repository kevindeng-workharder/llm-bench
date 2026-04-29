#!/bin/bash
# vLLM + gemma-4-E4B-it on card1 (7900 XT, 20 GB).
#
# Companion launcher to qwen36-35b-mxfp4-card0-dual.sh: that one runs on
# card0, this one runs on card1. Both expose OpenAI-compatible endpoints
# on different ports so a host-side client can hit each independently.
#
# gemma-4-E4B is multimodal (Gemma4ForConditionalGeneration) but we use
# it text-only here. Architecture: dense, hidden_size matches a roughly
# 4B-equivalent text decoder. Safetensors size 14.9 GB → ~5 GB headroom
# on a 20 GB card for KV cache + activations.
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../../vllm-serve/server-env.sh" 2>/dev/null || \
    source /home/ubuntu/vllm-serve/server-env.sh

# Bind to card1 only (card0 is the llama.cpp server's).
# IMPORTANT: HIP_VISIBLE_DEVICES alone makes torch see zero devices on this
# riscv cross-compile setup. Use ROCR_VISIBLE_DEVICES (the lower-level HSA
# selector) without HIP_VISIBLE_DEVICES to filter to a single physical GPU.
unset HIP_VISIBLE_DEVICES
export ROCR_VISIBLE_DEVICES=1

# gemma-4 path quirks observed during chat-gemma4 setup:
export TOKENIZERS_PARALLELISM=false

exec /home/ubuntu/ai-2.10/bin/python3 -m vllm.entrypoints.openai.api_server \
    --model /data/gemma-4-E2B-it \
    --served-model-name gemma4-e2b \
    --dtype float16 \
    --max-model-len 4096 \
    --max-num-seqs 2 \
    --gpu-memory-utilization 0.85 \
    --tensor-parallel-size 1 \
    --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}' \
    --host 0.0.0.0 --port 8002
