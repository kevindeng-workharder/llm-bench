#!/bin/bash
# llama.cpp + Qwen3.6-35B-A3B MXFP4_MOE on card0 (7900 XTX, 24 GB).
#
# Variant of qwen36-35b-mxfp4-tp1.sh tuned for the dual-card setup where
# this server shares the host with another (vLLM) server on card1.
#   --parallel 2   matches the user's "支持双并发" requirement
#   -c 8192        4k per slot (parallel=2 → n_ctx_seq = c/parallel = 4096)
#   port 8001      so the vLLM server on card1 can take port 8002 / 8000
#   HIP_VISIBLE_DEVICES=0   bind to card0 only (keeps card1 free for vLLM)
set -e
export LD_LIBRARY_PATH=/opt/llama/lib:/opt/rocm/lib:${LD_LIBRARY_PATH:-}
export HIP_VISIBLE_DEVICES=0
export ROCR_VISIBLE_DEVICES=0
exec /opt/llama/bin/llama-server \
    -m /data/Qwen3.6-35B-A3B-MXFP4_MOE.gguf \
    --alias qwen36-a3b \
    -ngl 99 \
    -mg 0 --split-mode none \
    -c 8192 \
    --parallel 2 \
    --cont-batching \
    --flash-attn on \
    --host 0.0.0.0 --port 8001 \
    --threads 8 \
    --no-mmap \
    --jinja \
    --cache-ram 0 \
    --temp 0.7 --top-p 0.95 --top-k 40 --min-p 0.0 \
    --repeat-penalty 1.05
