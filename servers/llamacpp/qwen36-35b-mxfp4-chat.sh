#!/bin/bash
# llama.cpp + Qwen3.6-35B-A3B MXFP4_MOE on single 7900 XTX, optimised for
# **deep-context single-session chat** (NOT batched bench).
#
# vs the bench launcher (qwen36-35b-mxfp4-tp1.sh):
#   --parallel 1     — give all the KV pool to one chat session
#   -c 65536         — 64k context window (vs 16k @ parallel=8 → 2k/slot)
#   --flash-attn on  — explicit (auto would also enable it; this is belt+braces)
#
# Memory math (Apr 2026):
#   model weights:   20.2 GB
#   KV @ 64k single: ~1.3 GB (10 full-attn layers × 65536 cells × K+V × 256 × fp16)
#   compute buffer:  ~500 MB
#   total:           ~22 GB on a 24 GB card → comfortable
set -e
export LD_LIBRARY_PATH=/opt/llama/lib:/opt/rocm/lib:${LD_LIBRARY_PATH:-}
export HIP_VISIBLE_DEVICES=0
export ROCR_VISIBLE_DEVICES=0
exec /opt/llama/bin/llama-server \
    -m /data/Qwen3.6-35B-A3B-MXFP4_MOE.gguf \
    --alias qwen36-a3b \
    -ngl 99 \
    -mg 0 --split-mode none \
    -c 65536 \
    --parallel 1 \
    --cont-batching \
    --flash-attn on \
    --host 0.0.0.0 --port 8000 \
    --threads 8 \
    --no-mmap \
    --chat-template chatml \
    --cache-ram 0 \
    --temp 0.7 --top-p 0.95 --top-k 40 --min-p 0.0 \
    --repeat-penalty 1.05
