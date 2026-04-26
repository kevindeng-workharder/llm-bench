#!/bin/bash
# llama.cpp + Qwen3.6-35B-A3B MXFP4_MOE on single 7900 XTX.
# Uses our riscv64 cross-compiled build with the MMQ kernel disabled
# (FORCE_CUBLAS) and AMD_WMMA_AVAILABLE gated off (RISCV_HIP_NO_WMMA).
# Verified correct at all N (no garbage output).
set -e
export LD_LIBRARY_PATH=/opt/llama/lib:/opt/rocm/lib:${LD_LIBRARY_PATH:-}
export HIP_VISIBLE_DEVICES=0
export ROCR_VISIBLE_DEVICES=0
exec /opt/llama/bin/llama-server \
    -m /data/Qwen3.6-35B-A3B-MXFP4_MOE.gguf \
    --alias qwen36-a3b \
    -ngl 99 \
    -mg 0 --split-mode none \
    -c 16384 \
    --parallel 8 \
    --cont-batching \
    --host 0.0.0.0 --port 8000 \
    --threads 8 \
    --no-mmap \
    --chat-template chatml \
    --cache-ram 0 \
    --temp 0.7 --top-p 0.95 --top-k 40 --min-p 0.0 \
    --repeat-penalty 1.05
