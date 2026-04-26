#!/bin/bash
# vLLM graph mode, single 7900 XTX, Qwen3.6-27B-FP8 (DENSE fp8 quant).
# 27B at FP8 is ~27GB — tight on 24GB; will likely need GPU_MEM_UTIL low
# or fall back to TP=2.  Run this first; if it OOMs, use the tp2 variant.
exec env \
    MODEL=/data/Qwen3.6-27B-FP8 \
    SERVED_NAME=qwen3_6-27b-fp8 \
    TP_SIZE=1 \
    MAX_MODEL_LEN=2048 \
    DTYPE=auto \
    GPU_MEM_UTIL=0.92 \
    COMPILATION_MODE=graph \
    VENV_PREFIX=/home/ubuntu/ai-2.10 \
    PORT=8000 \
    bash /home/ubuntu/vllm-serve/launch-server.sh
