#!/bin/bash
# vLLM graph mode, DUAL 7900 XTX (TP=2), Qwen3.6-27B-FP8.
# Comfortable memory budget split across both cards.
exec env \
    MODEL=/data/Qwen3.6-27B-FP8 \
    SERVED_NAME=qwen3_6-27b-fp8 \
    TP_SIZE=2 \
    MAX_MODEL_LEN=4096 \
    DTYPE=auto \
    GPU_MEM_UTIL=0.85 \
    COMPILATION_MODE=graph \
    VENV_PREFIX=/home/ubuntu/ai-2.10 \
    PORT=8000 \
    bash /home/ubuntu/vllm-serve/launch-server.sh
