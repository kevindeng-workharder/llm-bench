#!/bin/bash
# vLLM graph mode, single 7900 XTX, Qwen3-0.6B (DENSE fp16, smallest test).
# Thin wrapper around the canonical /home/ubuntu/vllm-serve/launch-server.sh.
exec env \
    MODEL=/data/Qwen3-0.6B \
    SERVED_NAME=qwen3-0_6b \
    TP_SIZE=1 \
    MAX_MODEL_LEN=4096 \
    DTYPE=float16 \
    GPU_MEM_UTIL=0.85 \
    COMPILATION_MODE=graph \
    VENV_PREFIX=/home/ubuntu/ai-2.10 \
    PORT=8000 \
    bash /home/ubuntu/vllm-serve/launch-server.sh
