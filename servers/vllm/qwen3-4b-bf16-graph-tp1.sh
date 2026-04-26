#!/bin/bash
# Probe: vLLM 0.19 Qwen3-4B with bfloat16 (instead of fp16) to test if the
# NaN at rows 1,3 of hidden_states is fp16 overflow or a kernel bug.
exec env \
    MODEL=/data/Qwen3-4B \
    SERVED_NAME=qwen3-4b \
    TP_SIZE=1 \
    MAX_MODEL_LEN=4096 \
    DTYPE=bfloat16 \
    GPU_MEM_UTIL=0.85 \
    COMPILATION_MODE=graph \
    VENV_PREFIX=/home/ubuntu/ai-2.10 \
    PORT=8000 \
    bash /home/ubuntu/vllm-serve/launch-server.sh
