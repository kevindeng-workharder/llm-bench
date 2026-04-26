#!/bin/bash
# vLLM eager mode, single 7900 XTX, Qwen3-4B (DENSE fp16). Comparison
# baseline for the graph-vs-eager speedup measurement on dense models.
exec env \
    MODEL=/data/Qwen3-4B \
    SERVED_NAME=qwen3-4b \
    TP_SIZE=1 \
    MAX_MODEL_LEN=4096 \
    DTYPE=float16 \
    GPU_MEM_UTIL=0.85 \
    COMPILATION_MODE=eager \
    VENV_PREFIX=/home/ubuntu/ai-2.10 \
    PORT=8000 \
    bash /home/ubuntu/vllm-serve/launch-server.sh
