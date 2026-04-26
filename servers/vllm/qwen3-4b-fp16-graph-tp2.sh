#!/bin/bash
# vLLM graph mode, DUAL 7900 XTX (TP=2), Qwen3-4B (DENSE fp16). Tests RCCL
# all-reduce + cross-GPU communication path on this riscv64+ROCm stack.
exec env \
    MODEL=/data/Qwen3-4B \
    SERVED_NAME=qwen3-4b \
    TP_SIZE=2 \
    MAX_MODEL_LEN=4096 \
    DTYPE=float16 \
    GPU_MEM_UTIL=0.85 \
    COMPILATION_MODE=graph \
    VENV_PREFIX=/home/ubuntu/ai-2.10 \
    PORT=8000 \
    bash /home/ubuntu/vllm-serve/launch-server.sh
