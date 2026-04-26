#!/bin/bash
# vLLM graph mode, DUAL 7900 XTX (TP=2), Qwen2.5-14B-Instruct (DENSE fp16).
# 14B fp16 is ~28GB and will not fit on a single 24GB card; TP=2 splits
# weights across both GPUs. MEMORY.md notes a previous run achieved
# ~28 tok/s aggregate at this config.
exec env \
    MODEL=/data/Qwen2.5-14B-Instruct \
    SERVED_NAME=qwen2_5-14b \
    TP_SIZE=2 \
    MAX_MODEL_LEN=4096 \
    DTYPE=float16 \
    GPU_MEM_UTIL=0.85 \
    COMPILATION_MODE=graph \
    VENV_PREFIX=/home/ubuntu/ai-2.10 \
    PORT=8000 \
    bash /home/ubuntu/vllm-serve/launch-server.sh
