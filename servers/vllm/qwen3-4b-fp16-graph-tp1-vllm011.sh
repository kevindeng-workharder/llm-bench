#!/bin/bash
# Control test: Qwen3-4B graph mode on the LEGACY vLLM 0.11 venv. The newer
# /home/ubuntu/ai-2.10 venv (vLLM 0.19 + torch 2.10 + real triton 3.4) is
# what the rest of the matrix uses. The /home/ubuntu/ai venv (vLLM 0.11 +
# torch 2.8) was the previously-verified configuration.
#
# Hypothesis: vLLM 0.19 introduced a batched-correctness regression on this
# stack. If 0.11 produces 0/N garbage at N=4/8 while 0.19 produces 2/4 and
# 5/8, the regression is confirmed.
#
# Note: 0.11's launch-server.sh is the same script — it just dispatches to
# the older python interpreter via VENV_PREFIX. The compilation-config
# `mode:0` flag worked in 0.11 too.
exec env \
    MODEL=/data/Qwen3-4B \
    SERVED_NAME=qwen3-4b \
    TP_SIZE=1 \
    MAX_MODEL_LEN=4096 \
    DTYPE=float16 \
    GPU_MEM_UTIL=0.85 \
    COMPILATION_MODE=graph \
    VENV_PREFIX=/home/ubuntu/ai \
    PORT=8000 \
    bash /home/ubuntu/vllm-serve/launch-server.sh
