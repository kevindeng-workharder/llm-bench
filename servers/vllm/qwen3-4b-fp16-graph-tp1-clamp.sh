#!/bin/bash
# Same as qwen3-4b-fp16-graph-tp1, but assumes the triton-attention
# clamp patch (scripts/instruments/patch-all-triton-clamp.py) has been
# applied to the running venv. The patch clamps `acc` to fp16 range
# in every triton attention kernel epilogue (prefix_prefill,
# triton_prefill_attention, triton_decode_attention,
# kernel_paged_attention_2d) before tl.store, sidestepping the
# fp16 overflow that becomes inf when cast to fp16 output.
exec env \
    MODEL=/data/Qwen3-4B \
    SERVED_NAME=qwen3-4b \
    TP_SIZE=1 \
    MAX_MODEL_LEN=4096 \
    DTYPE=float16 \
    GPU_MEM_UTIL=0.85 \
    COMPILATION_MODE=graph \
    VENV_PREFIX=/home/ubuntu/ai-2.10 \
    PORT=8000 \
    bash /home/ubuntu/vllm-serve/launch-server.sh
