#!/bin/bash
# vLLM graph mode, DUAL 7900 XTX (TP=2), Qwen3.6-27B Quark W8A8-INT8 — MULTIMODAL
# (image + video) at 40K context, 2-way concurrency.
#
# Same model/quant as qwen3_6-27b-quark-int8-graph-tp2.sh (text-only); this one
# turns the qwen3_5 VL checkpoint's vision path ON. Four flags do it:
#   1. --limit-mm-per-prompt image:1,video:1     (vs 0/0 text-only)
#   2. --mm-processor-kwargs max_pixels=200704   (448px cap; bounds ViT work and
#      the per-image pre-merge patch count N — see docs/qwen3_6-27b-multimodal.md)
#   3. --media-io-kwargs video.backend=pyav      (opencv video loader is unfetchable
#      on the riscv apt mirror; pyav decodes via av.open — run scripts/multimodal-deps-apt.sh)
#   4. --mm-encoder-attn-backend TRITON_ATTN     (ViT attention O(N); the rocm default
#      falls to TORCH_SDPA = O(N^2) and OOMs at HD. THE load-bearing fix — see doc.)
#
# max-model-len 40960 + max-num-seqs 2: with the vision tower loaded the KV pool is
# ~96K tokens, so 2 x 40960 = 81920 fits (Maximum concurrency ~2.36x). cudagraph [1,2].
#
# Runs on the writable rootfs venv /home/ubuntu/vllm-venv (gemv INT8 patch baked in;
# the system `av` symlinked into site-packages) — NOT /data/ai-2.11 like the other
# launchers. See docs/qwen3_6-27b-multimodal.md ("Environment") for venv contents.
set -eu
source /home/ubuntu/vllm-serve-env.sh
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
export CC=/opt/rocm/llvm/bin/clang
export VLLM_NCCL_SO_PATH=/home/ubuntu/librccl-rebuilt.so.1.0
export LD_PRELOAD=/home/ubuntu/librccl-rebuilt.so.1.0
export VLLM_RPC_TIMEOUT=7200000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=7200
exec /home/ubuntu/vllm-venv/bin/python -m vllm.entrypoints.openai.api_server \
    --model /data/Qwen3.6-27B-Quark-W8A8-INT8 \
    --served-model-name qwen3_6-27b-int8 \
    --quantization quark --dtype bfloat16 \
    --max-model-len 40960 --max-num-seqs 2 --max-num-batched-tokens 16384 \
    --gpu-memory-utilization 0.85 \
    --tensor-parallel-size 2 --distributed-executor-backend mp \
    --trust-remote-code --no-enable-prefix-caching \
    --limit-mm-per-prompt '{"image":1,"video":1}' \
    --mm-processor-kwargs '{"max_pixels":200704}' \
    --media-io-kwargs '{"video":{"backend":"pyav"}}' \
    --mm-encoder-attn-backend TRITON_ATTN \
    --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2],"max_cudagraph_capture_size":2,"cudagraph_num_of_warmups":0}' \
    --host 0.0.0.0 --port 8000
