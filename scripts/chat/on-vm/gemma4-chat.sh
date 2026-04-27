#!/bin/bash
# Launcher for gemma4-chat.py on the VM: set the required riscv+ROCm
# runtime env vars and exec python3 in the ai-2.10 venv.
set -e
export LD_LIBRARY_PATH=/home/ubuntu/ai-2.10/lib/python3.13/site-packages/torch/lib:/opt/rocm-riscv/lib
export TORCH_USE_RTLD_GLOBAL=1
export HSA_CODE_OBJECT_CACHE=1
export HIP_FORCE_DEV_KERNARG=1
export USE_LIBUV=0
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export TOKENIZERS_PARALLELISM=false    # silence HF warning in the interactive loop
exec /home/ubuntu/ai-2.10/bin/python3 /home/ubuntu/gemma4-chat.py "$@"
