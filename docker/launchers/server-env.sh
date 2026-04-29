# Canonical runtime env for running vLLM on riscv64 + ROCm gfx1100.
# Source this from your launcher scripts. Works for both the in-process
# `LLM()` class and the `vllm.entrypoints.openai.api_server` module.
#
# Override before sourcing to point at a different ROCm prefix / venv:
#   ROCM_PREFIX=...  VENV_PREFIX=...  source server-env.sh
#
ROCM_PREFIX="${ROCM_PREFIX:-/opt/rocm-riscv}"
# Default to the ai-2.10 venv (pytorch 2.10 + vllm 0.19 + real triton 3.4).
# For the 0.11 baseline, override with VENV_PREFIX=/home/ubuntu/ai.
VENV_PREFIX="${VENV_PREFIX:-/home/ubuntu/ai-2.10}"
PYTHON_MINOR="${PYTHON_MINOR:-3.13}"

# Runtime loader.  For vLLM 0.19 + torch 2.10 against a non-default venv
# (e.g. /home/ubuntu/ai-2.10), the venv's torch/lib MUST come FIRST —
# /opt/rocm-riscv/lib/libtorch_*.so are symlinks into the DEFAULT venv
# (ai/, 2.8), so ROCm-first order would load 2.8 libs against 2.10
# headers.  The venv's torch/lib has libamdhip64.so as a symlink back
# to the real /opt/rocm-riscv/lib/libamdhip64.so.6.* (install-runtime-
# stubs.sh), so venv-first is also safe for the 2.8 venv.
export LD_LIBRARY_PATH="${VENV_PREFIX}/lib/python${PYTHON_MINOR}/site-packages/torch/lib:${ROCM_PREFIX}/lib"

# ROCm / HIP tunings — required
export TORCH_USE_RTLD_GLOBAL=1       # symbol visibility across shared libs
export HSA_CODE_OBJECT_CACHE=1       # cache compiled kernels (skip recompile)
export HIP_FORCE_DEV_KERNARG=1       # kernel args pass via device memory (stable on riscv)
export USE_LIBUV=0                   # pytorch distributed: fall back to TCP, libuv has build issues

# vLLM V1 attention backend selection — rocm_attn uses C++ paged_attention_rocm
# which avoids triton for decode; only prefill path needs the pure-torch
# SDPA replacement in prefix_prefill.py.
export VLLM_V1_USE_PREFILL_DECODE_ATTENTION=1
export VLLM_ROCM_CUSTOM_PAGED_ATTN=1

# Multiprocess worker launch mode (required for TP>1 without ray)
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Skip torch.compile / inductor. With real triton 3.4 on the 0.19 path
# this is technically unnecessary when we pass compilation_config mode=NONE
# (which skips torch.compile anyway), but setting it here is a defensive
# backstop in case some code path bypasses the config. Unset for perf
# experiments that want inductor on.
export TORCH_COMPILE_DISABLE=1

# Force single-process engine (not a worker subprocess). Required by both
# the `--enforce-eager` debug path and the graph path under our runtime.
# Without this, vllm spawns an EngineCore subprocess whose stderr gets
# swallowed — broken errors look like "Engine core initialization failed".
export VLLM_ENABLE_V1_MULTIPROCESSING="${VLLM_ENABLE_V1_MULTIPROCESSING:-0}"

# --- TP>1 only (RCCL) ---
# vLLM looks for libnccl.so.2; we symlink it to librccl.so.1 in
# install-runtime-stubs.sh, but also export VLLM_NCCL_SO_PATH for safety.
export VLLM_NCCL_SO_PATH="${ROCM_PREFIX}/lib/librccl.so.1"

# Disable NCCL transports that don't work in riscv-QEMU:
# - P2P: VFIO doesn't expose P2P DMA between GPU BARs in guest
# - SHM: shared memory across workers is serialized through QEMU
# - IB:  no InfiniBand
# Fall back to Socket (TCP over loopback) which works but is the slowest path.
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_IB_DISABLE=1
export RCCL_MSCCL_ENABLE=0
export NCCL_IGNORE_CPU_AFFINITY=1

# RCCL topology autodiscovery reads /sys/.../arch on x86 but fails on
# riscv64. Point at a hand-crafted topo XML (version 2 required).
# See ./rccl-topo.xml.
export NCCL_TOPO_FILE="${NCCL_TOPO_FILE:-$(dirname "${BASH_SOURCE[0]}")/rccl-topo.xml}"

# Logging — INFO first time to verify RCCL init, switch to WARN for noise-free bench
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
