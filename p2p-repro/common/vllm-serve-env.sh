# Canonical runtime env for running vLLM on riscv64 + ROCm gfx1100.
# Source this from your launcher scripts. Works for both the in-process
# `LLM()` class and the `vllm.entrypoints.openai.api_server` module.
#
# Override before sourcing to point at a different ROCm prefix / venv:
#   ROCM_PREFIX=...  VENV_PREFIX=...  source server-env.sh
#
# 2026-05-12: default updated to native ROCm 7.2.3 + PyTorch 2.11 + patched
# RCCL 2.27 (rocvirtual.cpp HiddenHostcallBuffer + xml.cc __riscv patches —
# see perf-data/TP2_ROOT_CAUSE.md). TP=2 verified 131 tok/s @ N=32 graph
# mode on Qwen3-4B, matching the previous 6.2.4-shim path without needing
# the shim. For the legacy 6.2.4 stack override before sourcing:
#   ROCM_PREFIX=/opt/rocm-riscv  VENV_PREFIX=/home/ubuntu/ai-2.10  source server-env.sh
#
# DEPLOYED LOCATION on each guest: /home/ubuntu/vllm-serve-env.sh
# This is a verbatim copy kept under version control. If you change the guest
# copy, update this one too (and vice-versa).
ROCM_PREFIX="${ROCM_PREFIX:-/opt/rocm-riscv-7.2.3}"
VENV_PREFIX="${VENV_PREFIX:-/data/vllm0.21-pt2.11}"
PYTHON_MINOR="${PYTHON_MINOR:-3.13}"

# Runtime loader.  The venv's torch/lib MUST come FIRST — its
# libamdhip64.so / librccl.so are symlinks into ${ROCM_PREFIX}/lib, so
# venv-first is consistent with both the 7.2.3 (default) and 6.2.4 stacks.
export LD_LIBRARY_PATH="${VENV_PREFIX}/lib/python${PYTHON_MINOR}/site-packages/torch/lib:${ROCM_PREFIX}/lib"

# ROCm / HIP tunings — required
export TORCH_USE_RTLD_GLOBAL=1       # symbol visibility across shared libs
export HSA_CODE_OBJECT_CACHE=1       # cache compiled kernels (skip recompile)
export HIP_FORCE_DEV_KERNARG=1       # kernel args pass via device memory (stable on riscv)

# CRITICAL for P2P (2026-05-23 fix): without this RCCL P2P AllReduce kernel
# deadlocks because cross-GPU atomic signal writes go to coarse-grain GPU
# memory and are not visible to the peer GPU's polling kernel. Fine-grain PCIe
# routes atomics through PCIe (bypasses GPU L2 cache) so peer sees them
# immediately. RCCL itself warns at init: "Missing HSA_FORCE_FINE_GRAIN_PCIE=1
# ... can lead to ... hang".
export HSA_FORCE_FINE_GRAIN_PCIE=1
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
# 2026-05-12: with the default ROCM_PREFIX=/opt/rocm-riscv-7.2.3, this points
# at the patched librccl.so.1.0 (RCCL 2.27.7-HEAD:96a25b5+, Bug 3+4 fixes).
# On the 6.2.4 fallback path this falls back to 6.2.4 RCCL (NCCL 2.20).
export VLLM_NCCL_SO_PATH="${ROCM_PREFIX}/lib/librccl.so.1"

# Disable NCCL transports that don't work in riscv-QEMU:
# - P2P: VFIO doesn't expose P2P DMA between GPU BARs in guest
# - SHM: shared memory across workers is serialized through QEMU
# - IB:  no InfiniBand
# Fall back to Socket (TCP over loopback) which works but is the slowest path.
# export NCCL_P2P_DISABLE=1  # P2P enabled Phase 3+4
# export NCCL_SHM_DISABLE=1
export NCCL_IB_DISABLE=1
export RCCL_MSCCL_ENABLE=0
export NCCL_IGNORE_CPU_AFFINITY=1

# RCCL topology autodiscovery reads /sys/.../arch on x86 but fails on
# riscv64. Point at a hand-crafted topo XML (version 2 required).
# See ./rccl-topo.xml.
export NCCL_TOPO_FILE="${NCCL_TOPO_FILE:-$(dirname "${BASH_SOURCE[0]}")/rccl-topo.xml}"

# Logging — INFO first time to verify RCCL init, switch to WARN for noise-free bench
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

# NOTE on the three P2P scenarios (see ../README.md):
#   - p2p-direct (single-VM, Infinity Fabric):  P2P + SHM enabled, IB disabled
#   - p2p-shm    (single-VM, host SHM):          NCCL_TOPO_FILE=rccl-topo-split.xml
#   - p2p-ib     (cross-VM, RoCE):               unset NCCL_TOPO_FILE; NCCL_IB_DISABLE=0,
#                                                NCCL_NET=IB, NCCL_IB_HCA=roceP3p1s0
# Each scenario's launcher applies its own overrides AFTER sourcing this file.
