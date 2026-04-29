# syntax=docker/dockerfile:1.7
#
# Inference container — riscv64 build of llama.cpp (Qwen) + vLLM (Gemma)
# with ROCm 6.2.4 userspace, pinned to gfx1100 (RX 7900 series).
#
# Build:
#   cd /home/kevin/llm-bench/docker
#   docker buildx build --platform=linux/riscv64 -t inference:riscv64 \
#       --load .
#
# Run (env-driven, see entrypoint.sh for full doc):
#   docker run -d --device=/dev/kfd --device=/dev/dri \
#     -v /data/models:/data -p 8001:8001 \
#     -e ENGINE=llamacpp -e MODEL=/data/Qwen3.6-35B-A3B-MXFP4_MOE.gguf \
#     -e ALIAS=qwen36-a3b -e GPU=0 -e PORT=8001 \
#     inference:riscv64
#
# Models live OUTSIDE the image — mount /data as a volume.

FROM --platform=linux/riscv64 ubuntu:25.10

# ── System deps bundled offline from the VM image (NO apt download) ──
#
# This tarball contains exactly what the old Dockerfile would have apt-installed:
#   /usr/bin/python3.13                           — Python 3.13.7 interpreter
#   /usr/lib/python3.13/                          — stdlib (test/idlelib/tkinter excluded)
#   /usr/lib/riscv64-linux-gnu/                   — system libs (libnuma/libdrm/libffi/...)
#   /etc/ssl + /etc/ca-certificates*              — TLS roots
#
# libomp.so itself comes from the rocm-riscv layer below
# (/opt/rocm-riscv/lib/libomp.so) — it's already in LD_LIBRARY_PATH.
COPY payload/sysdeps.tar.gz /tmp/
RUN tar xzf /tmp/sysdeps.tar.gz -C / && \
    rm /tmp/sysdeps.tar.gz && \
    ln -sf /usr/bin/python3.13 /usr/bin/python3 && \
    ldconfig

# ── Layer 1: ROCm 6.2.4 riscv64 (~1.9 GB, smallest delta over time) ──
COPY payload/opt-rocm-riscv.tar.gz /tmp/
RUN tar xzf /tmp/opt-rocm-riscv.tar.gz -C / && \
    rm /tmp/opt-rocm-riscv.tar.gz && \
    ln -sf /opt/rocm-riscv /opt/rocm && \
    ldconfig

# ── Layer 2: vLLM venv 0.19.0+rocm624 with v2 NaN clamp fix already applied
#             plus torch 2.8.0a0+riscv64.rocm and triton 3.4.0+gitc817b9b6 ──
COPY payload/home-ubuntu-ai-2.10.tar.gz /tmp/
RUN tar xzf /tmp/home-ubuntu-ai-2.10.tar.gz -C / && \
    rm /tmp/home-ubuntu-ai-2.10.tar.gz

# ── Layer 3: llama.cpp riscv64 build b358 (server + libs in /opt/llama) ──
# Smallest layer last so iterating on llama.cpp builds doesn't bust the cache
# of the heavier vLLM layer above.
COPY payload/opt-llama.tar.gz /tmp/
RUN tar xzf /tmp/opt-llama.tar.gz -C / && \
    rm /tmp/opt-llama.tar.gz

# ── Reference launcher scripts (kept as documentation; entrypoint takes
#    precedence) ──
COPY launchers/ /opt/launchers/
RUN chmod +x /opt/launchers/*.sh

# ── Entrypoint dispatcher ──
COPY entrypoint.sh /usr/local/bin/inference-entrypoint
RUN chmod +x /usr/local/bin/inference-entrypoint

# ── Env (ordered by likely override frequency) ──
ENV PYTHONPATH=/home/ubuntu/ai-2.10/lib/python3.13/site-packages
# IMPORTANT: torch's lib MUST come before /opt/rocm-riscv/lib in LD_LIBRARY_PATH.
# /opt/rocm-riscv/lib contains a stale copy of libc10.so / libtorch.so from
# the cross-compile build; if it wins the search order, libtorch_cpu.so fails
# to find newer symbols (e.g. c10::TensorImpl::incref_pyobject) and import
# torch dies with ImportError.
ENV LD_LIBRARY_PATH=/home/ubuntu/ai-2.10/lib/python3.13/site-packages/torch/lib:/opt/rocm-riscv/lib:/opt/llama/lib
ENV PATH=/home/ubuntu/ai-2.10/bin:/opt/llama/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# ROCm runtime tunables — copied from old Dockerfile.
# These are the settings verified working for gfx1100 in the riscv VM.
ENV HSA_CODE_OBJECT_CACHE=1
ENV MIOPEN_DISABLE_CACHE=1
ENV HIP_FORCE_DEV_KERNARG=1
ENV GPU_MAX_HW_QUEUES=4
ENV PYTORCH_NO_CUDA_MEMORY_CACHING=0
ENV TORCH_CUDNN_V8_API_DISABLED=1
ENV TOKENIZERS_PARALLELISM=false

WORKDIR /workspace
EXPOSE 8000
# CMD intentionally NOT set: when --help is the default CMD it ends up as $1
# and shadows ENGINE env var (the case-match in entrypoint.sh assigns
# ENGINE="--help" before the env-defaulting logic runs). With no CMD, $@ is
# empty for env-driven invocations, so the env value wins. The script's
# `${ENGINE:---help}` fallback still shows help when nothing is set at all.
ENTRYPOINT ["/usr/local/bin/inference-entrypoint"]
