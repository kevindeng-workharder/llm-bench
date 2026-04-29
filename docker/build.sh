#!/bin/bash
# Re-generate the payload tarballs from the riscv64 VM disk image, then
# `docker buildx` the inference image.
#
# Why this script exists: the heavy bits (ROCm 6.2.4 user-space, vLLM venv,
# llama.cpp binaries) live on the riscv64 VM's filesystem. To put them in
# a container image we need to:
#
#   1. Mount the VM disk image read-only (loop-back).
#   2. Tar the payload directories.
#   3. Build the image with those tarballs in the build context.
#   4. Unmount.
#
# The whole flow is idempotent — running again replaces the tarballs and
# rebuilds. Safe to skip step 1-2 if you didn't change the VM (use --no-extract).
#
# Usage:
#   ./build.sh                        # full extract + build
#   ./build.sh --no-extract           # reuse existing tarballs, just rebuild
#   ./build.sh --vm-image PATH        # use a different VM image
#   IMAGE_TAG=inference:dev ./build.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

VM_IMAGE="${VM_IMAGE:-/home/kevin/ubuntu-25.10-preinstalled-server-riscv64.img}"
IMAGE_TAG="${IMAGE_TAG:-inference:riscv64}"
MNT="${MNT:-/mnt/inference-build-mount}"
PAYLOAD_DIR="$SCRIPT_DIR/payload"
EXTRACT=1

while [ $# -gt 0 ]; do
    case "$1" in
        --no-extract) EXTRACT=0; shift ;;
        --vm-image)   VM_IMAGE="$2"; shift 2 ;;
        --tag)        IMAGE_TAG="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,30p' "$0" | sed 's/^# \?//'
            exit 0 ;;
        *) echo "unknown arg: $1"; exit 1 ;;
    esac
done

echo "==> VM_IMAGE   = $VM_IMAGE"
echo "==> IMAGE_TAG  = $IMAGE_TAG"
echo "==> PAYLOAD    = $PAYLOAD_DIR"
echo "==> EXTRACT    = $EXTRACT"

# ─────────── Extract payload tarballs ───────────
if [ "$EXTRACT" -eq 1 ]; then
    [ -f "$VM_IMAGE" ] || { echo "ERROR: VM image not found: $VM_IMAGE"; exit 1; }

    mkdir -p "$PAYLOAD_DIR"
    sudo mkdir -p "$MNT"

    echo "==> looping VM image"
    LOOP=$(sudo losetup --show -fP "$VM_IMAGE")
    echo "    loop = $LOOP"
    cleanup() {
        sudo umount "$MNT" 2>/dev/null || true
        sudo losetup -d "$LOOP" 2>/dev/null || true
    }
    trap cleanup EXIT

    echo "==> mounting ${LOOP}p1 (root partition) read-only at $MNT"
    sudo mount -o ro "${LOOP}p1" "$MNT"

    [ -d "$MNT/opt/llama" ]            || { echo "ERROR: $MNT/opt/llama missing — wrong image?"; exit 1; }
    [ -d "$MNT/opt/rocm-riscv" ]       || { echo "ERROR: $MNT/opt/rocm-riscv missing"; exit 1; }
    [ -d "$MNT/home/ubuntu/ai-2.10" ]  || { echo "ERROR: $MNT/home/ubuntu/ai-2.10 missing"; exit 1; }

    # ── sysdeps.tar.gz: replaces apt-get install. Bundles Python 3.13,
    #    system libs (libnuma/libdrm/libffi/...) and CA certs from the VM,
    #    so the build is fully offline.
    echo "==> tar python + system libs + ca-certs + gcc-15 toolchain → sysdeps.tar.gz"
    # Includes gcc/binutils + headers because Triton (used by vLLM's V1
    # attention path) JIT-compiles a HIPUtils Python extension at runtime
    # and bails out with "Failed to find C compiler" if none is on PATH.
    sudo tar -czf "$PAYLOAD_DIR/sysdeps.tar.gz" \
        --owner=root --group=root \
        --exclude='usr/lib/python3.13/__pycache__' \
        --exclude='usr/lib/python3.13/test' \
        --exclude='usr/lib/python3.13/idlelib' \
        --exclude='usr/lib/python3.13/turtledemo' \
        --exclude='usr/lib/python3.13/tkinter' \
        --exclude='usr/lib/riscv64-linux-gnu/dri' \
        --exclude='usr/lib/riscv64-linux-gnu/perl*' \
        --exclude='usr/lib/riscv64-linux-gnu/ImageMagick*' \
        --exclude='usr/lib/riscv64-linux-gnu/gconv' \
        --exclude='usr/lib/gcc/riscv64-linux-gnu/14' \
        --exclude='usr/lib/gcc/riscv64-linux-gnu/15/include-fixed' \
        --exclude='usr/lib/gcc/riscv64-linux-gnu/*/plugin' \
        --exclude='usr/libexec/gcc/riscv64-linux-gnu/14' \
        -C "$MNT" \
        usr/bin/python3.13 \
        usr/lib/python3.13 \
        usr/lib/riscv64-linux-gnu \
        etc/ssl etc/ca-certificates etc/ca-certificates.conf etc/alternatives \
        usr/share/ca-certificates \
        usr/bin/gcc usr/bin/gcc-15 usr/bin/cc usr/bin/cpp usr/bin/cpp-15 \
        usr/bin/as usr/bin/ld usr/bin/ld.bfd usr/bin/ar usr/bin/nm \
        usr/bin/riscv64-linux-gnu-gcc-15 usr/bin/riscv64-linux-gnu-cpp-15 \
        usr/bin/riscv64-linux-gnu-as usr/bin/riscv64-linux-gnu-ld usr/bin/riscv64-linux-gnu-ld.bfd \
        usr/lib/gcc \
        usr/libexec/gcc \
        usr/include
    # /etc/alternatives is required because Debian/Ubuntu uses it to manage
    # BLAS/LAPACK alternative implementations: /usr/lib/.../libblas.so.3
    # is a symlink to /etc/alternatives/libblas.so.3-* which then points at
    # the actual implementation under /usr/lib/.../blas/. Without it numpy
    # fails at "ImportError: libblas.so.3: cannot open shared object file".
    sudo chown "$USER:$USER" "$PAYLOAD_DIR/sysdeps.tar.gz"

    echo "==> tar /opt/llama → opt-llama.tar.gz"
    sudo tar -czf "$PAYLOAD_DIR/opt-llama.tar.gz" \
        --owner=root --group=root \
        -C "$MNT" opt/llama
    sudo chown "$USER:$USER" "$PAYLOAD_DIR/opt-llama.tar.gz"

    echo "==> tar /opt/rocm-riscv → opt-rocm-riscv.tar.gz (excluding include.bak)"
    sudo tar -czf "$PAYLOAD_DIR/opt-rocm-riscv.tar.gz" \
        --owner=root --group=root \
        --exclude='opt/rocm-riscv/include.bak' \
        -C "$MNT" opt/rocm-riscv
    sudo chown "$USER:$USER" "$PAYLOAD_DIR/opt-rocm-riscv.tar.gz"

    echo "==> tar /home/ubuntu/ai-2.10 → home-ubuntu-ai-2.10.tar.gz"
    sudo tar -czf "$PAYLOAD_DIR/home-ubuntu-ai-2.10.tar.gz" \
        --owner=root --group=root \
        --exclude='home/ubuntu/ai-2.10/share/jupyter' \
        --exclude='home/ubuntu/ai-2.10/share/man' \
        -C "$MNT" home/ubuntu/ai-2.10
    sudo chown "$USER:$USER" "$PAYLOAD_DIR/home-ubuntu-ai-2.10.tar.gz"

    echo "==> payload tarballs:"
    ls -lh "$PAYLOAD_DIR"/*.tar.gz
fi

# ─────────── buildx build (riscv64 via qemu-user-static binfmt) ───────────
echo "==> docker buildx build --platform=linux/riscv64 -t $IMAGE_TAG"
docker buildx build \
    --platform=linux/riscv64 \
    --tag "$IMAGE_TAG" \
    --load \
    "$SCRIPT_DIR"

echo
echo "==> done. Image:"
docker images "$IMAGE_TAG"
echo
echo "Try it:"
echo "  docker run --rm $IMAGE_TAG --help"
echo "  docker compose up -d   # bring up qwen-llamacpp + gemma-vllm together"
