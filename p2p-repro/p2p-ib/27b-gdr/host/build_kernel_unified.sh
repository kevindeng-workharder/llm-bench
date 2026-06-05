#!/bin/bash
# UNIFIED 6.19.5 kernel: P2P hacks + full IB/mlx5 + amdgpu.
# Base config = the PROVEN -p2p-ib config (PCI_P2PDMA+HSA_AMD_P2P+ZONE_DEVICE+IB, already boots beta).
# Source     = patched qemu_soc tree (cpu_supports_p2pdma + amdgpu is_large_bar + kfd hacks).
# Toolchain  = gcc-15 (aligned with qemu_soc/kernel/build_kernel.sh).
set -e
SELF="$(cd "$(dirname "$0")" && pwd)"
# KSRC = the Beta-SoC kernel tree (kernel.org linux-6.19.5 + qemu_soc beta-SoC patches).
# The full tree is NOT in this repo; point KSRC at your checkout (override via env).
KSRC="${KSRC:-/home/ubuntu/qemu_soc/kernel/linux-6.19.5}"
# PATCH + BASE_CONFIG are committed alongside this script (repo-local defaults).
PATCH="${PATCH:-$SELF/kernel-6.19.5-p2p.patch}"                  # the P2P kernel patch
BASE_CONFIG="${BASE_CONFIG:-$SELF/kernel-6.19.5-p2p-ib.config}"  # proven -p2p-ib config base
OUTDIR=/home/ubuntu/p2p_build/kernel
mkdir -p "$OUTDIR"

export ARCH=riscv
export CROSS_COMPILE=riscv64-linux-gnu-
TMPBIN=$(mktemp -d)
ln -sf /usr/bin/riscv64-linux-gnu-gcc-15 "$TMPBIN/riscv64-linux-gnu-gcc"
ln -sf /usr/bin/riscv64-linux-gnu-g++-15 "$TMPBIN/riscv64-linux-gnu-g++" 2>/dev/null || true
for t in ar as ld nm objcopy objdump ranlib readelf strip; do
    src=$(command -v riscv64-linux-gnu-$t 2>/dev/null || true)
    [ -n "$src" ] && ln -sf "$src" "$TMPBIN/riscv64-linux-gnu-$t"
done
export PATH="$TMPBIN:$PATH"
echo "=== gcc ==="; ${CROSS_COMPILE}gcc --version | head -1
cd "$KSRC"

echo "=== mrproper ==="; make mrproper 2>&1 | tail -2
echo "=== apply p2p patch (idempotent; tree already has it) ==="
if patch -p1 --dry-run --silent < "$PATCH" 2>/dev/null; then patch -p1 < "$PATCH"; echo "applied"; else echo "already-applied/forward"; patch -p1 --forward < "$PATCH" 2>&1 | tail -4 || true; fi
echo "=== GATE: cpu_supports_p2pdma hack present in source? ==="
sed -n '/static bool cpu_supports_p2pdma/,/^}/p' drivers/pci/p2pdma.c | grep -q 'return true' \
  && echo "  HACK PRESENT" || { echo "  FATAL: p2pdma hack missing"; exit 1; }

echo "=== config: PROVEN -p2p-ib base + LOCALVERSION=-p2p-all + force P2P keys ==="
cp "$BASE_CONFIG" .config
./scripts/config --set-str CONFIG_LOCALVERSION "-p2p-all"
./scripts/config --disable LOCALVERSION_AUTO
./scripts/config --enable PCI_P2PDMA --enable HSA_AMD_P2P --enable INFINIBAND_ON_DEMAND_PAGING
make olddefconfig 2>&1 | tail -3

echo "=== GATE: verify critical config survived olddefconfig ==="
miss=0
for k in CONFIG_PCI_P2PDMA=y CONFIG_HSA_AMD_P2P=y CONFIG_DMABUF_MOVE_NOTIFY=y \
         CONFIG_INFINIBAND=m CONFIG_MLX5_INFINIBAND=m CONFIG_MLX5_CORE=m \
         CONFIG_INFINIBAND_ON_DEMAND_PAGING=y CONFIG_INFINIBAND_USER_ACCESS=m \
         CONFIG_INFINIBAND_USER_MEM=y CONFIG_MLX5_CORE_EN=y CONFIG_DRM_AMDGPU=m \
         CONFIG_ZONE_DEVICE=y CONFIG_HSA_AMD=y; do
  if grep -qx "$k" .config; then echo "  ok   $k"; else echo "  MISS $k"; miss=1; fi
done
[ "$miss" = 0 ] || { echo "FATAL: critical config missing — aborting before the long build"; exit 2; }

echo "=== make Image + modules (5-15 min) ==="
make -j"$(nproc)" Image modules 2>&1 | tail -8
echo "=== modules_install -> staging ==="
rm -rf "$OUTDIR/mod_staging"
make INSTALL_MOD_PATH="$OUTDIR/mod_staging" modules_install 2>&1 | tail -3

echo "=== output Image-6.19.5-p2p-all ==="
cp arch/riscv/boot/Image "$OUTDIR/Image-6.19.5-p2p-all"
cp arch/riscv/boot/Image /home/ubuntu/p2p_archive/artifacts/Image-6.19.5-p2p-all
ls -lh "$OUTDIR/Image-6.19.5-p2p-all"
echo "=== staged modules dir + key .ko present? ==="
ls "$OUTDIR/mod_staging/lib/modules/"
find "$OUTDIR/mod_staging" \( -name 'mlx5_ib.ko*' -o -name 'mlx5_core.ko*' -o -name 'ib_core.ko*' -o -name 'amdgpu.ko*' \) | sed "s#$OUTDIR/mod_staging##"
rm -rf "$TMPBIN"
echo "=== UNIFIED BUILD COMPLETE ==="