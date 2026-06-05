#!/bin/bash
# Run as ROOT (echo PW | sudo -S bash install_unified_modules.sh).
# Install unified-kernel modules into BOTH guest rootfs images via offline loop-mount.
# Run AFTER build_kernel_unified.sh (needs mod_staging).
set -e
STAGING=/home/ubuntu/p2p_build/kernel/mod_staging
MODVER=$(ls "$STAGING/lib/modules/" 2>/dev/null | head -1)
[ -n "$MODVER" ] || { echo "FATAL: no staged modules at $STAGING — build first"; exit 1; }
echo "=== installing kernel modules version: $MODVER ==="
MNT=/mnt/guestroot; mkdir -p "$MNT"
for IMG in /home/ubuntu/kevin/ubuntu-25.10-preinstalled-server-riscv64.img /home/ubuntu/kevin/ubuntu-vm2.img; do
  echo "=== $IMG ==="
  LOOP=$(losetup -fP --show "$IMG")
  mount "${LOOP}p1" "$MNT"
  [ -d "$MNT/lib/modules" ] || { echo "  p1 not rootfs, trying p2"; umount "$MNT"; mount "${LOOP}p2" "$MNT"; }
  echo "  existing module dirs: $(ls "$MNT/lib/modules/" 2>/dev/null | tr '\n' ' ')"
  rsync -a "$STAGING/lib/modules/$MODVER/" "$MNT/lib/modules/$MODVER/"
  depmod -b "$MNT" "$MODVER" 2>&1 | tail -1 || true
  echo "  mlx5_ib: $(ls "$MNT/lib/modules/$MODVER/kernel/drivers/infiniband/hw/mlx5/mlx5_ib.ko"* 2>/dev/null)"
  echo "  amdgpu : $(ls "$MNT/lib/modules/$MODVER/kernel/drivers/gpu/drm/amd/amdgpu/amdgpu.ko"* 2>/dev/null)"
  sync; umount "$MNT"; losetup -d "$LOOP"
  echo "  done $IMG"
done
echo "=== MODULE INSTALL COMPLETE ($MODVER) ==="
