#!/bin/bash
# IB VM1 (leader) on the UNIFIED kernel (p2p hack + IB stack). serial->file for detached run.
# = start_vm1_64g.sh but -kernel Image-6.19.5-p2p-all, -serial file, -monitor none.
set -eu
cd /home/ubuntu
# NOTE(2026-06-13): MUST use the de-bypass qemu (qemu_p2p_fresh) that writes the
# `riscv,p2pdma-capable` DT prop; the old p2p_build/qemu-10.0.2 does NOT -> kernel p2pdma OFF
# at runtime -> cross-VM GDR / dual-GPU P2P dmabuf reg HANGS (RCCL sees CONFIG_PCI_P2PDMA=y,
# forces GDR, runtime reg fails -> hard hang, not a host-bounce fallback). `-L pc-bios` else
# "failed to find romfile efi-virtio.rom". See README "de-bypass qemu" note.
exec /home/ubuntu/qemu_p2p_fresh/qemu-10.0.2-beta/build/qemu-system-riscv64 \
  -L /home/ubuntu/qemu_p2p_fresh/qemu-10.0.2-beta/build/pc-bios \
  -machine beta,config-file=/home/ubuntu/p2p_archive/artifacts/beta_direct_baremetal-64GB-pref.json \
  -device loader,file=/home/ubuntu/fw_jump_0x4000000000.bin,addr=0x4000000000 \
  -kernel /home/ubuntu/p2p_archive/artifacts/Image-6.19.5-p2p-all \
  -append "root=/dev/sda1 rootwait rdinit=/init earlycon=uart8250,mmio32,0xD0087000,115200n8 console=ttyS0,115200n8 mem=64G iommu.passthrough=1" \
  -drive file=/home/ubuntu/kevin/ubuntu-25.10-preinstalled-server-riscv64.img,format=raw,id=hd0,if=none \
  -drive file=/home/ubuntu/kevin/models.img,format=raw,id=hd1,if=none,readonly=on \
  -device virtio-scsi-pci,bus=pcie0.0,id=scsi0 \
  -device scsi-hd,drive=hd0,bus=scsi0.0,channel=0,scsi-id=0,lun=0 \
  -device scsi-hd,drive=hd1,bus=scsi0.0,channel=0,scsi-id=0,lun=1 \
  -netdev user,id=net0,hostfwd=tcp::2224-:22 \
  -device virtio-net-pci,netdev=net0,bus=pcie1.0 \
  -object iommufd,id=iommufd0 \
  -device vfio-pci,host=0000:23:00.0,bus=pcie2.0,multifunction=on,addr=0.0,iommufd=iommufd0 \
  -device vfio-pci,host=0000:23:00.1,bus=pcie2.0,addr=0.1,iommufd=iommufd0 \
  -device vfio-pci,host=0000:01:00.0,bus=pcie3.0,addr=0.0,iommufd=iommufd0 \
  -nographic -monitor none -serial file:/home/ubuntu/ib-vm1-console.log
