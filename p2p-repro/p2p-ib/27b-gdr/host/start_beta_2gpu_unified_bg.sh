#!/bin/bash
# Dual-GPU (single VM) regression on the UNIFIED kernel, +iommu.passthrough=1
# (the old 2gpu launcher lacked it -> NCCL "Missing iommu=pt" warn + slow P2P). serial->file.
set -eu
cd /home/ubuntu
exec /home/ubuntu/p2p_build/qemu-10.0.2/build/qemu-system-riscv64 \
  -machine beta,config-file=/home/ubuntu/p2p_archive/artifacts/beta_direct_baremetal-64GB-pref.json \
  -device loader,file=/home/ubuntu/fw_jump_0x4000000000.bin,addr=0x4000000000 \
  -kernel /home/ubuntu/p2p_archive/artifacts/Image-6.19.5-p2p-all \
  -append "root=/dev/sda1 rootwait rdinit=/init earlycon=uart8250,mmio32,0xD0087000,115200n8 console=ttyS0,115200n8 mem=64G iommu.passthrough=1" \
  -drive file=/home/ubuntu/kevin/ubuntu-25.10-preinstalled-server-riscv64.img,format=raw,id=hd0,if=none \
  -drive file=/home/ubuntu/kevin/models.img,format=raw,id=hd1,if=none \
  -device virtio-scsi-pci,bus=pcie0.0,id=scsi0 \
  -device scsi-hd,drive=hd0,bus=scsi0.0,channel=0,scsi-id=0,lun=0 \
  -device scsi-hd,drive=hd1,bus=scsi0.0,channel=0,scsi-id=0,lun=1 \
  -netdev user,id=net0,hostfwd=tcp::2224-:22 \
  -device virtio-net-pci,netdev=net0,bus=pcie1.0 \
  -object iommufd,id=iommufd0 \
  -device vfio-pci,host=0000:23:00.0,bus=pcie2.0,multifunction=on,addr=0.0,iommufd=iommufd0 \
  -device vfio-pci,host=0000:23:00.1,bus=pcie2.0,addr=0.1,iommufd=iommufd0 \
  -device vfio-pci,host=0000:43:00.0,bus=pcie3.0,multifunction=on,addr=0.0,iommufd=iommufd0 \
  -device vfio-pci,host=0000:43:00.1,bus=pcie3.0,addr=0.1,iommufd=iommufd0 \
  -nographic -monitor none -serial file:/home/ubuntu/dual-gpu-console.log
