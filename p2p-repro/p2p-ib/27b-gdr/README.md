# p2p-ib · 27B Quark-INT8 · graph-mode **+ GDR** (GPUDirect RDMA)

The GDR counterpart to [`../27b-graph`](../27b-graph): identical model, cross-VM
RoCE path, and cudagraph config, but the NIC now DMAs **directly to/from GPU
VRAM** instead of bouncing through host memory. This closes the parent README's
*"Known limit: GDR not tested (no peer-bridge across PCIe roots → host-bounce)"*.

- **Status:** ✅ verified 2026-06-05; **re-confirmed 2026-06-07 on vllm-venv** — NCCL reports
  **`use ring PXN 0 GDR 1`** + `via NET/IB/0/GDRDMA`, correct output, **5.60 tok/s** single /
  19.89 N=4 (TTFT-cancelled). The re-test caught that the vLLM launcher also needs
  **`NCCL_NET_GDR_LEVEL=SYS`** + **`NCCL_DMABUF_ENABLE=1`** (not just `RCCL_FORCE_ENABLE_DMABUF=1`),
  or it silently runs `GDR 0` — now fixed in the launchers. See [RESULTS.md](RESULTS.md).
- **Why it had been failing:** TWO independent software bugs, both required to
  fix — one kernel, one RCCL. The HANDOFF that blamed "QEMU PCIe topology" was
  wrong; nothing about the topology changed.

## Root cause (two bugs, both must be fixed)

**1. Kernel — missing `cpu_supports_p2pdma()` patch.**
`Image-6.19.5-p2p-ib` was built from the *stock* `qemu_soc_vendor` tree. On
RISC-V, stock `drivers/pci/p2pdma.c:cpu_supports_p2pdma()` returns `false`
(the `true` branch is `#ifdef CONFIG_X86`), so `pci_p2pdma_distance()` hits
`map_through_host_bridge` (p2pdma.c:767) and returns **< 0** for any pair of
passthrough devices on different PCIe segments — including GPU↔NIC. amdgpu's
dma-buf exporter (`amdgpu_dma_buf.c:99`) then sets `peer2peer = false`, so the
*real* `ibv_reg_dmabuf_mr` of GPU VRAM fails. The dual-GPU kernel had a 1-line
hack forcing `return true`; the IB kernel didn't. The check is device-agnostic,
so the same hack enables GPU↔NIC exactly like GPU↔GPU.

> Note: the mlx5/uverbs dma-buf *capability* was always present —
> [`tools/dmabuf_test.c`](tools/dmabuf_test.c) (a bad-fd `ibv_reg_dmabuf_mr`)
> returns `EBADF`, not `EOPNOTSUPP`. The kernel just refused the *real* P2P
> registration because of the missing hack.

**2. RCCL — reads gzipped `/proc/config.gz` as plaintext.**
`ROCm-RCCL-7.2.3/src/misc/rocmwrap.cc`: after the HSA dma-buf query *passes*,
RCCL "double-checks" by reading the kernel config; its path list has
`/proc/config.gz` **first** and reads it with `fopen`+`fgets` as **plain text**
(it's gzip-compressed) → never matches `CONFIG_PCI_P2PDMA=y` → `break`s the loop
**before** ever trying `/boot/config-$(uname -r)` → prints
`DMA_BUF_SUPPORT Failed due to OS kernel support`. So even a perfectly correct
kernel is rejected. **Fix:** `export RCCL_FORCE_ENABLE_DMABUF=1` (rocmwrap.cc:154
skips the whole file check; the HSA query alone then governs).

**3. NCCL — GDR level must be forced to `SYS` (found 2026-06-07).** Even with bugs 1+2 fixed, the
vLLM launcher logged `GDR 0` until it *also* exported **`NCCL_NET_GDR_LEVEL=SYS`** +
**`NCCL_DMABUF_ENABLE=1`**. The cross-root GPU↔NIC pair exceeds NCCL's default GDR distance, so GDR is
disabled for that link unless the level is forced to `SYS`. [`tools/run_nccl_test.sh`](tools/run_nccl_test.sh)
always set these (hence it showed `GDR 1`); the vLLM launcher had only `RCCL_FORCE_ENABLE_DMABUF=1` and so
silently fell back to host-bounce. With all **three** env vars the vLLM run logs `use ring PXN 0 GDR 1` +
`via NET/IB/0/GDRDMA`. So the guest launchers differ from `27b-graph` by **three** env vars, not one.

## Files

| file | role |
|------|------|
| `start_vm1_leader.sh` | LEADER (node 0) = `27b-graph` leader **+ `RCCL_FORCE_ENABLE_DMABUF=1`** → VM1 `/home/ubuntu/graph27b_vm.sh` |
| `start_vm2_follower_headless.sh` | FOLLOWER (node 1) = same + the env var → VM2 `/home/ubuntu/graph27b_vm.sh` |
| `host/build_kernel_unified.sh` | build `Image-6.19.5-p2p-all`: applies the patch to the Beta-SoC kernel tree + the proven `-p2p-ib` `.config` base + gcc-15. Gates on a config + hack verify. |
| `host/kernel-6.19.5-p2p.patch` | **the actual P2P kernel patch.** `cpu_supports_p2pdma()→true` (`p2pdma.c` — the GDR lever), amdgpu `is_large_bar`+P2P-DBG (`amdgpu_device.c`), `kfd_topology.c`, and the Beta-SoC DWC PCIe controller Kconfig/Makefile. Applied onto a `linux-6.19.5` tree that already carries the qemu_soc Beta-SoC patches. |
| `host/kernel-6.19.5-p2p-ib.config` | the proven `-p2p-ib` kernel `.config` used as the build base (`PCI_P2PDMA=y` `HSA_AMD_P2P=y` `ZONE_DEVICE=y` + full IB). guest.config lacked `ZONE_DEVICE`, so it could *not* be the base — `olddefconfig` would drop `PCI_P2PDMA`. |
| `host/install_unified_modules.sh` | offline loop-mount both guest images and `rsync`+`depmod` the unified modules (incl. `mlx5_ib`, `amdgpu`) |
| `host/start_vm{1,2}_64g_unified_bg.sh` | QEMU launchers pointing `-kernel` at `Image-6.19.5-p2p-all` (serial→file, detachable) |
| `host/start_beta_2gpu_unified_bg.sh` | single-VM dual-GPU launcher on the same kernel (`+iommu.passthrough=1`) — the unified kernel serves both scenarios |
| `tools/dmabuf_test.c` | minimal `ibv_reg_dmabuf_mr(fd=-1)` probe → `EBADF` = capability present |
| `tools/run_nccl_test.sh`, `tools/nccl_alltest.py` | 2-node RoCE all-reduce — checks `GDR 1` in ~1 min without a 15-min model load |

## Reproduce

```bash
# --- HOST: build the unified kernel + install modules into both guest images ---
bash host/build_kernel_unified.sh                       # -> /home/ubuntu/p2p_archive/artifacts/Image-6.19.5-p2p-all
echo <host-sudo-pw> | sudo -S bash host/install_unified_modules.sh

# --- HOST: boot the two IB guests on the unified kernel ---
sudo setsid bash host/start_vm1_64g_unified_bg.sh &     # GPU 23:00 + NIC port0, ssh 2224
sudo setsid bash host/start_vm2_64g_unified_bg.sh &     # GPU 43:00 + NIC port1, ssh 2225

# --- per-guest: assign IB IP (does NOT auto-assign — mlx5 link comes up ~178s,
#     after networkd) and mount the model image ---
ssh -p 2224 ubuntu@127.0.0.1 'sudo ip addr add 10.99.0.1/24 dev enP3p1s0np0; sudo ip link set enP3p1s0np0 up; sudo mount -o ro,norecovery /dev/sdb /data'
ssh -p 2225 ubuntu@127.0.0.1 'sudo ip addr add 10.99.0.2/24 dev enP3p1s0np1; sudo ip link set enP3p1s0np1 up; sudo mount -o ro,norecovery /dev/sdb /data'
#   note: VM2's IB netdev is enP3p1s0np**1** (NIC port 1), not np0.

# --- fast GDR check (no model load, ~1 min): expect "use ring PXN 0 GDR 1" ---
ssh -p 2225 ubuntu@127.0.0.1 'cat > ~/run_nccl_test.sh' < tools/run_nccl_test.sh   # + nccl_alltest.py, both VMs
ssh -p 2225 ubuntu@127.0.0.1 'nohup bash ~/run_nccl_test.sh 1 enP3p1s0np1 >~/nccl_test.log 2>&1 &'
ssh -p 2224 ubuntu@127.0.0.1 'bash ~/run_nccl_test.sh 0 enP3p1s0np0 2>&1 | grep -E "GDR [0-9]|ALLREDUCE_OK|Failed"'

# --- full 27B serve: deploy launchers, start leader then follower ---
ssh -p 2224 ... start_vm1_leader.sh -> /home/ubuntu/graph27b_vm.sh   # see 27b-graph/README for the scp dance
ssh -p 2225 ... start_vm2_follower_headless.sh -> /home/ubuntu/graph27b_vm.sh
ssh -p 2224 ubuntu@127.0.0.1 'cd ~ && setsid bash graph27b_vm.sh </dev/null >/tmp/vllm_vm1.log 2>&1'
ssh -p 2225 ubuntu@127.0.0.1 'cd ~ && setsid bash graph27b_vm.sh </dev/null >/tmp/vllm_vm2.log 2>&1'
```

> Launch gotcha: do **not** put a foreground `sleep` between the two `ssh`
> launches in a single shell step if your harness blocks foreground sleep — the
> command dies after the first node and the second never starts. Fire both,
> no sleep.

## Verify GDR is real (not just advertised)

```bash
# NCCL transport line — the verdict
ssh -p 2224 ubuntu@127.0.0.1 'grep -aE "DMA_BUF|use ring PXN|force enabled" /tmp/vllm_vm1.log | tail -4'
#   want:  DMA_BUF Support is force enabled ... (RCCL_FORCE_ENABLE_DMABUF=1)
#          Connected all rings, use ring PXN 0 GDR 1      <-- GDR 1, no "DMA_BUF_SUPPORT Failed"

# end-to-end + steady-state decode
ssh -p 2224 ubuntu@127.0.0.1 'curl -s :8000/v1/completions -d "{\"model\":\"qwen3_6-27b-int8\",\"prompt\":\"The capital of France is\",\"max_tokens\":128,\"temperature\":0}"'
ssh -p 2224 ubuntu@127.0.0.1 'grep -oE "Avg generation throughput: [0-9.]+ tokens/s" /tmp/vllm_vm1.log | tail -3'
```

## Honest perf note

GDR engaging is the technical win; the throughput gain over host-staged is
**modest** here — ~5.4–6.2 tok/s vs ~5.2 (`27b-graph` single-stream). Single-
stream 27B decode all-reduces a small tensor per token, so GDR mostly saves
small-transfer bounce latency; on this QEMU-emulated platform the bottleneck is
more the emulation than the GPU↔NIC copy. GDR's payoff grows with bigger /
more communication-heavy transfers. The point of this scenario is that the data
path is now **direct** and correct, removing the host bounce buffer.
