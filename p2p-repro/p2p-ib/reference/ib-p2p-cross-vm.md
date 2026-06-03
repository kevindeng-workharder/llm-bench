# Cross-VM IB-P2P TP=2 inference (Mellanox CX-7 RoCE / NCCL NET=IB)

This document covers the host topology, software path, and the concrete
blockers we hit getting vLLM TP=2 inference running across **two separate QEMU
guests** that talk to each other over a Mellanox dual-port CX-7 NIC using RoCE
(RDMA over Converged Ethernet, NCCL's `NET=IB` transport).

It is the cross-VM counterpart to the single-VM Infinity Fabric P2P path
documented elsewhere; both work, but they exercise very different parts of the
stack. See `results/2026-05-25.md` in the [llm-bench
repo](https://github.com/kevindeng-workharder/llm-bench) for the head-to-head
benchmark.

## When to use this mode

- You want to validate the IB / RoCE data path without paying for two physical
  hosts.
- You want each guest to have its own kernel / RCCL / vLLM image so faults are
  isolated to one VM.
- You're stress-testing the iommufd + VFIO multi-instance code path.

If you have both GPUs in one host and don't care about wire-level RDMA,
single-VM Infinity Fabric P2P (`start_ubuntu_vfio_dual.sh`) is simpler and
~6–12 % faster (per `results/2026-05-25.md`).

## Topology

```
Host (10.103.11.199, 62 GiB RAM, AMD Turin)
├── VM1  (16 GiB, ssh hostfwd 2224→22)
│   ├── PCIe passthrough: GPU0 (0000:23:00.0/.1, gfx1100)
│   ├── PCIe passthrough: NIC port 0 (0000:01:00.0, mlx5_core → roceP3p1s0)
│   └── 10.99.0.1/30 on enP3p1s0np0
└── VM2  (16 GiB, ssh hostfwd 2225→22)
    ├── PCIe passthrough: GPU1 (0000:43:00.0/.1, gfx1100)
    ├── PCIe passthrough: NIC port 1 (0000:01:00.1, mlx5_core → roceP3p1s0)
    └── 10.99.0.2/30 on enP3p1s0np1
```

Both GPUs and both NIC ports are bound to `vfio-pci` on the host before either
QEMU launches.

## Host prep (once per host boot)

```bash
sudo /opt/rocm-riscv-build/vm/setup-vfio-2vm.sh
```

This is idempotent. It unbinds each device from its host driver
(amdgpu / snd_hda_intel / mlx5_core) and rebinds to `vfio-pci`.

## Machine config: 16 GiB ddr0 per guest

The default `beta_direct_baremetal-*.json` in `p2p_archive/artifacts/` is
24 GiB-per-guest. With **two** of those on the 62 GiB host, the OOM killer
chooses one mid-vLLM-init — iommufd pins the entire guest RAM region eagerly,
so neither the kernel pagecache nor swap can help. Use 16 GiB instead:

```bash
/opt/rocm-riscv-build/vm/make-16gb-machine-config.sh \
    /home/ubuntu/p2p_archive/artifacts/beta_direct_baremetal-24GB-pref.json \
    /home/ubuntu/p2p_archive/artifacts/beta_direct_baremetal-16GB-pref.json
```

(Two changes: the `ddr0` `memory_regions` entry and the `ddr.channels[0]` entry
both go `0x600000000 → 0x400000000`. Everything else — DDR controller CSR
addresses, the 64 GiB DDR0 physical window — stays the same.)

## Launching the two guests

```bash
nohup /opt/rocm-riscv-build/vm/start_vm1.sh > vm1.log 2>&1 &
nohup /opt/rocm-riscv-build/vm/start_vm2.sh > vm2.log 2>&1 &
```

Each script:
- Loads `beta_direct_baremetal-16GB-pref.json` as the machine config
- Loads `fw_jump_0x4000000000.bin` + `Image-6.19.5-p2p-local` (kernel with
  mlx5_core, IB core, RDMA RXE, INFINIBAND_USER_ACCESS modules)
- Appends `mem=16G` to the kernel cmdline (belt-and-braces; the machine config
  already caps ddr0 at 16 GiB)
- Attaches the rootfs (`ubuntu-25.10-preinstalled-server-riscv64.img` for VM1,
  a copy at `ubuntu-vm2.img` for VM2) read-write, and the shared `models.img`
  read-only

## In-guest post-boot setup (run once per guest boot)

```bash
# VM1
ssh -p 2224 ubuntu@127.0.0.1 \
    'sudo bash /home/ubuntu/vm_boot_setup.sh 10.99.0.1/30 enP3p1s0np0'
# VM2
ssh -p 2225 ubuntu@127.0.0.1 \
    'sudo bash /home/ubuntu/vm_boot_setup.sh 10.99.0.2/30 enP3p1s0np1'
```

This script (also in this repo at `vm/vm_boot_setup.sh`):
1. Mounts the shared `/data` partition read-only with `noload` (skip journal
   replay; both VMs mount it concurrently)
2. **Disables IPv6 everywhere** and flushes link-local addresses — Gloo's TCP
   transport picks IPv6 LL addresses on multi-homed hosts and then asserts
   `ss1.ss_family == ss2.ss_family` mid-handshake (`device.cc:285`).
3. Assigns the `/30` IPv4 on the Mellanox iface and brings it up.

Verify with `ping -c 3 10.99.0.{2,1}` from VM1 and VM2 respectively.

## Stage tests before vLLM

Run these in order. They progressively widen what's exercised.

### Stage 1 — bare RDMA verbs (no NCCL, no GPU)

`scripts/diagnostics/rdma_write_test.c` is a ~250-line libibverbs program
that does an RDMA_WRITE between the two NIC ports. Compile and run on each
side; it should hit ~179 Gb/s with 100 % data verification.

If this hangs you have a fabric-level problem (GID, MTU, PSN, PFC) that no
amount of NCCL tuning will fix.

### Stage 2 — NCCL all-reduce cross-VM

`scripts/diagnostics/nccl_cross_vm.py` + `run_nccl_cross_vm.sh`. Spawns 2
ranks (rank 0 on VM1, rank 1 on VM2) and does a sweep:
4 B / 1 KiB / 1 MiB / 64 MiB all-reduces over `NET=IB`.

Expected on this hardware: **18 GB/s effective AllReduce bandwidth** at 64 MiB
(measured 17.5–18.2 GB/s across 14 consecutive clean runs; the older 16.9 GB/s
figure was on a busier host). For reference, raw libibverbs `RDMA_WRITE` with no
NCCL and no GPU (`scripts/diagnostics/rdma_write_test.c`) hits 22.8 GB/s, so NCCL
reaches ~80% of line rate even with every GPU↔NIC DMA bouncing through host
memory. The first 4 B AllReduce takes ~16 s — that is the lazy IB-connect setup
over the host-bounce path, not a hang; subsequent collectives are fast.

⚠️ **`NCCL_IB_HCA` must not be `mlx5_0`** — the guest kernel exposes the device
as `roceP3p1s0` (PCI-derived udev name) because `mlx5_core` doesn't claim it
under that classic name path in our build. Either set
`NCCL_IB_HCA=roceP3p1s0` or leave it unset and let RCCL auto-discover.

### Stage 3 — vLLM TP=2 cross-VM

Leader (VM1) runs the OpenAI API server; follower (VM2) runs `vllm serve
--headless`:

```bash
# VM1
bash /opt/rocm-riscv-build/vm/start_vllm_xnode_vm1.sh
# VM2
bash /opt/rocm-riscv-build/vm/start_vllm_xnode_vm2.sh
```

`Application startup complete` on VM1 means it's live on `0.0.0.0:8000`.

## Blockers we hit and how we fixed them

These are the non-obvious ones; if you're reproducing this, expect to debug
them again unless you take the matching mitigation:

1. **Host OOM at second guest boot.** Two 24 GiB QEMUs + iommufd's eager pin
   of the entire guest RAM region overshoots a 62 GiB host. Swap doesn't help
   because pinned pages aren't reclaimable. Fix: shrink the `ddr0` machine
   region to 16 GiB (see `make-16gb-machine-config.sh`). Per-VM QEMU RSS drops
   from 26 GiB → 17 GiB.

2. **vLLM's MQ chose the wrong IP.** `vllm/utils/network_utils.py:get_ip()`
   probes the outbound IP by `socket.connect(("8.8.8.8", 80))` — in a QEMU NAT
   guest that always returns `10.0.2.15`, which is not reachable from the
   peer. Set `VLLM_HOST_IP=10.99.0.{1,2}` to override.

3. **NCCL_IB_HCA=mlx5_0 didn't match the device.** See Stage 2 above. Old
   Stage 2 launcher scripts had `NCCL_IB_HCA=mlx5_0` set but unexported, so
   NCCL fell through to auto-discovery and "worked" — vLLM's launcher *did*
   export it and then NCCL failed with `NET/IB : No device found`. Use
   `roceP3p1s0` or unset.

4. **Follower node ran the wrong vLLM entrypoint.** Launching
   `vllm.entrypoints.openai.api_server` on the follower triggers
   `EngineCore._initialize_kv_caches()`, which calls `collective_rpc` and
   asserts because the follower has no `rpc_broadcast_mq`. Fix: on the
   follower, run `vllm serve --headless` instead — that path hits
   `cli/serve.py:run_headless()` which constructs only the
   `MultiprocExecutor + worker monitor` and skips the leader-only init.

5. **`pkill -f vllm` left orphans on port 29500.** Worker processes rename
   themselves to `VLLM::Worker_TP0` etc; `pkill -f` matches against
   `/proc/<pid>/cmdline` (which still contains "vllm") but if the cmdline read
   races with process state changes you can miss them. After a crash, always
   re-check: `sudo ss -lntp | grep 29500`; kill any holder by PID before
   restarting the leader.

6. **Connect-time `strcmp(NULL)` SIGSEGV from a stale previous run.** When a
   prior cross-VM run left orphan ranks alive — typically stuck in the 4 B
   AllReduce's lazy IB-connect, which takes ~16 s here because it bounces
   through host memory — the *next* run's RCCL topology build reads a
   dangling/dirty pointer and crashes in glibc `strcmp` (NULL deref,
   `unhandled signal 11 ... at 0x0 in libc.so.6`). This is **not** ASLR- or
   IOMMU-related: we chased `iommu.passthrough` / `iommu.strict=0` and then
   `randomize_va_space` and both were red herrings (raw `RDMA_WRITE` and 14/14
   clean NCCL runs are unaffected). The real variable is leftover state. The
   gdb run "fixing" it was a coincidence — gdb just ran slow enough to let the
   stale rank finish dying. Mitigation: hard-kill stale ranks on **both**
   guests before every relaunch — `pkill -9 -f nccl_cross_vm; pkill -9 -f
   lat_test` on VM1 *and* VM2 — exactly the Blocker 5 discipline. With clean
   teardown, 14/14 consecutive runs succeed at 18 GB/s. (The RCCL pointer is
   worth hardening upstream, but clean teardown fully avoids it in practice.)

## What it does **not** test

- **GPUDirect RDMA.** AMD GPUDirect across PCIe roots needs a peer-bridge that
  isn't present on this host, so all GPU↔NIC DMA goes through host memory.
  NCCL's IB transport stages through pinned host buffers, which is fine for
  correctness but caps effective bandwidth below what GDR would deliver.
- **TP > 2 or PP.** All scripts hard-code TP=2 + PP=1.
- **Anything other than `mp` distributed executor.** Ray would require
  multi-VM Ray setup we haven't built.
