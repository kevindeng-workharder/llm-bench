# p2p-direct — single-VM dual-GPU over Infinity Fabric P2P  ⟨PLACEHOLDER⟩

> **Status: NOT yet archived.** Scaffold only — fill in after p2p-ib.

## What this scenario is

One QEMU guest with **both** gfx1100 GPUs passed through. NCCL uses its **P2P
transport** — direct GPU↔GPU over the AMD Infinity Fabric / PCIe P2P path, no
NIC, no host-memory bounce. This is the fastest of the three (the reference IB
doc notes single-VM Infinity Fabric is ~6–12 % faster than cross-VM IB) and the
simplest to launch (one guest, one process, no `--nnodes`).

The P2P AllReduce correctness fix lives in the shared env already:
`HSA_FORCE_FINE_GRAIN_PCIE=1` (see `../common/vllm-serve-env.sh`) — without it
the P2P kernel deadlocks on cross-GPU atomic signals.

## Known assets to harvest (on the host / guest)

- VM provisioning: `/opt/rocm-riscv-build/vm/start_ubuntu_vfio_dual.sh` (boots one guest, both GPUs)
- a single-VM TP=2 launcher (per task history: Qwen3.6-27B INT8, 32K ctx + image mm) — likely a variant of `/home/ubuntu/start_shm_48k.sh` WITHOUT the split topo, i.e. P2P/IPC transport
- topo XML: the 2-GPU-same-host `rccl-topo.xml` (P2P), not `rccl-topo-split.xml`
- env knobs: `NCCL_P2P_DISABLE` unset/0, `NCCL_SHM_DISABLE` unset, `NCCL_IB_DISABLE=1`

## TODO to make this reproducible (mirror p2p-ib's structure)

- [ ] `start_dual_gpu.sh` — verified single-VM TP=2 launcher (one api_server, no --nnodes)
- [ ] confirm NCCL log shows the **P2P** transport (`via P2P/IPC` / `P2P/direct`), not SHM/NET
- [ ] `bench.py` (can reuse `../p2p-ib/bench.py`) + record tok/s
- [ ] `RESULTS.md` — process shape, NCCL-transport proof, throughput
- [ ] note the topo XML + any HSA_FORCE_FINE_GRAIN_PCIE caveat
- [ ] cross-link to the single-VM doc if one exists under `/opt/rocm-riscv-build/docs/`
