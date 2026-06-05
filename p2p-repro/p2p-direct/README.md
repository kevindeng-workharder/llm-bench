# p2p-direct — single-VM dual-GPU over Infinity Fabric P2P

One QEMU/riscv64 guest with **both** gfx1100 GPUs passed through, TP=2 in a
single process. NCCL/RCCL uses its **P2P/IPC transport** — GPU↔GPU directly over
the AMD Infinity Fabric / PCIe P2P path, no NIC, no host-memory bounce. The
fastest and simplest of the three single-host scenarios.

- **Status:** ✅ verified 2026-06-05 — `via P2P/IPC`, **~16.2 tok/s** single-
  stream decode (Qwen3.6-27B Quark INT8, TP=2), **beating p2p-shm's ~15**, no
  hangs at N=1/2/4. See [RESULTS.md](RESULTS.md).
- **Contrast:** [../p2p-shm](../p2p-shm) (same VM, *forced* onto host SHM via a
  split topo) · [../p2p-ib](../p2p-ib) (cross-VM RoCE; its
  [`27b-gdr/`](../p2p-ib/27b-gdr) adds GDR).

## What makes it P2P — and what it requires

| requirement | why | where |
|---|---|---|
| **patched kernel** `Image-6.19.5-p2p-all` | stock RISC-V `cpu_supports_p2pdma()` returns false → `pci_p2pdma_distance(GPU0,GPU1)<0` → no KFD GPU↔GPU `p2p_links` → RCCL falls back to SHM. The patch forces it true. | build/install: [`../p2p-ib/27b-gdr/host/`](../p2p-ib/27b-gdr/host) |
| **`iommu.passthrough=1`** (guest cmdline) | without it the emulated RISC-V IOMMU translates *every* GPU DMA → ~2.6× slower (4.3 vs 11 tok/s). NCCL also warns "Missing iommu=pt" (it greps the x86 string; the riscv form *is* in effect). | host launcher [`../p2p-ib/27b-gdr/host/start_beta_2gpu_unified_bg.sh`](../p2p-ib/27b-gdr/host/start_beta_2gpu_unified_bg.sh) |
| **`rccl-topo.xml`** (this dir) | fake-same-root topology — the two real GPUs (`0002:`/`0003:`) placed under one bridge, `gdr=1` — so RCCL *selects* `P2P/IPC`, not SHM. The non-split twin of p2p-shm's `rccl-topo-split.xml`. | `start_dual_gpu.sh` → `NCCL_TOPO_FILE` (default in `../common/vllm-serve-env.sh`) |
| **`HSA_FORCE_FINE_GRAIN_PCIE=1`** | else the P2P custom-all-reduce kernel deadlocks on cross-GPU atomic signals. | already in `../common/vllm-serve-env.sh` |
| **gemv INT8 patch** | the INT8 decode GEMV fast-path; without it single-stream is GEMV-bound. Loaded via `sitecustomize.py` + `PYTHONPATH=/home/ubuntu` → `/home/ubuntu/gemv-patch/triton_scaled_mm.py`. | guest, shared by all scenarios |

## Two knobs p2p-ib turns OFF but p2p-direct leaves ON

`start_dual_gpu.sh` derives from p2p-ib `27b-graph`'s leader but **drops two
flags** p2p-ib needed only for cross-VM correctness — both cost throughput:

- **custom all-reduce.** p2p-ib runs `--disable-custom-all-reduce` (RoCE can't do
  vLLM's P2P CA). Single-VM P2P **can** — CA goes straight over Infinity Fabric.
  Leave it **enabled** (i.e. omit the flag).
- **async-scheduling.** p2p-ib `27b-graph` **must** use `--no-async-scheduling`
  or it hangs (the sampled-token copy CUDA event never signals under cudagraph).
  **That hang is tied to the cross-VM distributed-executor path, not cudagraph
  alone** — single-VM p2p-direct runs *with* async-scheduling and does **not**
  hang (verified N=1/2/4, clean startup + soak). Leave it **on**.

Removing both took single-stream 11.2 → **16.2** tok/s.

## Files

| file | role |
|---|---|
| `start_dual_gpu.sh` | verified single-VM TP=2 launcher (CA on, async on, `NCCL_IB_DISABLE=1`) → deploy to VM `/home/ubuntu/p2p-direct-2gpu.sh` |
| `rccl-topo.xml` | the fake-same-root P2P topology (non-split twin of p2p-shm's split XML) |
| `bench.py` | concurrency sweep (`python bench.py 1 2 4 8`): aggregate + per-req decode tok/s + hang/deadlock detection |
| `RESULTS.md` | transport proof, the iommu=pt / CA / async breakdown, throughput |

## Run

```bash
# HOST: build the unified kernel + boot the dual-GPU guest with iommu=pt
#   (build_kernel_unified.sh + install_unified_modules.sh live in ../p2p-ib/27b-gdr/host/)
sudo setsid bash ../p2p-ib/27b-gdr/host/start_beta_2gpu_unified_bg.sh &   # both GPUs, iommu.passthrough=1, ssh 2224

# guest one-time: mount the model image (attached read-only -> needs norecovery)
ssh -p 2224 ubuntu@127.0.0.1 'sudo mount -o ro,norecovery /dev/sdb /data'

# deploy launcher + topo, then run (single process, NO --nnodes)
scp start_dual_gpu.sh p2p-host:/tmp/ && ssh p2p-host 'scp -P 2224 /tmp/start_dual_gpu.sh ubuntu@127.0.0.1:/home/ubuntu/p2p-direct-2gpu.sh'
scp rccl-topo.xml    p2p-host:/tmp/ && ssh p2p-host 'scp -P 2224 /tmp/rccl-topo.xml    ubuntu@127.0.0.1:/home/ubuntu/'
ssh -p 2224 ubuntu@127.0.0.1 'cd ~ && setsid bash p2p-direct-2gpu.sh </dev/null >/tmp/vllm.log 2>&1'

# verify transport + bench  (~16 min cold start)
ssh -p 2224 ubuntu@127.0.0.1 'grep -aoE "via P2P/IPC|via SHM|via NET" /tmp/vllm.log | sort -u'   # want: via P2P/IPC
ssh -p 2224 ubuntu@127.0.0.1 'python3 ~/bench.py 1 2 4'
```
