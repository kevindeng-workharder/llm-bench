# p2p-shm — single-VM dual-GPU forced through host SHM transport

Same single guest with both gfx1100 GPUs as [../p2p-direct](../p2p-direct), but
NCCL is steered onto its **SHM transport** (GPU → pinned host shared memory →
GPU) instead of direct P2P. Two changes vs p2p-direct, nothing else:

1. **`NCCL_TOPO_FILE=rccl-topo-split.xml`** — a *split* topology that puts the two
   GPUs under different fake CPU/NUMA nodes (`gdr=0`), so RCCL picks
   `SHM/direct/direct` instead of `P2P/IPC`.
2. **`--disable-custom-all-reduce`** — so the TP all-reduce goes through NCCL
   (→ SHM), not vLLM's custom all-reduce (which would still run over P2P/Infinity
   Fabric and **bypass** the SHM path, making the "SHM" label a lie).

Useful when P2P DMA is unavailable or under debug — it sidesteps P2P entirely.

- **Status:** ✅ verified 2026-06-05 — `via SHM/direct/direct`, **~15.4 tok/s**
  single-stream decode (Qwen3.6-27B Quark INT8, TP=2), no hangs at N=1/2/4. See
  [RESULTS.md](RESULTS.md).
- **vs [p2p-direct](../p2p-direct):** P2P is ~5% faster single-stream (16.2 vs
  15.4) — small, because single-stream INT8 decode is gemv-bound and the
  per-token all-reduce is tiny. Same model/ctx/kernel; only the two knobs differ.

## Requires (same host setup as p2p-direct)

Unified kernel `Image-6.19.5-p2p-all` + guest `iommu.passthrough=1` (build/boot:
[`../p2p-ib/27b-gdr/host/`](../p2p-ib/27b-gdr/host)), the gemv INT8 patch, and
`HSA_FORCE_FINE_GRAIN_PCIE=1` (shared env). The kernel's P2P hack isn't strictly
needed for SHM, but this scenario runs on the same image as the others.

## Files

| file | role |
|---|---|
| `start_shm.sh` | launcher — split topo + `--disable-custom-all-reduce` → deploy to VM `/home/ubuntu/p2p-shm-2gpu.sh` |
| `rccl-topo-split.xml` | the split (fake cross-NUMA) topology that forces SHM; twin of p2p-direct's `rccl-topo.xml` |
| `bench.py` | concurrency sweep (shared with p2p-direct) |
| `RESULTS.md` | transport proof + throughput + p2p-direct comparison |

## Run

```bash
# boot the dual-GPU guest (same as p2p-direct):
sudo setsid bash ../p2p-ib/27b-gdr/host/start_beta_2gpu_unified_bg.sh &
ssh -p 2224 ubuntu@127.0.0.1 'sudo mount -o ro,norecovery /dev/sdb /data'

# deploy launcher + split topo, run (single process)
scp start_shm.sh        p2p-host:/tmp/ && ssh p2p-host 'scp -P 2224 /tmp/start_shm.sh        ubuntu@127.0.0.1:/home/ubuntu/p2p-shm-2gpu.sh'
scp rccl-topo-split.xml p2p-host:/tmp/ && ssh p2p-host 'scp -P 2224 /tmp/rccl-topo-split.xml ubuntu@127.0.0.1:/home/ubuntu/'
ssh -p 2224 ubuntu@127.0.0.1 'cd ~ && setsid bash p2p-shm-2gpu.sh </dev/null >/tmp/vllm.log 2>&1'

# verify SHM + bench  (~15 min cold start)
ssh -p 2224 ubuntu@127.0.0.1 'grep -aoE "via SHM[a-z/]*|via P2P/IPC|via NET" /tmp/vllm.log | sort -u'   # want: via SHM/direct/direct
ssh -p 2224 ubuntu@127.0.0.1 'python3 ~/bench.py 1 2 4'
```

## Note on the guest's original `start_shm_48k.sh`

The pre-existing SHM launcher used 48K context **and kept custom all-reduce
enabled** — so its all-reduce actually ran over P2P (CA), and only NCCL's non-CA
paths saw SHM. This archived `start_shm.sh` uses the same 2048-ctx config as
p2p-direct and disables CA, so the all-reduce genuinely goes through SHM — a fair
p2p-direct-vs-p2p-shm transport comparison.
