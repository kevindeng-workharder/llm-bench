# p2p-shm-pp2 — pipeline parallelism (PP=2) single-VM, SHM transport

The **PP cell** of the [p2p-shm](../p2p-shm) transport: single VM, both GPUs as two pipeline
stages, with NCCL forced onto its **SHM** transport (host shared memory) via the split topo —
vs [p2p-shm](../p2p-shm) which is the same SHM transport but **tensor-parallel**. Completes the
TP×PP matrix.

- **Status:** ✅ verified 2026-06-06 — single-stream **14.40 tok/s**, N=4 aggregate **43.55**.
  Output correct. See [RESULTS.md](RESULTS.md).
- **Two findings:** (1) PP **loses to TP** on single-VM (14.40 < TP's 15.41) — fast link, TP's
  all-reduce is cheap. (2) PP is **transport-insensitive** on single-VM: PP@SHM 14.40 ≈ PP@P2P
  14.21 ([p2p-direct-pp2](../p2p-direct-pp2)), whereas TP@SHM 15.41 < TP@P2P 16.20 (−5 %). PP
  sends 1 handoff/token so the GPU↔GPU path barely matters; TP all-reduces every layer so it
  does. (Same reason PP wins cross-VM — see [p2p-ib-pp2](../p2p-ib-pp2).)

## Requires

Same single-VM dual-GPU setup as [../p2p-direct](../p2p-direct) (unified kernel, gemv patch),
plus `rccl-topo-split.xml` deployed to `/home/ubuntu/` (the SHM-forcing topo, shared with
[../p2p-shm](../p2p-shm)).

## Files

| file | role |
|---|---|
| `start_shm_pp.sh` | launcher — single-VM PP (`--pipeline-parallel-size 2`) + `NCCL_TOPO_FILE=rccl-topo-split.xml` + `--disable-custom-all-reduce` to force SHM. → deploy to VM `/home/ubuntu/shm_pp.sh` |
| `RESULTS.md` | numbers + the PP-loses-single-VM and PP-transport-insensitive findings |

(Uses `rccl-topo-split.xml` from [../p2p-shm](../p2p-shm) — not duplicated here.)

## Run

```bash
# single-VM dual-GPU guest up (see ../p2p-ib/27b-gdr/host/start_beta_2gpu_unified_bg.sh), /data mounted.
scp -P 2224 ../p2p-shm/rccl-topo-split.xml ubuntu@127.0.0.1:/home/ubuntu/
scp -P 2224 start_shm_pp.sh                ubuntu@127.0.0.1:/home/ubuntu/shm_pp.sh
ssh -p 2224 ubuntu@127.0.0.1 'cd ~ && setsid bash shm_pp.sh </dev/null >/tmp/vllm_vm1.log 2>&1 &'
# (GDN kernels cached on this image -> ~15-20min boot, no 2h recompile)
```
