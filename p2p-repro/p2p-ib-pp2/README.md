# p2p-ib-pp2 — pipeline parallelism (PP=2) across two VMs over RoCE/IB

The **PP cell** of the [p2p-ib](../p2p-ib) transport (2 guests, 1 GPU each, NCCL over
RoCE/IB) — same cross-VM wire as the TP scenarios there, but **pipeline-parallel** (each VM
= one pipeline stage, one activation handoff per token) instead of **tensor-parallel** (each
layer split, all-reduce every layer).

- **Status:** ✅ verified 2026-06-06 — **PP beats TP ~2.3–2.5× on cross-VM IB**:
  single-stream **7.63 tok/s** (TP ~3.0), N=4 aggregate **24.97 tok/s** (TP ~11). Output
  correct, no hang. See [RESULTS.md](RESULTS.md).
- **The crossover:** single-VM ([p2p-direct-pp2](../p2p-direct-pp2)) → **TP wins** (fast
  Infinity-Fabric, PP's bubble loses); cross-VM IB (here) → **PP wins** (slow link, TP's
  per-layer all-reduce dominates, PP sends one handoff/token). **Rule: PP for slow
  interconnects, TP for single-VM.**
- **GDR:** `via NET/IB/`, `DMA_BUF force enabled`, but `GDR 0` (host-bounce — same path as
  [../p2p-ib/27b-graph](../p2p-ib/27b-graph), so the comparison is apples-to-apples).

## Requires

The cross-VM IB setup from [../p2p-ib/27b-gdr](../p2p-ib/27b-gdr) (unified kernel
`Image-6.19.5-p2p-all`, 2-VM VFIO, RoCE IPs 10.99.0.1/.2, gemv patch, `RCCL_FORCE_ENABLE_DMABUF=1`).

## Files

| file | role |
|---|---|
| `start_vm1_leader.sh` | LEADER (node 0, VM1) = the 27b-gdr leader with `--tensor-parallel-size 2` → `--pipeline-parallel-size 2` and text-only (`--limit-mm 0/0`). → deploy to VM1 `/home/ubuntu/pp27b_vm.sh` |
| `start_vm2_follower_headless.sh` | FOLLOWER (node 1, VM2), same change → VM2 `/home/ubuntu/pp27b_vm.sh` |
| `RESULTS.md` | PP-vs-TP cross-VM numbers + the crossover analysis + bring-up gotchas |

## Run

```bash
# bring up the 2 IB guests (see ../p2p-ib/27b-gdr/README.md "Reproduce"), assign IPs, mount /data.
# deploy + launch leader then follower (no foreground sleep between):
scp -P 2224 start_vm1_leader.sh            ubuntu@127.0.0.1:/home/ubuntu/pp27b_vm.sh
scp -P 2225 start_vm2_follower_headless.sh ubuntu@127.0.0.1:/home/ubuntu/pp27b_vm.sh
ssh -p 2224 ubuntu@127.0.0.1 'cd ~ && setsid bash pp27b_vm.sh </dev/null >/tmp/vllm_vm1.log 2>&1 &'
ssh -p 2225 ubuntu@127.0.0.1 'cd ~ && setsid bash pp27b_vm.sh </dev/null >/tmp/vllm_vm2.log 2>&1 &'
# confirm PP ranks + GDR + bench (first boot ~1-2h GDN recompile; cached after):
ssh -p 2224 ubuntu@127.0.0.1 'grep -aoE "PP rank|use ring PXN [0-9] GDR [0-9]|Application startup complete" /tmp/vllm_vm1.log'
```
