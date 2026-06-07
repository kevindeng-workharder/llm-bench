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
- **Multimodal on vllm-venv (2026-06-07):** image **and video** verified on cross-VM PP. A
  Sintel-trailer clip was described scene-by-scene (red-haired protagonist, dragon, ruined city,
  "SINTEL" title) — real temporal understanding, **11.7K video tokens**. Runs on the rootfs
  `/home/ubuntu/vllm-venv` (gemv patch + **pyav**), the standard now that VM2 is a clone of the
  25.10 rootfs. The PP leader carries the whole vision tower, so video uses a **smaller window**
  than single-VM TP (16384, not 40960). See [RESULTS.md](RESULTS.md).

## Requires

The cross-VM IB setup from [../p2p-ib/27b-gdr](../p2p-ib/27b-gdr) (unified kernel
`Image-6.19.5-p2p-all`, 2-VM VFIO, RoCE IPs 10.99.0.1/.2, gemv patch, `RCCL_FORCE_ENABLE_DMABUF=1`).

**Venv:** `/home/ubuntu/vllm-venv` on each guest's rootfs (gemv patch + pyav). VM2 gets it by being
a **clone of the VM1 25.10 rootfs** (`ubuntu-vm2.img` = `cp` of
`ubuntu-25.10-preinstalled-server-riscv64.img`), which also carries the warm GDN Triton cache (no
~2 h recompile). Per-VM IP + `/data` mount via
[`../p2p-ib/tools/vm_boot_setup.sh`](../p2p-ib/tools/vm_boot_setup.sh) (`10.99.0.1/30 enP3p1s0np0`
for VM1, `.2/np1` for VM2).

## Files

| file | role |
|---|---|
| `start_vm1_leader.sh` / `start_vm2_follower_headless.sh` | **text** PP leader/follower (max-model-len 2048, seqs 8) |
| `start_vm1_leader_image-mm.sh` / `start_vm2_follower_headless_image-mm.sh` | **image** PP (vision path on, `TRITON_ATTN`, `max_pixels 200704`) |
| `start_vm1_leader_video-mm.sh` / `start_vm2_follower_headless_video-mm.sh` | **video** PP (pyav, `TRITON_ATTN`; max-model-len **16384**, seqs **1**, util **0.93** — see RESULTS "video memory") |
| `RESULTS.md` | PP-vs-TP cross-VM numbers + the crossover + the video result + bring-up gotchas |

All launchers run on `/home/ubuntu/vllm-venv`. **Leaders** call `…/vllm-venv/bin/python -m vllm…`;
**followers** call `…/vllm-venv/bin/python …/vllm-venv/bin/vllm serve …` — the explicit-python form is
required because `vllm-venv/bin/vllm`'s shebang points at the old `/data` python, so a plain path swap
would silently keep running `/data`.

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
