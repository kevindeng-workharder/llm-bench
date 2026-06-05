# p2p-ib — cross-VM vLLM TP=2 over Mellanox CX-7 RoCE (NCCL NET=IB)

Two separate QEMU/riscv64 guests, one gfx1100 GPU each, talking over a
dual-port Mellanox CX-7 NIC using RoCE (RDMA over Converged Ethernet, NCCL's
`NET=IB` transport). Validates the wire-level RDMA data path without two
physical hosts.

- **Status:** ✅ verified working 2026-06-03 (Qwen3-4B FP16, ~10 tok/s, peak 12.5). See [RESULTS.md](RESULTS.md).
- **Deep background / fabric tuning / stage tests:** [reference/ib-p2p-cross-vm.md](reference/ib-p2p-cross-vm.md) (the authoritative host-side doc, 214 lines).
- **Contrast:** for single-VM dual-GPU there is no NIC in the loop — see [../p2p-direct](../p2p-direct) (Infinity Fabric) and [../p2p-shm](../p2p-shm) (host SHM).

## Topology

```
Host p2p-host (10.103.11.199, AMD Turin)
├── VM1  ssh -p 2224   GPU0 (gfx1100) + NIC port0 → roceP3p1s0 → 10.99.0.1/24 on enP3p1s0np0   [LEADER, node 0]
└── VM2  ssh -p 2225   GPU1 (gfx1100) + NIC port1 → roceP3p1s0 → 10.99.0.2/24 on enP3p1s0np1   [FOLLOWER, node 1]
```

Both GPUs and both NIC ports are bound to `vfio-pci` on the host before either
QEMU launches. By default GPU↔NIC DMA bounces through host memory — correct, but caps
bandwidth below line rate. The [`27b-gdr/`](27b-gdr/) variant enables true GDR
(NIC↔GPU direct VRAM DMA); see it for the two-bug root cause and the fix.

## Files in this directory

| file | role |
|------|------|
| `start_vm1_leader.sh` | LEADER launcher → deploy to VM1 `/home/ubuntu/graph_vm.sh` |
| `start_vm2_follower_headless.sh` | FOLLOWER launcher (`vllm serve --headless`) → deploy to VM2 `/home/ubuntu/graph_vm.sh` |
| `bench.py` | single-stream decode throughput (N=1, 80 tok) |
| `RESULTS.md` | the verified 2026-06-03 run (timeline, NCCL-over-IB proof, bench) |
| `reference/ib-p2p-cross-vm.md` | comprehensive host-side doc (VFIO, machine config, stage tests, all 6 blockers) |
| `../common/vllm-serve-env.sh` | shared runtime env, sourced by both launchers (deployed at `/home/ubuntu/vllm-serve-env.sh`) |

## Variants / sub-tests

| dir | what it adds |
|-----|--------------|
| [`27b-eager/`](27b-eager/) | larger model + eager mode: Qwen3.6-27B-VL Quark INT8, `--enforce-eager`. Proves the path is model/mode-agnostic, and characterises eager throughput (~0.29 tok/s) + the bottleneck (eager kernel-dispatch bound on the riscv64 host CPU). Includes `bench.py` (prefill/decode split) and `profile.sh` (GPU busy% + py-spy). |
| [`27b-graph/`](27b-graph/) | same 27B model in cudagraph `FULL_DECODE_ONLY` **+ `--no-async-scheduling`**. cudagraph confirms the eager bottleneck was dispatch (~15× faster raw). Default graph **hangs** — root-caused (py-spy) to vLLM async-scheduling's sampled-token copy CUDA event never signalling under cudagraph (not a collective deadlock, not the watchdog). With `--no-async-scheduling`: **stable** — ~3.0 tok/s single-stream, **~11 tok/s aggregate at N=4** (136 reqs, 0 hang / 20-min soak). |
| [`27b-gdr/`](27b-gdr/) | **GDR (GPUDirect RDMA)** variant of `27b-graph`: NIC↔GPU **direct** VRAM DMA, no host bounce — `use ring PXN 0 GDR 1`. Needs the unified kernel `Image-6.19.5-p2p-all` (a 1-line `cpu_supports_p2pdma` hack) **and** `RCCL_FORCE_ENABLE_DMABUF=1` (works around RCCL reading gzipped `/proc/config.gz` as plaintext). ~5.4–6.2 tok/s single-stream. Two independent bugs, both software — not the "topology" the HANDOFF blamed. |

## Prerequisites (once per host boot) — see reference doc for detail

```bash
# 1. Bind both GPUs + both NIC ports to vfio-pci (idempotent)
sudo /opt/rocm-riscv-build/vm/setup-vfio-2vm.sh

# 2. Boot the two guests (16 GiB each; 24 GiB OOMs the 62 GiB host)
nohup /opt/rocm-riscv-build/vm/start_vm1.sh > vm1.log 2>&1 &
nohup /opt/rocm-riscv-build/vm/start_vm2.sh > vm2.log 2>&1 &

# 3. Per-guest network setup (assigns the /24, disables IPv6 so Gloo doesn't
#    pick an IPv6 link-local and assert mid-handshake)
ssh -p 2224 ubuntu@127.0.0.1 'sudo bash /home/ubuntu/vm_boot_setup.sh 10.99.0.1/24 enP3p1s0np0'
ssh -p 2225 ubuntu@127.0.0.1 'sudo bash /home/ubuntu/vm_boot_setup.sh 10.99.0.2/24 enP3p1s0np1'

# verify fabric: from VM1  ->  ping -c3 10.99.0.2  ;  from VM2 -> ping -c3 10.99.0.1
#                rdma link show  ->  roceP3p1s0/1 state ACTIVE physical_state LINK_UP
```

Optional but recommended before vLLM: run the staged tests in the reference doc
(bare `RDMA_WRITE` ≈ 22.8 GB/s, then NCCL cross-VM all-reduce ≈ 18 GB/s). If
those pass, the fabric is sound and any remaining failure is in vLLM, not IB.

## Launch (the verified path)

Deploy the two launchers (both land at `/home/ubuntu/graph_vm.sh` on their
respective guest) and the shared env, then start leader first, follower a few
seconds later. `setsid` + redirect detaches them so an ssh disconnect won't
SIGHUP the server.

```bash
# deploy (from this directory, via the host)
scp start_vm1_leader.sh           p2p-host:/tmp/ && ssh p2p-host 'scp -P 2224 /tmp/start_vm1_leader.sh           ubuntu@127.0.0.1:/home/ubuntu/graph_vm.sh'
scp start_vm2_follower_headless.sh p2p-host:/tmp/ && ssh p2p-host 'scp -P 2225 /tmp/start_vm2_follower_headless.sh ubuntu@127.0.0.1:/home/ubuntu/graph_vm.sh'
scp ../common/vllm-serve-env.sh   p2p-host:/tmp/ && \
  ssh p2p-host 'scp -P 2224 /tmp/vllm-serve-env.sh ubuntu@127.0.0.1:/home/ubuntu/ ; scp -P 2225 /tmp/vllm-serve-env.sh ubuntu@127.0.0.1:/home/ubuntu/'

# launch (run each on its guest; detached, logs to /tmp/vllm_vmN.log)
ssh -p 2224 ubuntu@127.0.0.1 'cd /home/ubuntu && setsid bash graph_vm.sh </dev/null >/tmp/vllm_vm1.log 2>&1'
ssh -p 2225 ubuntu@127.0.0.1 'cd /home/ubuntu && setsid bash graph_vm.sh </dev/null >/tmp/vllm_vm2.log 2>&1'
```

`Application startup complete` in `/tmp/vllm_vm1.log` (≈420 s) means it's live
on `0.0.0.0:8000`. Watch with:
`ssh -p 2224 ubuntu@127.0.0.1 'tail -f /tmp/vllm_vm1.log'`.

## Verify

```bash
# 1. process shape — follower MUST have a Worker but NO EngineCore
ssh -p 2225 ubuntu@127.0.0.1 'pgrep -af "VLLM::|vllm serve"'   # vllm serve + VLLM::Worker, no EngineCore

# 2. end-to-end completion
ssh -p 2224 ubuntu@127.0.0.1 \
  'curl -s http://127.0.0.1:8000/v1/completions -H "Content-Type: application/json" \
     -d "{\"model\":\"qwen3-4b\",\"prompt\":\"The capital of France is\",\"max_tokens\":80,\"temperature\":0}"'

# 3. throughput
ssh -p 2224 ubuntu@127.0.0.1 'python3 - < bench.py'   # ~10 tok/s avg, peak 12.5
```

## The two blockers that actually cost us (both vLLM control-plane, not NCCL)

Every bare NCCL / torch.distributed / pynccl test passed; only full vLLM hung.
Both root causes are in vLLM's control plane, not the IB data path.

1. **`VLLM_HOST_IP` unset → MQ deadlock.** vLLM's `get_ip()` probes the
   outbound IP with `socket.connect(("8.8.8.8",80))`; in a QEMU NAT guest that
   returns `10.0.2.15`, which the peer can't reach. The MessageQueue's remote
   ZMQ socket binds to that, and the worker hangs in `wait_until_ready`'s
   `remote_socket.recv()`. **Fix:** export `VLLM_HOST_IP=10.99.0.{1,2}` (each
   guest's IB IP). Tell-tale: log shows `mq_connect_ip=10.0.2.15`; after the
   fix it's `10.99.0.1`.

2. **Follower ran the wrong entrypoint → EngineCore assert.** On this build
   (`v0.21.1.dev0+gad7125a43`, d20260522), running
   `python -m vllm.entrypoints.openai.api_server` on the follower spins up a
   full EngineCore which calls `collective_rpc("get_kv_cache_spec")` and trips
   `assert self.rpc_broadcast_mq is not None` →
   `collective_rpc should not be called on follower node`. The EngineCore dies
   and takes the follower's Worker with it; the leader's Worker then waits
   forever for rank 1 at the profile_run logits all_gather (looks like an
   all_gather hang — it isn't). **Fix:** the follower runs
   `vllm serve <model> --headless` → `cli/serve.py:run_headless()` →
   `node_rank_within_dp>0` builds only `MultiprocExecutor(monitor_workers=False)`
   + worker monitor. Worker-only, no EngineCore, no assert. The leader keeps
   `api_server` unchanged. Tell-tale: follower log shows
   `Launching vLLM ... headless multiproc executor`; process tree has no
   `VLLM::EngineCore`.

The reference doc lists four more (host OOM at 2nd guest, `NCCL_IB_HCA=mlx5_0`
mismatch, port-29500 orphans, stale-rank `strcmp(NULL)` SIGSEGV). Read it
before a fresh reproduction.

## Teardown discipline (matters — stale ranks corrupt the next run)

Always hard-kill on **both** guests before relaunching; a leftover rank stuck
in the lazy IB-connect makes the next run's RCCL topology build deref a stale
pointer and SIGSEGV in glibc `strcmp` (reference doc Blocker 6).

```bash
ssh -p 2224 ubuntu@127.0.0.1 'pkill -9 -f "vllm|VLLM::|EngineCore" ; rm -f /tmp/vllm_vm1.log'
ssh -p 2225 ubuntu@127.0.0.1 'pkill -9 -f "vllm|VLLM::|EngineCore" ; rm -f /tmp/vllm_vm2.log'
# then confirm port 29500 is free on VM1:  sudo ss -lntp | grep 29500
```

## Known limits

GDR is **achieved** in [`27b-gdr/`](27b-gdr/) (`GDR 1`; needs the unified kernel
+ `RCCL_FORCE_ENABLE_DMABUF=1`) — the host-bounce was a missing kernel patch plus
an RCCL config-parsing bug, **not** a PCIe-topology limit. TP=2 + PP=1
only. `mp` executor only (no Ray). See reference doc "What it does not test".
