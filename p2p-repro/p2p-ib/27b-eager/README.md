# p2p-ib · 27B Quark-INT8 · eager-mode test

A complete, reproducible test of a **larger model in eager mode** on the
cross-VM IB path: same two-guest RoCE architecture as the parent
[p2p-ib](../README.md) scenario, but swapping Qwen3-4B/graph for
**Qwen3.6-27B-VL (Qwen3-Next, Gated-DeltaNet hybrid), Quark W8A8 INT8,
`--enforce-eager`**.

- **Status:** ✅ deploys & serves correctly; ⚠️ eager decode is **~0.29 tok/s**
  (bottleneck identified — see [RESULTS.md](RESULTS.md)).
- **Point of this test:** prove the p2p-ib path is model/mode-agnostic, and
  characterise the *eager* performance and *why* it is what it is.

## What this exercises (vs the 4B/graph baseline)

| | parent p2p-ib (baseline) | this test |
|---|---|---|
| model | Qwen3-4B FP16 | Qwen3.6-27B-VL **Quark W8A8 INT8** (29 GB) |
| exec mode | graph (cudagraph FULL_DECODE) | **`--enforce-eager`** (no cudagraph, no torch.compile) |
| max-model-len | 4096 | 2048 |
| VL model | no | **yes** → `--mm-processor-kwargs` REQUIRED (else ViT profile_run OOMs) |
| architecture (leader+headless follower) | identical | identical |

Eager is the **deadlock-safe** mode: it sidesteps the cross-VM cudagraph
deadlock that the Gated-DeltaNet hybrid would otherwise hit (the long saga in
the project history). The cost of that safety is the throughput characterised
here.

## Files

| file | role |
|------|------|
| `start_vm1_leader.sh` | LEADER (node 0) → deploy to VM1 `/home/ubuntu/eager27b_vm.sh` |
| `start_vm2_follower_headless.sh` | FOLLOWER (node 1, `vllm serve --headless`) → VM2 `/home/ubuntu/eager27b_vm.sh` |
| `bench.py` | streaming throughput — separates **TTFT (prefill)** from **decode tok/s** |
| `profile.sh` | bottleneck profiler — GPU busy% + py-spy MainThread sampling on both ranks |
| `RESULTS.md` | verified run + the bottleneck diagnosis (dispatch-bound) |

## Run the test

Prereqs: host + guests up and IB verified (see parent [p2p-ib/README.md](../README.md)
"Prerequisites"). Stop any other vLLM first to free VRAM (27B INT8 ≈ 13.5 GB/rank).

```bash
# 1. deploy (both land at /home/ubuntu/eager27b_vm.sh on their guest)
scp start_vm1_leader.sh           p2p-host:/tmp/ && ssh p2p-host 'scp -P 2224 /tmp/start_vm1_leader.sh           ubuntu@127.0.0.1:/home/ubuntu/eager27b_vm.sh'
scp start_vm2_follower_headless.sh p2p-host:/tmp/ && ssh p2p-host 'scp -P 2225 /tmp/start_vm2_follower_headless.sh ubuntu@127.0.0.1:/home/ubuntu/eager27b_vm.sh'

# 2. launch leader then follower (detached; logs to /tmp/vllm_vmN.log)
ssh -p 2224 ubuntu@127.0.0.1 'cd /home/ubuntu && setsid bash eager27b_vm.sh </dev/null >/tmp/vllm_vm1.log 2>&1'
ssh -p 2225 ubuntu@127.0.0.1 'cd /home/ubuntu && setsid bash eager27b_vm.sh </dev/null >/tmp/vllm_vm2.log 2>&1'

# Application startup complete in /tmp/vllm_vm1.log ≈ 690 s (27B load + VL init;
# eager has NO "Capturing CUDA graphs" phase).

# 3. correctness + throughput (run on VM1)
scp bench.py p2p-host:/tmp/ && ssh p2p-host 'scp -P 2224 /tmp/bench.py ubuntu@127.0.0.1:/home/ubuntu/'
ssh -p 2224 ubuntu@127.0.0.1 'python3 /home/ubuntu/bench.py'   # warmup + 3×32-tok streaming

# 4. find the bottleneck (GPU busy% + py-spy hotspots on both ranks)
scp profile.sh p2p-host:/tmp/   # then run on the host with the guest sudo password:
ssh p2p-host 'VM_SUDO_PASS=<guest-sudo-pw> bash /tmp/profile.sh'
```

## What to expect

- **Startup:** ~690 s to ready (no cudagraph capture; the time is 27B weight
  load ~13.5 GB/rank + VL encoder init). No OOM at gpu-mem-util 0.85.
- **Decode:** ~0.288 tok/s (≈3.5 s/token), TTFT ~5 s. Very consistent across runs.
- **profile.sh** should show: both Workers' MainThread `active+gil` in the
  Triton-INT8 launch path (`triton_scaled_mm` → `JITFunction.run`) and
  Gated-DeltaNet kernels; GPU busy% bimodal (one rank ~20% starved, the other
  ~100% spinning in NCCL waiting for it) and oscillating between ranks. That is
  the signature of **eager kernel-dispatch bound on the slow riscv64 host CPU**,
  not GPU-compute-bound and not primarily IB-bound. Full analysis in RESULTS.md.

## Notes / safety

- `profile.sh` runs py-spy under sudo *inside* the guest; pass the guest sudo
  password via `VM_SUDO_PASS` — it is intentionally NOT hard-coded so this tree
  stays safe to publish.
- py-spy on riscv64: Python stacks only (`--native` is unsupported); that is
  enough to localise the launch-path hotspot here.
