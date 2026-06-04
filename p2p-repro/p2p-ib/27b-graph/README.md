# p2p-ib · 27B Quark-INT8 · graph-mode (cudagraph) test

The graph-mode counterpart to [`../27b-eager`](../27b-eager): same model and
cross-VM RoCE path, `--enforce-eager` replaced with cudagraph
`FULL_DECODE_ONLY` **plus `--no-async-scheduling`** (required — see below).

- **Status:** ✅ verified 2026-06-04 — stable, **~3.0 tok/s** single-stream /
  **~11 tok/s aggregate at N=4** (136 reqs, 0 hang over a 20-min soak). See
  [RESULTS.md](RESULTS.md).
- **Why this exists:** (a) cudagraph confirms the eager bottleneck was CPU
  kernel-dispatch (~15× faster raw); (b) it surfaces a real vLLM bug —
  async-scheduling hangs under cudagraph — and its fix.

## The two changes vs 27b-eager

```
-    --enforce-eager
+    --no-async-scheduling
+    --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4],"max_cudagraph_capture_size":4,"cudagraph_num_of_warmups":0}'
```

`--no-async-scheduling` is **not optional** here. Without it, plain graph hangs
on the first sustained request: vLLM async-scheduling copies each step's sampled
tokens D2H asynchronously and waits on a CUDA event (`async_copy_ready_event`)
that **never signals under cudagraph** on this ROCm/riscv64 stack →
`.synchronize()` spins forever, both GPUs pegged at 100%. (async scheduling is
auto-enabled for the `mp` backend, so you must explicitly turn it off.) Full
diagnosis + the py-spy stacks are in RESULTS.md.

## Files

| file | role |
|------|------|
| `start_vm1_leader.sh` | LEADER (node 0) → deploy to VM1 `/home/ubuntu/graph27b_vm.sh` |
| `start_vm2_follower_headless.sh` | FOLLOWER (node 1, `vllm serve --headless`) → VM2 `/home/ubuntu/graph27b_vm.sh` |
| `RESULTS.md` | root-cause (async-scheduling × cudagraph) + N=1/N=4 results |
| bench / profile | reuse [`../27b-eager/bench.py`](../27b-eager/bench.py) and [`../27b-eager/profile.sh`](../27b-eager/profile.sh) |

## Run

```bash
# deploy (both land at /home/ubuntu/graph27b_vm.sh)
scp start_vm1_leader.sh           p2p-host:/tmp/ && ssh p2p-host 'scp -P 2224 /tmp/start_vm1_leader.sh           ubuntu@127.0.0.1:/home/ubuntu/graph27b_vm.sh'
scp start_vm2_follower_headless.sh p2p-host:/tmp/ && ssh p2p-host 'scp -P 2225 /tmp/start_vm2_follower_headless.sh ubuntu@127.0.0.1:/home/ubuntu/graph27b_vm.sh'

# launch (stop any other vLLM first to free VRAM)
ssh -p 2224 ubuntu@127.0.0.1 'cd /home/ubuntu && setsid bash graph27b_vm.sh </dev/null >/tmp/vllm_vm1.log 2>&1'
ssh -p 2225 ubuntu@127.0.0.1 'cd /home/ubuntu && setsid bash graph27b_vm.sh </dev/null >/tmp/vllm_vm2.log 2>&1'

# single-stream bench (reuse the eager test's bench)
ssh -p 2224 ubuntu@127.0.0.1 'python3 /home/ubuntu/eager_bench.py'     # ~3.0 tok/s
# concurrency is where it shines — N=4 ThreadPoolExecutor → ~11 tok/s aggregate
```

## What to expect

- **Startup ~11–30 min**, slow *and variable*. The long pole is the Qwen3-VL
  multimodal processor load (`from_pretrained` → recursive `deepcopy` of the HF
  config, pathologically slow on riscv64). No `Capturing CUDA graphs` deadlock,
  no OOM. Just be patient / cache the processor if startup latency matters.
- **Decode:** ~3.0 tok/s single-stream, **~11 tok/s aggregate at N=4** — stable
  (136 requests, 0 hang in a 20-min N=4 soak). Without `--no-async-scheduling`
  it would hang on the first sustained request.

## Recommendation

**graph + `--no-async-scheduling`, served with concurrency** is the usable 27B
config on this path (~11 tok/s aggregate at N=4, ~10× eager, stable). Eager is
the slow-but-simple fallback. Proper upstream fix = make the async-scheduling
copy event work under cudagraph on ROCm; until then, keep async scheduling off.
