# p2p-ib · 27B Quark-INT8 · graph-mode (cudagraph) test

The graph-mode counterpart to [`../27b-eager`](../27b-eager). Same model and
cross-VM RoCE path, but `--enforce-eager` is replaced with **cudagraph
`FULL_DECODE_ONLY`**. This is the test that both (a) confirms the eager
bottleneck was kernel-dispatch and (b) checks whether the Gated-DeltaNet hybrid
hits the cross-VM cudagraph deadlock.

- **Status:** ✅ verified 2026-06-03 — **no deadlock, no OOM**, decode
  **~4.5 tok/s** (≈ 15× the eager 0.29). See [RESULTS.md](RESULTS.md).

## What changes vs 27b-eager

Only the execution mode (everything else identical — model, quant, TP, IB env,
leader + headless follower):

```
-    --enforce-eager
+    --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4],"max_cudagraph_capture_size":4,"cudagraph_num_of_warmups":0}'
```

## Files

| file | role |
|------|------|
| `start_vm1_leader.sh` | LEADER (node 0) → deploy to VM1 `/home/ubuntu/graph27b_vm.sh` |
| `start_vm2_follower_headless.sh` | FOLLOWER (node 1, `vllm serve --headless`) → VM2 `/home/ubuntu/graph27b_vm.sh` |
| `RESULTS.md` | verified run + the eager↔graph comparison |
| bench / profile | reuse [`../27b-eager/bench.py`](../27b-eager/bench.py) and [`../27b-eager/profile.sh`](../27b-eager/profile.sh) — model/mode-agnostic |

## Run

```bash
# deploy (both land at /home/ubuntu/graph27b_vm.sh)
scp start_vm1_leader.sh           p2p-host:/tmp/ && ssh p2p-host 'scp -P 2224 /tmp/start_vm1_leader.sh           ubuntu@127.0.0.1:/home/ubuntu/graph27b_vm.sh'
scp start_vm2_follower_headless.sh p2p-host:/tmp/ && ssh p2p-host 'scp -P 2225 /tmp/start_vm2_follower_headless.sh ubuntu@127.0.0.1:/home/ubuntu/graph27b_vm.sh'

# launch (stop any other vLLM first to free VRAM)
ssh -p 2224 ubuntu@127.0.0.1 'cd /home/ubuntu && setsid bash graph27b_vm.sh </dev/null >/tmp/vllm_vm1.log 2>&1'
ssh -p 2225 ubuntu@127.0.0.1 'cd /home/ubuntu && setsid bash graph27b_vm.sh </dev/null >/tmp/vllm_vm2.log 2>&1'

# bench (reuse the eager test's bench)
ssh -p 2224 ubuntu@127.0.0.1 'python3 /home/ubuntu/eager_bench.py'   # decode ~4.5 tok/s
```

## What to expect

- **Startup ~675 s.** Extra phase vs eager: `Capturing CUDA graphs` (the
  deadlock-risk window). Watch for `Padding mamba page` just before — that is
  vLLM making the Gated-DeltaNet state cudagraph-compatible. No OOM at
  gpu-mem-util 0.85 with capture sizes [1,2,4].
- **Decode ~4.5 tok/s**, TTFT ~5.6 s. ~15× the eager path — cudagraph collapses
  the ~1000 per-token kernel launches into one replay, erasing the riscv64
  dispatch overhead that bottlenecked eager.
- If a build *does* deadlock at capture, the mitigation on record is
  `NCCL_LAUNCH_ORDER_IMPLICIT=1` (see project history); it was **not** needed here.

## Recommendation

For usable 27B serving on this cross-VM IB path, **graph mode is the answer**
(4.5 tok/s usable vs 0.29 eager). Eager remains the deadlock-safe fallback /
debugging mode. The 4B/graph baseline is ~10 tok/s, so 27B graph carries only
the ~2.2× model-size penalty over 4B — the model size, not the framework, is
now the limiter.
