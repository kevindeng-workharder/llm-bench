# p2p-ib — verified run (2026-06-03)

Qwen3-4B FP16, TP=2, cross-VM over Mellanox CX-7 RoCE (`NCCL NET=IB`), graph
mode (`FULL_DECODE_ONLY`, capture sizes 1/2/4/8). GDR disabled → every
GPU↔NIC DMA bounces through host memory.

## Outcome

`Application startup complete` on VM1 `0.0.0.0:8000` at **T≈420 s** from launch.
End-to-end `/v1/completions` returns correct text. This is the run that
confirmed the **follower-headless** fix (see README.md / start_vm2_follower_headless.sh).

## Process structure (the proof the fix is correct)

```
VM1 (leader, node-rank 0):   python -m vllm.entrypoints.openai.api_server   (Ssl)
                             VLLM::EngineCore                                (Sl)
                             VLLM::Worker_TP0                                (Sl)
VM2 (follower, node-rank 1): vllm  serve  /data/Qwen3-4B  --headless         (Ssl)
                             VLLM::Worker_TP1                                (Sl)
                             ── NO VLLM::EngineCore ──   ← the whole point
```

Before the fix, VM2 also had a `VLLM::EngineCore` that asserted
(`collective_rpc should not be called on follower node`) and died, killing
VM2's Worker and hanging VM1's Worker at the logits all_gather.

## Startup timeline (from the launch poller)

```
 80 s  VM2: "Launching vLLM ... headless multiproc executor"   ← headless path taken
160 s  VM2 Worker rank 1 → parallel_state world_size=2 rank=1 (NCCL init), waits for rank 0
260 s  VM1 Worker rank 0 → world_size=2 rank=0 (leader caught up)
280 s  both on nccl==2.27.7, rendezvous
300 s  model load begins (both workers)
320 s  safetensors 33%
360 s  Capturing CUDA graphs (decode, FULL)        ← the old deadlock scenario, passes
380 s  init engine (profile)                       ← old hang point (profile_run all_gather), passes
420 s  Application startup complete                ← live on :8000
```

## NCCL is genuinely on the IB/RoCE data path (VM1 log)

```
NCCL INFO NET/IB : Using [0]roceP3p1s0:1/RoCE [RO]; OOB enP3p1s0np0:10.99.0.1<0>
NCCL INFO Using network IB
NCCL INFO Channel 00/0 : 1[0] -> 0[0] [receive] via NET/IB/0 comm 0x... nRanks 02
NCCL INFO Channel 00/0 : 0[0] -> 1[0] [send]    via NET/IB/0 comm 0x... nRanks 02
NCCL INFO Connected all trees
NCCL INFO ncclCommInitRank_impl comm 0x... rank 0 nranks 2 ... - Init COMPLETE
```

## Network / fabric state

```
VM1  enP3p1s0np0  UP  10.99.0.1/24   (RDMA dev roceP3p1s0/1  state ACTIVE  LINK_UP)
VM2  enP3p1s0np1  UP  10.99.0.2/24   (RDMA dev roceP3p1s0/1  state ACTIVE  LINK_UP)
```

## Throughput (bench.py: N=1, max_tokens=80, temperature=0)

| run | tok/s |
|-----|-------|
| first generation (cold) | 10.10 |
| warm run 1 | **12.52** |
| warm run 2 | 10.18 |
| warm run 3 | 8.54 |
| warm run 4 | 8.80 |

avg ≈ 10 tok/s, peak 12.52. The **11.76 tok/s** reference figure sits inside
this band. The 8.5–12.5 spread is inherent to the GDR-disabled host-bounce
path: each decode step's all_reduce/all_gather stages through the riscv64
host CPU, so per-step collective latency jitters. Sample completion:

```
prompt:  "The capital of France is"
output:  " Paris. The capital of Germany is Berlin. The capital of Italy is
          Rome. The capital of Spain is Madrid. ..."   (80 completion tokens)
```

## Software versions

- vLLM `v0.21.1.dev0+gad7125a43` (built 2026-05-22) — the build whose follower
  semantics require `--headless`
- RCCL `2.27.7-HEAD:96a25b5+` (ROCm 7.2.3), HIP `7.2.53211`
- PyTorch 2.11, Python 3.13, kernel `6.19.5-p2p` (riscv64)
- Model `/data/Qwen3-4B` FP16
