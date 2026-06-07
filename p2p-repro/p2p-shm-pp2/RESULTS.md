# RESULTS — p2p-shm-pp2 (PP=2 single-VM, NCCL forced onto SHM transport)

The **PP cell** of the [p2p-shm](../p2p-shm) transport. Completes the TP×PP matrix.
Headline: **PP is barely transport-sensitive on single-VM** (P2P ≈ SHM), unlike TP — and PP
still loses to TP on single-VM.

**Verified:** 2026-06-06, host `10.103.11.199`, single guest, both gfx1100 GPUs.
**Model / mode:** `Qwen3.6-27B-Quark-W8A8-INT8`, **PP=2** (2 GPUs = 2 pipeline stages in one
VM), text-only, max-model-len 8192, max-num-seqs 8, cudagraph `FULL_DECODE_ONLY` `[1,2,4,8]`,
gemv INT8 patch (`/home/ubuntu/vllm-venv`).
**Transport:** `NCCL_TOPO_FILE=rccl-topo-split.xml` (fake cross-NUMA → RCCL picks SHM) +
`--disable-custom-all-reduce` — same SHM-forcing config as [p2p-shm](../p2p-shm) TP.
**Launcher:** [`start_shm_pp.sh`](start_shm_pp.sh).

## Throughput

Clean streaming decode (TTFT-cancelled, `ignore_eos`, warmed). KV pool 58,514 tok (8192 ctx,
7.14x). Output correct (`The capital of France is` → "Paris").

| metric | p2p-shm-pp2 (SHM PP) |
|---|--:|
| single-stream decode | **14.40** |
| N=4 per-client decode | 10.89 |
| N=4 aggregate decode | **43.55** |

## The two findings

**1. PP loses to TP on single-VM** — same as [p2p-direct-pp2](../p2p-direct-pp2): the fast
single-VM link makes TP's all-reduce cheap, so PP's batch=1 pipeline bubble isn't worth it.

| single-VM | P2P/IPC | SHM | P2P → SHM |
|---|--:|--:|--:|
| **TP** | 16.20 ([p2p-direct](../p2p-direct)) | 15.41 ([p2p-shm](../p2p-shm)) | **−5 %** |
| **PP** | 14.21 ([p2p-direct-pp2](../p2p-direct-pp2)) | **14.40** (this) | **+1.3 %** (≈ noise) |

**2. PP is transport-insensitive on single-VM; TP is not.** PP@P2P (14.21) ≈ PP@SHM (14.40),
but TP@P2P (16.20) is ~5 % faster than TP@SHM (15.41). Reason: PP sends **one** activation
handoff per token, so swapping the GPU↔GPU path (Infinity-Fabric P2P vs host-SHM bounce)
barely registers; TP **all-reduces every layer** (~64/token), so the slower SHM path costs it.
This is the single-VM mirror of why PP wins cross-VM ([p2p-ib-pp2](../p2p-ib-pp2)): PP's tiny
comm footprint makes it robust to a worse interconnect.

## Note on transport proof

The split topo + `--disable-custom-all-reduce` is the exact config that logged
`via SHM/direct/direct` for [p2p-shm](../p2p-shm/RESULTS.md) TP. Here the launcher runs at
`NCCL_DEBUG=WARN`, which suppresses the per-channel `via …` line, so SHM isn't grep-proven in
this run — but it's moot: PP's near-identical P2P-vs-SHM number (finding #2) shows the
transport choice barely affects PP either way.
