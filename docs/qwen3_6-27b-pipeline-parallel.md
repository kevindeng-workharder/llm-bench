# Qwen3.6-27B pipeline parallelism (PP=2)

➡️ **Archived under p2p-repro as the PP cells of the transport matrix.** The three transport
scenarios (p2p-ib / p2p-direct / p2p-shm) are TP=2; their PP=2 counterparts are filed as
`<transport>-pp2`:

- **[`p2p-repro/p2p-direct-pp2/`](../p2p-repro/p2p-direct-pp2/)** ✅ — single-VM P2P × PP
  ([README](../p2p-repro/p2p-direct-pp2/README.md) /
  [RESULTS](../p2p-repro/p2p-direct-pp2/RESULTS.md)): PP **~8–19 % slower than TP** on
  single-VM (14.21 single / 43.62 agg @ N=4), plus the ~2 h first-boot recompile gotcha.
- **`p2p-repro/p2p-ib-pp2/`** ☐ — cross-VM IB × PP, the cell where PP *may* beat TP (to run).

Launcher: [`servers/vllm/qwen3_6-27b-quark-int8-graph-pp2.sh`](../servers/vllm/qwen3_6-27b-quark-int8-graph-pp2.sh).

**TL;DR:** PP works on the hybrid GDN model (`Qwen3_5` declares `SupportsPP`) but is ~8–19 %
slower than TP on single-VM (fast Infinity-Fabric → TP's all-reduce is cheap; PP adds a
batch=1 pipeline bubble). PP's win is slow interconnects (cross-VM IB) — see
[`p2p-repro/p2p-direct-pp2/`](../p2p-repro/p2p-direct-pp2/).
