# Qwen3.6-27B pipeline parallelism (PP=2)

➡️ **Archived as a p2p-repro scenario.** PP=2 is a 4th "way to run the model across the two
GPUs", and its conclusion ties directly to the transport scenarios, so it lives there:

**[`p2p-repro/pp2/`](../p2p-repro/pp2/)** — [README](../p2p-repro/pp2/README.md) +
[RESULTS](../p2p-repro/pp2/RESULTS.md) (PP=2 vs TP=2 numbers, the single-VM verdict, and the
~2 h first-boot recompile gotcha).

Launcher: [`servers/vllm/qwen3_6-27b-quark-int8-graph-pp2.sh`](../servers/vllm/qwen3_6-27b-quark-int8-graph-pp2.sh).

**TL;DR:** PP works on the hybrid GDN model (`Qwen3_5` declares `SupportsPP`) but is
**~8–19 % slower than TP on single-VM** (single 14.21 vs 15.4; N=4 agg 43.6 vs ~53.7) —
fast Infinity-Fabric makes TP's all-reduce cheap and PP can't recover its batch=1 pipeline
bubble. **PP's win is slow interconnects (cross-VM IB), not single-VM.**
