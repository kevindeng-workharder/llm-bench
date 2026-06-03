# p2p-shm — single-VM dual-GPU forced through host SHM transport  ⟨PLACEHOLDER⟩

> **Status: NOT yet archived.** Scaffold only — fill in after p2p-ib.

## What this scenario is

Same single guest with both gfx1100 GPUs as p2p-direct, but NCCL is steered
onto its **SHM transport** (GPU → pinned host shared memory → GPU) instead of
direct P2P. Done by feeding RCCL a **split topology XML** that says the two
GPUs are on different CPU sockets, so RCCL picks `SHM/direct/direct` rather
than `P2P/IPC`. Slower than p2p-direct but exercises the host-staging path and
sidesteps P2P entirely — useful when P2P DMA is unavailable or being debugged.

## Known assets to harvest (on the guest)

- launcher: `/home/ubuntu/start_shm_48k.sh` — **this is the working SHM script.**
  Key bits already in it:
  - `export NCCL_TOPO_FILE=/home/ubuntu/rccl-topo-split.xml`  ← the split topo that forces SHM
  - `export NCCL_SHM_DISABLE=0`
  - model: Qwen3.6-27B Quark W8A8 INT8, `--max-model-len 49152` (48K), TP=2, image mm
- topo XML: `/home/ubuntu/rccl-topo-split.xml` (the cross-socket fake) — **must be archived too**
- contrast file: the non-split `rccl-topo.xml` is what p2p-direct uses

## TODO to make this reproducible (mirror p2p-ib's structure)

- [ ] copy in `start_shm_48k.sh` (clean up, add header) + `rccl-topo-split.xml`
- [ ] confirm NCCL log shows the **SHM** transport (`via SHM/direct`), not P2P/NET
- [ ] `bench.py` + record tok/s (note: 27B INT8 @ 48K ctx, different workload than p2p-ib's 4B)
- [ ] `RESULTS.md` — process shape, NCCL-transport proof, throughput, KV/ctx budget
- [ ] document how the split topo is constructed (so it can be regenerated, not just copied)
