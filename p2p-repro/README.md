# P2P reproduction archive — riscv64 + ROCm gfx1100 + vLLM TP=2 (+ a PP=2 control)

Reproducible launchers, docs, and verified results for running vLLM
tensor-parallel (TP=2) inference on the riscv64/QEMU + ROCm 7.2.3 (gfx1100)
stack, organised by **how the two GPUs talk to each other**. Each scenario is
self-contained under its own directory. The three transport scenarios are all **TP=2**;
their **PP=2** counterparts (same transport, different parallelism mode) live in
`<transport>-pp2` dirs — see the TP×PP matrix below.

Host: `p2p-host` (10.103.11.199, AMD Turin). Stack: ROCm 7.2.3 + PyTorch 2.11 +
RCCL 2.27.7 (`96a25b5+`) + vLLM `v0.21.1.dev0+gad7125a43`, kernel `6.19.5-p2p`.

## The three scenarios

| scenario | VMs | GPU↔GPU path | NCCL transport | launcher(s) | status |
|----------|-----|--------------|----------------|-------------|--------|
| [**p2p-ib**](p2p-ib/) | 2 guests | NIC ↔ NIC over RoCE (host-bounce) | `NET=IB` | leader `api_server` + follower `vllm serve --headless` | ✅ **verified 2026-06-03** |
| [**p2p-direct**](p2p-direct/) | 1 guest | Infinity Fabric / PCIe P2P, direct | `P2P/IPC` | single `api_server` | ✅ **verified 2026-06-05** (~16 tok/s) |
| [**p2p-shm**](p2p-shm/) | 1 guest | pinned host shared memory | `SHM/direct` (split topo XML) | single `api_server` | ✅ **verified 2026-06-05** (~15 tok/s) |

**How to choose:** p2p-direct is fastest and simplest (one guest, no NIC) —
use it unless you specifically need to exercise something else. p2p-shm forces
the host-staging path (P2P unavailable / under debug). p2p-ib validates the
wire-level RDMA / RoCE path and the multi-node `mp` executor across two
isolated guests. Per the IB reference doc, single-VM Infinity Fabric is
~6–12 % faster than cross-VM IB on this hardware.

All three share the same model runtime; they differ only in the NCCL transport
and (for IB) the number of guests. The differentiator lives in each launcher's
env overrides applied *after* sourcing the shared `common/vllm-serve-env.sh`.

## The TP × PP matrix

The three scenarios above are the **TP** cells (one per transport). Their **PP** counterparts
live in `<transport>-pp2` dirs — same transport, **pipeline-parallel** (split layers by depth,
one handoff/stage) instead of tensor-parallel (split each layer, all-reduce/layer):

| transport | TP cell | PP cell (`-pp2`) |
|---|---|---|
| cross-VM IB | [p2p-ib](p2p-ib/) ✅ | [p2p-ib-pp2](p2p-ib-pp2/) ✅ — **PP WINS ~1.7×** (7.32 single / 24.95 agg vs TP 4.41 / 15.03, vllm-venv 2026-06-07; earlier 2.5× used a pre-gemv TP) |
| single-VM P2P | [p2p-direct](p2p-direct/) ✅ | [p2p-direct-pp2](p2p-direct-pp2/) ✅ — PP **~8–19 % slower** (single-VM → TP wins) |
| single-VM SHM | [p2p-shm](p2p-shm/) ✅ | [p2p-shm-pp2](p2p-shm-pp2/) ✅ — 14.40 single (≈ P2P-PP 14.21; PP transport-insensitive) < TP 15.41 |

PP trades per-layer all-reduce for one per-stage handoff, so it only pays off on a **slow
interconnect** — and the matrix shows the **crossover** cleanly: single-VM
([p2p-direct-pp2](p2p-direct-pp2/): 14.21 single) PP is ~8–19 % *slower* than TP, but cross-VM IB
([p2p-ib-pp2](p2p-ib-pp2/): 7.32 single / 24.95 agg@N=4) PP is **~1.7× faster** than TP
(4.41 / 15.03, re-benched on vllm-venv 2026-06-07) because TP's per-layer all-reduce is throttled by the slow IB link. **Rule: PP for
cross-VM/slow interconnects, TP for single-VM.** Corollary (single-VM SHM vs P2P): PP is nearly
**transport-insensitive** (PP@P2P 14.21 ≈ PP@SHM 14.40) while TP is not (TP@P2P 16.20 vs TP@SHM
15.41) — same root cause (PP sends 1 handoff/token; TP all-reduces every layer).

## Layout

```
p2p-repro/
├── README.md                 ← you are here (scenario overview + how to choose)
├── common/
│   └── vllm-serve-env.sh      shared runtime env, sourced by every launcher
│                              (deployed on each guest at /home/ubuntu/vllm-serve-env.sh)
├── p2p-ib/                    ★ fully archived
│   ├── README.md              end-to-end reproduction + the 2 vLLM blockers
│   ├── start_vm1_leader.sh    LEADER (node 0, api_server)
│   ├── start_vm2_follower_headless.sh   FOLLOWER (node 1, vllm serve --headless)
│   ├── bench.py               single-stream decode throughput
│   ├── RESULTS.md             verified run: timeline, NCCL-over-IB proof, tok/s
│   └── reference/ib-p2p-cross-vm.md   comprehensive host-side doc (VFIO, stages, 6 blockers)
├── p2p-direct/                ★ archived 2026-06-05 (P2P/IPC, ~16 tok/s — beats SHM)
│   ├── README.md  RESULTS.md  start_dual_gpu.sh  rccl-topo.xml  bench.py
├── p2p-shm/                   ★ archived 2026-06-05 (SHM/direct, ~15 tok/s)
│   ├── README.md  RESULTS.md  start_shm.sh  rccl-topo-split.xml  bench.py
├── p2p-direct-pp2/            ★ archived 2026-06-06 (p2p-direct × PP — TP wins on single-VM)
│   └── README.md  RESULTS.md   (launcher in ../servers/vllm/...-graph-pp2.sh — no topo XML)
├── p2p-ib-pp2/                ★ archived 2026-06-06 (cross-VM IB × PP — PP WINS ~1.7× over TP, re-benched vllm-venv)
└── p2p-shm-pp2/               ★ archived 2026-06-06 (single-VM SHM × PP — ≈ P2P-PP; PP transport-insensitive)
```

## Shared prerequisites (all scenarios)

- ROCm 7.2.3 prefix at `/opt/rocm-riscv-7.2.3`, vLLM venv at `/data/vllm0.21-pt2.11`.
- `common/vllm-serve-env.sh` deployed to `/home/ubuntu/vllm-serve-env.sh` on every guest.
- For single-VM scenarios: the matching `rccl-topo*.xml` (P2P vs split) next to the env.
- `HSA_FORCE_FINE_GRAIN_PCIE=1` (already in the shared env) — required for any P2P path.
- VM provisioning + VFIO binding live under `/opt/rocm-riscv-build/vm/` on the host.

## Notes on this archive

- Scripts here are **verified copies** kept under version control; they are
  deployed to the guests (paths noted in each script header). If you edit a
  guest copy, update the archived one too.
- This archive currently lives at `~/Documents/p2p-repro/` (local). It can be
  pushed to the `llm-bench` repo or copied to the host alongside
  `/opt/rocm-riscv-build/{vm,docs}` with no path changes.
- Two memories capture the hard-won IB gotchas:
  `vllm-crossvm-host-ip` and `vllm-crossvm-follower-headless`.

## Provenance

p2p-ib was brought up and benchmarked end-to-end on 2026-06-03 (`p2p-ib/RESULTS.md`),
with GDR added 2026-06-05 (`p2p-ib/27b-gdr/`). p2p-direct was verified and archived
2026-06-05 (`p2p-direct/RESULTS.md` — ~16 tok/s single-stream, beating SHM), and
p2p-shm verified the same day (`p2p-shm/RESULTS.md` — ~15 tok/s via `SHM/direct`).
All three transport scenarios are now archived and benchmarked. The PP cells are filed as
`<transport>-pp2`: [p2p-direct-pp2](p2p-direct-pp2/) (single-VM → TP wins) and
[p2p-ib-pp2](p2p-ib-pp2/) (cross-VM IB → **PP wins ~1.7×**), both added 2026-06-06 —
together they show the TP/PP crossover (fast link → TP, slow link → PP).
