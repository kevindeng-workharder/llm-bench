# RESULTS — p2p-ib 27B graph + GDR

**Verified:** 2026-06-05, host `10.103.11.199`.
**Kernel:** `Image-6.19.5-p2p-all` (uname `6.19.5-p2p-all`) on both guests —
patched `qemu_soc` source (`cpu_supports_p2pdma()→true`) + the proven `-p2p-ib`
`.config` base (`PCI_P2PDMA=y` `HSA_AMD_P2P=y` `ZONE_DEVICE=y` + full IB),
gcc-15.2. Modules installed into both guest rootfs images.
**Model / mode:** `Qwen3.6-27B-Quark-W8A8-INT8`, TP=2, cudagraph
`FULL_DECODE_ONLY` + `--no-async-scheduling` (identical to `27b-graph`).
**The one runtime change vs `27b-graph`:** `RCCL_FORCE_ENABLE_DMABUF=1`.

## Before / after

| | `27b-graph` (host-staged) | **`27b-gdr` (this)** |
|---|---|---|
| NCCL transport | `Connected all rings, use ring PXN 0 GDR **0**` | `... use ring PXN 0 GDR **1**` |
| `DMA_BUF_SUPPORT` | `Failed due to OS kernel support` | force-enabled, engaged |
| GPU↔NIC DMA | bounces through host memory | **direct to/from VRAM** |
| steady decode (N=1) | ~5.2 tok/s | **~5.4–6.2 tok/s** (peak 6.2) |
| kernel | `Image-6.19.5-p2p-ib` (stock tree) | `Image-6.19.5-p2p-all` (patched + IB) |

## Re-test 2026-06-07 (vllm-venv) — caught a real bug: GDR was silently OFF in the vLLM launcher

Re-benching on vllm-venv (TTFT-cancelled, same method as `27b-graph` / `pp2`) exposed that **GDR-1 was
not engaging from the vLLM launcher** — `use ring PXN 0 GDR 0`, despite the unified kernel +
`RCCL_FORCE_ENABLE_DMABUF=1`. But [`tools/run_nccl_test.sh`](tools/run_nccl_test.sh) (pure 2-node NCCL,
no model) **did** get `GDR 1` / `via NET/IB/0/GDRDMA` on the *same* setup. The difference was two env
vars the test sets but the vLLM launcher lacked: **`NCCL_NET_GDR_LEVEL=SYS`** (the cross-root GPU↔NIC
pair exceeds NCCL's default GDR distance, so GDR is off unless forced to SYS) and
**`NCCL_DMABUF_ENABLE=1`**. Added both → vLLM now logs `use ring PXN 0 GDR 1` + `via NET/IB/0/GDRDMA`.

| metric (vllm-venv, TTFT-cancelled, 2048/seqs8) | `27b-graph` (GDR 0) | **`27b-gdr` (GDR 1, fixed)** |
|---|--:|--:|
| single-stream | 4.41 | **5.60** |
| N=4 aggregate | 15.03 | **19.89** |

GDR-1 gives TP **+27 % single / +32 % N=4** over host-bounced GDR-0 — a cleaner win than the 06-05 note
(~5.4–6.2 vs ~5.2, avg-throughput metric) once the env bug is fixed and the baseline is consistent.
PP still beats even GDR-1 TP (pp2 7.32 / gdr 5.60 = **1.31×** single — PP's 1-handoff/token comm wins on
single-stream; see [../../p2p-ib-pp2/RESULTS.md](../../p2p-ib-pp2/RESULTS.md)). Likely the 06-05 launcher
carried these envs and a later cleanup dropped them to "just `RCCL_FORCE_ENABLE_DMABUF=1`".

## Proof 1 — fast 2-node RoCE all-reduce (`tools/run_nccl_test.sh`, no model load)

```
ubuntu:11610 [0] NCCL WARN DMA_BUF Support is force enabled, so explicitly setting RCCL_FORCE_ENABLE_DMABUF=1
ubuntu:11610 [0] NCCL INFO NET/IB : Using [0]roceP3p1s0:1/RoCE [RO]; OOB enP3p1s0np0:10.99.0.1<0>
[nccl_alltest] RANK 0 ALLREDUCE_OK sum0=3.0 expect=3.0
ubuntu:11610 [0] NCCL INFO Connected all rings, use ring PXN 0 GDR 1
```

Both ranks: `GDR 1` + `ALLREDUCE_OK` (rank0 ones=1, rank1 ones=2, reduced sum = 3.0).
Data is correct *and* the path is direct — not just an advertised capability.

## Proof 2 — full 27B cross-VM vLLM

`/tmp/vllm_vm1.log` (leader):
```
rocmwrap.cc:157 NCCL WARN DMA_BUF Support is force enabled, so explicitly setting RCCL_FORCE_ENABLE_DMABUF=1
NCCL INFO NET/IB : Using [0]roceP3p1s0:1/RoCE [RO]; OOB enP3p1s0np0:10.99.0.1<0>
NCCL INFO Connected all rings, use ring PXN 0 GDR 1
INFO: Application startup complete.   (~16 min cold start)
```

Completion (`max_tokens=160`, temp 0): `"The ocean is the lifeblood of our planet, coverin..."` — correct.

Steady-state `Avg generation throughput` (single stream, as decode ramps):
```
2.0 -> 5.3 -> 6.2 -> 6.0 -> 5.4 tokens/s     # settles ~5.4–6.2
```
End-to-end incl. prefill: 256 tok / 51.4 s ≈ 4.98 tok/s.

## Diagnostic trail (how the two bugs were isolated)

1. On `-p2p-ib` (and initially on `-p2p-all`): `DMA_BUF_SUPPORT Failed`, `GDR 0`.
2. `tools/dmabuf_test.c` → `ibv_reg_dmabuf_mr(fd=-1)` returns **`EBADF`**, not
   `EOPNOTSUPP` → the mlx5/uverbs dma-buf *capability is present*. So "Failed"
   was not a missing capability.
3. Read RCCL source (`rocmwrap.cc`, on host
   `/opt/rocm-riscv-build/src/ROCm-RCCL-7.2.3/`): the config-file check reads the
   gzipped `/proc/config.gz` as plaintext and `break`s before `/boot/config`
   → false negative. `RCCL_FORCE_ENABLE_DMABUF=1` skips it.
4. With the env var set but on the *stock* `-p2p-ib` kernel, the *real*
   registration would still fail at `pci_p2pdma_distance()<0` — hence the
   unified kernel (the `cpu_supports_p2pdma` hack) is also required. The two are
   independent: env var unblocks RCCL's *decision*, the kernel hack unblocks the
   *actual* P2P registration. Both present → `GDR 1`.

## Gotchas hit (record for the next reproduction)

- **IB IP doesn't auto-assign.** mlx5 throws a transient `synd 0x10: High
  temperature` health assert during init and the link only goes ACTIVE at
  ~178 s — after `systemd-networkd` ran — so the persisted IP is never applied.
  Assign manually: `10.99.0.1` on `enP3p1s0np0` (VM1), `10.99.0.2` on
  `enP3p1s0np1` (VM2 — port 1, hence **np1**).
- **`/data` (model image) needs `-o ro,norecovery`** — it's attached read-only,
  so ext4 can't replay its journal; a plain `mount` fails.
- **Launching 2 nodes:** no foreground `sleep` between the leader/follower `ssh`
  calls (some harnesses block it → second node never launches); `pkill` patterns
  for the venv python must use the `[v]llm` bracket trick or pkill matches its
  own shell.
