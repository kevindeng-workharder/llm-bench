# model-cases — models verified on the unified rootfs (vllm-venv / vLLM 0.21)

A catalogue of **which model serving cases run on the one unified rootfs**
(`ubuntu-25.10-preinstalled-server-riscv64.img` + the rootfs venv
`/home/ubuntu/vllm-venv`, vLLM `v0.21.1.dev0`, ROCm 7.2.3, dual gfx1100). This
complements [`../p2p-repro/`](../p2p-repro/) — that archives the **27B-Quark
TP/PP/transport matrix**; this archives **per-model coverage** (can model X be
served from this rootfs at all, and how).

## How the rootfs is laid out

- **venv:** `/home/ubuntu/vllm-venv` lives on the **rootfs** (`/dev/root`), so it
  travels with the image. Models live on **`/data`** (`models.img`, a separate
  disk mounted at boot). So a case = *rootfs venv + a model on `/data`*.
- **shared env:** every launcher sources `/home/ubuntu/vllm-serve-env.sh` (present
  on the rootfs) for the ROCm/RCCL env. Launcher exec paths are **absolute guest
  paths**, so a launcher's location in *this* repo never affects how it runs.
- **what is NOT on this image** (it was on an older, different setup, never
  archived): `/home/ubuntu/ai-2.10`, `/data/ai-2.11` (vLLM 0.19 venvs),
  `/opt/llama` (llama.cpp), and the whole `/home/ubuntu/vllm-serve/` dir
  (`server-env.sh`, `launch-server.sh`). Launchers that depended on those were
  **retrofitted** to be self-contained on `vllm-venv` (see each case).
  - `ai-2.10` specifically was a vLLM **0.19** venv shipped as a docker **payload
    tarball** (`docker/Dockerfile` → `payload/home-ubuntu-ai-2.10.tar.gz`), not on
    any rootfs backup and now lost. gemma "ran before" on it (0.19's older attention
    predates the LDS-heavy unified-attn kernel); it now serves on `vllm-venv` (0.21)
    via the [RDNA3 LDS fix](gemma-4-e2b/apply-gfx1100-lds-fix.py), so 0.19 is moot.

## Status

| model | quant | TP | launcher (`servers/vllm/…` unless noted) | runs on this rootfs? |
|---|---|---|---|---|
| Qwen3.6-27B-Quark-W8A8-INT8 | int8 | 2 | see [`../p2p-repro/`](../p2p-repro/) (TP/PP/transport matrix) | ✅ flagship, fully benched |
| [Qwen3.6-27B-AWQ](qwen3_6-27b-awq/) | AWQ | 2 | `qwen3_6-27b-awq-graph-tp2.sh` | ✅ **verified** (served; ~56 min cold compile) |
| [gemma-4-E2B-it](gemma-4-e2b/) | bf16 | 1 | `gemma4-e2b-card1-dual.sh` | ✅ **verified** — needed an RDNA3 LDS fix ([patch](gemma-4-e2b/apply-gfx1100-lds-fix.py): TILE_SIZE 32→16 for head_dim 256); 12.4 tok/s |
| [Qwen3-4B](qwen3-4b-fp16/) | fp16 | 1 | `qwen3-4b-fp16-graph-tp1.sh` | ✅ **verified** (self-contained launcher) |
| [Qwen3.6-27B-FP8](qwen3_6-27b-fp8/) | FP8 | 2/1 | `qwen3_6-27b-fp8-*.sh` | ⏸️ **deferred** (re-pointed; FP8-on-RDNA3 uncertain, not tested) |
| [Qwen3-30B-A3B-AWQ](qwen3-30b-awq/) | AWQ | 1/2 | `qwen3-30b-awq-*.sh` | ❌ **blocked** — model not on this `/data` |

## Lessons (apply to any new case here)

1. **Every distinct model/quant pays a one-time, multi-hour CPU-bound kernel
   compile on this QEMU/TCG host** (the same "No available shared memory broadcast
   block …" wait the 27B-Quark GDN paid). It is **not** a hang — `~/.triton/cache`
   grows while GPUs sit ~0 %. Cached after, so the *second* launch is fast.
2. **Self-contained launcher pattern** — if a launcher used the missing
   `vllm-serve/` infra, inline the launch: `source /home/ubuntu/vllm-serve-env.sh`
   then `exec /home/ubuntu/vllm-venv/bin/python -m vllm.entrypoints.openai.api_server
   --model /data/<MODEL> …`. (gemma re-sourced the env; 4B was fully inlined to
   replace the missing `launch-server.sh` wrapper.)
3. **Kill cleanly between cases** — vLLM renames workers via setproctitle to
   `VLLM::Worker_*`; they **leak ~22 GiB/GPU** if not reaped. Kill by PID (or
   `pkill -9 -f 'VLLM[:]'`) **and verify** `/sys/class/drm/card*/device/mem_info_vram_used`
   drops before the next launch, or the next case OOMs at startup.

## Run any case

```bash
# (single-VM dual-GPU guest up, /data mounted — see ../p2p-repro/p2p-ib/27b-gdr/host/)
scp -P 2224 ../servers/vllm/<launcher>.sh ubuntu@127.0.0.1:/home/ubuntu/run.sh
ssh -p 2224 ubuntu@127.0.0.1 'cd ~ && setsid bash run.sh </dev/null >/tmp/vllm.log 2>&1 &'
# first launch: long cold compile (watch ~/.triton/cache grow); cached after.
```
