# Qwen3.6-27B quantization bench (riscv64 + ROCm gfx1100, dual 7900 XTX, TP=2)

Three quantizations of **Qwen3.6-27B** (`qwen3_5`: hybrid linear + full attention,
multimodal checkpoint run text-only) benched via the serve + concurrent-client
harness on the dual-GPU QEMU-riscv VM (`vllm 0.19.1.dev0+rocm723`, graph mode,
`max_model_len=2048`, `max_num_seqs=8`, concurrent-sweep N=1/2/4/8).

## Results (aggregate tok/s)

| quant | N=1 | N=2 | N=4 | N=8 | status |
|---|---|---|---|---|---|
| **AWQ int4** (SHM) | **8.78** (ttft 4.8s) | **11.66** | **15.4** | **33.98** | ✅ 0 garbage, clean scaling |
| **FP8 W8A8** (socket) | 0.6 (ttft 116s) | crash | crash | crash | RCCL allgather timeout |
| **Quark INT8 W8A8** | — | — | — | — | serve won't start (see #3) |

FP8 was only run on the socket transport (before the SHM fix); with SHM it is
expected to behave like AWQ. AWQ is the only config benched end-to-end with SHM.

## Three findings

### 1. First compile is slow (~60 min), but the cache is persistent
`qwen3_5`'s linear-attention layers are Gated DeltaNet — first run JIT-compiles
`chunk_gated_delta_rule`, `chunk_scaled_dot_kkt`, `recompute_w_u` (+ quant GEMM)
triton kernels. On QEMU-riscv each takes 1–5 min; first serve startup was ~61 min
(`init engine ... took 3657s`). The cache lives in the **rootfs** at
`~/.triton/cache` (`/dev/sda1`, persistent across qemu restarts), so subsequent
starts are fast — verified the 267 FP8-era kernels survived two qemu restarts.

### 2. socket → SHM is the key for TP=2 concurrency
`vllm-serve-env.sh` forces `NCCL_P2P_DISABLE=1` **and** `NCCL_SHM_DISABLE=1`, so
RCCL falls back to TCP-loopback socket. Under concurrent allgather (N>=2) the
socket path deadlocks → 600s NCCL watchdog → `c10::DistBackendError` →
`EngineDeadError`. Even N=1 is crippled (TP=2 prefill allgather: ttft 116s).

Overriding `NCCL_SHM_DISABLE=0` (host-RAM ferry between TP workers) fixes it:
AWQ then runs N=1..8 cleanly, 8.78 → 33.98 t/s, ttft drops 116s → 4.8s. Same fix
as `docker/launchers/server-env.sh`'s default. All `qwen3_6-27b-*` launchers now
set this override.

### 3. Quark INT8 W8A8 per-channel needs vLLM >= 0.21
0.19's quark backend raises `NotImplementedError("No quark compatible scheme")`
for this checkpoint's scheme (weight `int8 per_channel static symmetric` +
activation `int8 per_channel dynamic`). vLLM main/0.21 added
`_is_dynamic_per_token_w8a8`, whose five conditions match this scheme exactly
→ `QuarkW8A8Int8(is_static_input_scheme=False)`. So an upgrade unblocks it — but
needs a fresh riscv64 + ROCm gfx1100 cross-compile of vLLM 0.21 (not `pip install`),
plus rebasing the RiVAI riscv patches across 0.19 → 0.21.

## Launchers / harness changes
- `servers/vllm/qwen3_6-27b-fp8-{graph-tp2,eager-tp2,graph-tp1}.sh`,
  `qwen3_6-27b-awq-graph-tp2.sh`, `qwen3_6-27b-quark-int8-graph-tp2.sh`
- All are self-contained: source `vllm-serve-env.sh`, then override
  `NCCL_SHM_DISABLE=0` + long `VLLM_RPC_TIMEOUT` (guards EngineCore against being
  killed mid-compile on first run), force text-only via `--limit-mm-per-prompt 0/0`.
- `runner/server.py`: `VM_HOST`/`VM_PORT` are now env-overridable (this VM
  forwards guest SSH on **2224**, not the default 2222).
