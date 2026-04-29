# `docker/` — riscv64 inference image (llama.cpp + vLLM)

A self-contained Docker image bundling everything you need to serve LLMs
on a `gfx1100` GPU (RX 7900 XTX / XT) — no host-side ROCm install
required. The only host requirements are:

1. Linux kernel with `amdgpu` module loaded
2. Docker (≥ 20.10)
3. Models on disk (mounted into the container as a volume — they are
   user data, not part of the "environment")

## What's bundled

| Component | Path inside image | Source |
|---|---|---|
| Python 3.13 interpreter + stdlib + system libs (libnuma/libdrm/libffi/libpci/...) + CA certs | `/usr/bin/python3.13`, `/usr/lib/python3.13`, `/usr/lib/riscv64-linux-gnu/...` | `payload/sysdeps.tar.gz` (extracted from VM, **NO apt download**) |
| ROCm 6.2.4 user-space (riscv64) — also provides `libomp.so` for torch | `/opt/rocm-riscv` (`/opt/rocm` symlink) | `payload/opt-rocm-riscv.tar.gz` |
| llama.cpp build b358 (riscv64) | `/opt/llama/{bin,lib}` | `payload/opt-llama.tar.gz` |
| Python venv: vLLM `0.19.0+rocm624` + torch `2.8.0a0+riscv64.rocm` + triton `3.4.0+gitc817b9b6` (v2-clamp NaN fix applied) | `/home/ubuntu/ai-2.10` | `payload/home-ubuntu-ai-2.10.tar.gz` |
| Reference launcher scripts | `/opt/launchers/` | `launchers/` |
| Entrypoint dispatcher | `/usr/local/bin/inference-entrypoint` | `entrypoint.sh` |

**Fully offline build**: nothing is downloaded from the internet during `docker build`. All payload comes from `payload/*.tar.gz` (extracted from the VM disk image). The base image (`ubuntu:25.10`, ~80 MB) is the only thing pulled, and it's already cached locally after the first build.

## Build

```bash
cd /home/kevin/llm-bench/docker
./build.sh
```

The build script:
1. Mounts the riscv64 VM disk image (`~/ubuntu-25.10-preinstalled-server-riscv64.img`) read-only via loopback.
2. Tars `/opt/llama`, `/opt/rocm-riscv`, `/home/ubuntu/ai-2.10` into `payload/`.
3. Calls `docker buildx build --platform=linux/riscv64 --load`.
4. Unmounts the loop device.

Use `./build.sh --no-extract` to skip step 1-2 and reuse existing tarballs.

The host needs:
- `qemu-user-binfmt` (the kernel binfmt_misc registration for riscv64)
- `docker.io` + `docker-buildx`

Both are installed by `sudo apt install docker.io docker-buildx qemu-user-binfmt binfmt-support`.

Build context is ~1.2 GB (compressed payload tarballs); final image is ~5 GB.

## Run

The image dispatches between **two engines** at runtime via either an
ENV var or a positional arg. Models live OUTSIDE the image — mount them
under `/data`.

### Quickstart: dual-server (compose)

Edit `MODELS_DIR` to point at your model folder, then:

```bash
MODELS_DIR=/data/models docker compose up -d
docker compose logs -f
```

This starts:
- `qwen-llamacpp` on `:8001` (card0, Qwen3.6-35B via llama.cpp)
- `gemma-vllm`    on `:8002` (card1, Gemma-4-E2B via vLLM)

### Single container, env-driven

```bash
# Qwen via llama.cpp on card0
docker run -d --name qwen \
  --device=/dev/kfd --device=/dev/dri/renderD128 \
  --group-add video --group-add render \
  --ipc=host --shm-size=4g \
  -v /data/models:/data:ro \
  -p 8001:8001 \
  -e ENGINE=llamacpp \
  -e MODEL=/data/Qwen3.6-35B-A3B-MXFP4_MOE.gguf \
  -e ALIAS=qwen36-a3b \
  -e GPU=0 -e PORT=8001 \
  -e PARALLEL=2 -e CTX=16384 \
  inference:riscv64

# Gemma via vLLM on card1
docker run -d --name gemma \
  --device=/dev/kfd --device=/dev/dri/renderD129 \
  --group-add video --group-add render \
  --ipc=host --shm-size=8g \
  -v /data/models:/data:ro \
  -p 8002:8002 \
  -e ENGINE=vllm \
  -e MODEL=/data/gemma-4-E2B-it \
  -e SERVED_NAME=gemma4-e2b \
  -e GPU=1 -e PORT=8002 \
  -e MAX_NUM_SEQS=2 \
  inference:riscv64
```

### Single container, positional/CLI

```bash
docker run --rm inference:riscv64 llamacpp \
    -m /data/SomeOther.gguf --port 8000 --threads 4
docker run --rm inference:riscv64 vllm \
    --model /data/some-hf-dir --port 8000
```

The first positional arg is the engine (`llamacpp` / `vllm` / `bash` / `--help`); everything after is appended to the underlying server command.

### Drop into a shell

```bash
docker run --rm -it \
  --device=/dev/kfd --device=/dev/dri \
  -v /data/models:/data \
  inference:riscv64 bash
```

Useful for poking at the venv (`python -c "import vllm; print(vllm.__version__)"`)
or running the bundled launchers manually (`/opt/launchers/qwen36-…-card0-dual.sh`).

## Verifying the API

```bash
curl -s http://localhost:8001/v1/models | jq .
curl -s http://localhost:8002/v1/models | jq .
```

For a real chat round-trip:

```bash
curl -sX POST http://localhost:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen36-a3b","messages":[{"role":"user","content":"hi"}],"max_tokens":50}' \
  | jq -r '.choices[0].message.content'
```

## Knobs (env vars)

See `entrypoint.sh` or run `docker run inference:riscv64 --help` for the full
list. Common ones:

| Var | Default | Effect |
|---|---|---|
| `ENGINE` | (required) | `llamacpp` / `vllm` / `bash` |
| `MODEL` | (required) | path under `/data` |
| `PORT` | `8000` | listen port |
| `GPU` | `0` | which GPU (0=card0 → renderD128, 1=card1 → renderD129) |
| `EXTRA_ARGS` | `""` | appended verbatim to the underlying server |
| (llama) `ALIAS` `PARALLEL` `CTX` `THREADS` `TEMP` `TOP_P` `TOP_K` `FLASH_ATTN` `CHAT_TEMPLATE` | sane defaults | passed through to llama-server |
| (vllm) `SERVED_NAME` `MAX_NUM_SEQS` `MAX_MODEL_LEN` `DTYPE` `GPU_MEM_UTIL` `TP_SIZE` `CUDAGRAPH_MODE` | sane defaults | passed through to vLLM |

## File map

```
docker/
├── Dockerfile             — image recipe (3 layers: ROCm, vLLM venv, llama.cpp)
├── entrypoint.sh          — dispatcher (parses ENV/CLI, exec's chosen server)
├── docker-compose.yml     — both servers up with one command
├── build.sh               — extract payload from VM image + buildx
├── launchers/             — reference bash launchers (kept inside image at /opt/launchers/)
├── payload/               — tarballs from VM image (gitignored, 1.2 GB total)
└── README.md              — this file
```

## Notes

- **Why riscv64?** The current research stack is riscv64 (ROCm cross-compiled, vLLM patched for riscv triton-free V1 engine). This image preserves that layout. An x86_64 native variant is straightforward to add later by swapping the base image and rebuilding the venv with upstream wheels.

- **`HIP_VISIBLE_DEVICES` vs `ROCR_VISIBLE_DEVICES`**: setting `HIP_VISIBLE_DEVICES` alone makes torch see zero devices on this riscv build. Always use `ROCR_VISIBLE_DEVICES=N` for vLLM. The entrypoint handles this for you.

- **vLLM v2-clamp NaN fix** is already applied in the bundled venv. If you rebuild from scratch using upstream vLLM, you'll need to re-apply the patch (see `kevindeng-workharder/vllm-riscv` on GitHub for the working tree).

- **`--reasoning-format none`** keeps Qwen3.6's `<think>...</think>` blocks in the `content` field instead of the separate `reasoning_content` field. This matters because LangChain's `ChatOpenAI` only reads `content`. The entrypoint sets this by default for llama.cpp.
