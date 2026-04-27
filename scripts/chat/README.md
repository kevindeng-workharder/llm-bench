# `scripts/chat/` — interactive chat & stress clients

Client-side tools for talking to a model server you've already started
(see `servers/`). Symmetric to `scripts/instruments/`, which patches the
server side.

All scripts default to `URL=http://localhost:8000`. Override with the
`URL` env var.

## Files

| File | Side | Purpose |
|---|---|---|
| `chat-llamacpp.py` | host | Streaming REPL for a llama.cpp `llama-server`. Three modes (chat / raw / completion), full sampling controls, conversation history. Default model name: `qwen36-a3b`. |
| `chat-gemma4.py` | host | Streaming REPL for a vLLM Gemma-4 server (TP=2 setup tested). |
| `stress-llamacpp.py` | host | Concurrent stress test client — fires N parallel completions and reports per-stream + aggregate throughput. Used to reproduce the multi-concurrency behaviour without going through the bench runner. |
| `gemma4-chat-host.sh` | host | Wrapper that SSHs into the VM and runs the in-process Gemma-4 LLM REPL with a TTY. The vLLM Gemma-4 path uses `LLM()` directly (not the OpenAI server) because the spec-decode + cudagraph capture is faster in-process. |
| `on-vm/gemma4-chat.py` | VM | The actual REPL that runs inside the VM (loaded by the host wrapper). Loads Gemma-4 via `vllm.LLM`, talks to it directly. |
| `on-vm/gemma4-chat.sh` | VM | Env wrapper around the on-VM Python (`LD_LIBRARY_PATH`, `TORCH_USE_RTLD_GLOBAL`, `HSA_CODE_OBJECT_CACHE`, `HIP_FORCE_DEV_KERNARG`, `USE_LIBUV=0`, etc — the full riscv+ROCm runtime env). |
| `install-on-vm.sh` | host | `scp`s the two `on-vm/*` files into `/home/ubuntu/` on the VM and `chmod +x`s them. Run once per VM, re-run after edits. |

## Typical flows

### Chat with the running llama.cpp Qwen3.6-35B server

Server side (already done, see `servers/llamacpp/qwen36-35b-mxfp4-tp1.sh`):

```bash
ssh -p 2222 ubuntu@localhost \
    'setsid nohup bash /home/ubuntu/llm-bench/servers/llamacpp/qwen36-35b-mxfp4-tp1.sh \
       < /dev/null > /tmp/llama-server.log 2>&1 & disown'
ssh -fN -p 2222 -L 8000:localhost:8000 ubuntu@localhost
```

Then on host:

```bash
python3 scripts/chat/chat-llamacpp.py
# inside the REPL:  /help  shows commands
```

### Concurrent stress test against llama.cpp

```bash
python3 scripts/chat/stress-llamacpp.py
```

(Edit the script's `N_CLIENTS` / prompts to taste — small one-off tool, not
parameterised heavily.)

### Chat with Gemma-4 (in-process LLM on VM, no OpenAI server)

First-time setup — deploy the on-VM scripts:

```bash
bash scripts/chat/install-on-vm.sh
```

Then any time:

```bash
bash scripts/chat/gemma4-chat-host.sh
# First run takes ~4 min (weight load + triton JIT + cudagraph capture).
# Subsequent replies replay the captured graph, ~20 tok/s on Gemma-4-E4B.
```

## Why these aren't part of `runner/`

`runner/` is the bench harness — non-interactive, captures structured JSON
results, drives `RemoteServer` lifecycle. `scripts/chat/` is the opposite —
interactive, no result capture, expects you to have already started the
server you want to talk to. Treat them as the "smoke test" / "feel the
model" complement to the bench.
