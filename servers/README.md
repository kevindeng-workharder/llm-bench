# Server launchers

Each `.sh` here is a self-contained recipe to start ONE backend in ONE
configuration on the riscv64 VM. They get rsync'd to the VM by
`scripts/sync-launchers.sh` and invoked by `runner/server.py`.

Naming convention:

```
<engine>/<model>-<quant>-<compile-mode>-tp<N>[-<flag>].sh
```

Examples:

- `vllm/qwen3-30b-awq-graph-tp1.sh`
- `vllm/qwen3-30b-awq-eager-tp1.sh`
- `vllm/qwen3-30b-awq-eager-tp1-serial.sh`     (max-num-seqs=1 workaround)
- `vllm/qwen3-30b-awq-graph-tp2.sh`            (dual-GPU)
- `vllm/qwen3-4b-fp16-graph-tp1.sh`
- `llamacpp/qwen36-35b-mxfp4-tp1.sh`
- `llamacpp/qwen36-35b-mxfp4-tp2.sh`

Each script must:

1. `set -e`
2. Source `/home/ubuntu/vllm-serve/server-env.sh` for vLLM, or set
   `LD_LIBRARY_PATH` directly for llama.cpp.
3. `exec` the actual server (NO backgrounding — the runner uses `setsid nohup`
   to detach).
4. Bind `0.0.0.0:8000` so the SSH port-forward picks it up.
5. Be deterministic — no random ports, no random tmpdirs.
