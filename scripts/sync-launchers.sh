#!/bin/bash
# Push the launcher scripts to the VM so RemoteServer can find them at the
# paths declared in configs/bench-matrix.yaml.
#
# Usage: ./scripts/sync-launchers.sh
set -e
HERE="$(cd "$(dirname "$0")/.." && pwd)"
VM_PORT="${VM_PORT:-2222}"
VM_HOST="${VM_HOST:-ubuntu@localhost}"

ssh -p "$VM_PORT" "$VM_HOST" 'mkdir -p /home/ubuntu/llm-bench/servers/vllm /home/ubuntu/llm-bench/servers/llamacpp'
rsync -e "ssh -p $VM_PORT" -av "$HERE/servers/" "$VM_HOST:/home/ubuntu/llm-bench/servers/"
ssh -p "$VM_PORT" "$VM_HOST" 'chmod +x /home/ubuntu/llm-bench/servers/*/*.sh'
echo "OK: launchers synced to $VM_HOST:/home/ubuntu/llm-bench/servers/"
