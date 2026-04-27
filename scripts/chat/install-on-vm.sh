#!/bin/bash
# Deploy the on-VM chat scripts (`gemma4-chat.py` + `gemma4-chat.sh`) to the
# VM's /home/ubuntu/. Run from the repo root.
#
# Usage:
#   ./scripts/chat/install-on-vm.sh
#
# Override target via env:
#   VM_HOST=ubuntu@localhost VM_PORT=2222 ./scripts/chat/install-on-vm.sh
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
VM_PORT="${VM_PORT:-2222}"
VM_HOST="${VM_HOST:-ubuntu@localhost}"

scp -P "$VM_PORT" "$HERE/on-vm/gemma4-chat.py" "$VM_HOST:/home/ubuntu/gemma4-chat.py"
scp -P "$VM_PORT" "$HERE/on-vm/gemma4-chat.sh" "$VM_HOST:/home/ubuntu/gemma4-chat.sh"
ssh -p "$VM_PORT" "$VM_HOST" 'chmod +x /home/ubuntu/gemma4-chat.py /home/ubuntu/gemma4-chat.sh'
echo "OK: deployed gemma4-chat.{py,sh} to $VM_HOST:/home/ubuntu/"
echo "    Now run from the host:  bash scripts/chat/gemma4-chat-host.sh"
