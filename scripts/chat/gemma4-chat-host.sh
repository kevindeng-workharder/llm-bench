#!/bin/bash
# Host-side launcher: SSH into the VM and start the gemma4-chat interactive
# REPL with a proper TTY so input() works and Ctrl-C is caught by the
# Python handler rather than the SSH client.
#
# On-VM scripts:    /home/ubuntu/gemma4-chat.{py,sh}
# Repo source:      scripts/chat/on-vm/gemma4-chat.{py,sh}
# Deploy with:      bash scripts/chat/install-on-vm.sh
#
# First run takes ~4 minutes (weight load + triton JIT + cudagraph capture).
# Subsequent responses replay the captured graph, ~20 tok/s on Gemma-4-E4B.
exec ssh -t -p 2222 ubuntu@localhost 'bash /home/ubuntu/gemma4-chat.sh'
