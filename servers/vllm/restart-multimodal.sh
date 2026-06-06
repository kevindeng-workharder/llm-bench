#!/bin/bash
# Hard-restart the multimodal server: kill leaked workers, wait for VRAM to free,
# relaunch the sibling launcher.
#
# vLLM renames its TP worker processes via setproctitle to "VLLM::...", so a plain
# pkill of the python path leaves them alive holding VRAM — must match VLLM:: too,
# else the relaunch OOMs on a still-occupied card.
DIR="$(cd "$(dirname "$0")" && pwd)"
pkill -9 -f 'VLLM::' 2>/dev/null || true
pkill -9 -f 'vllm.entrypoints.openai' 2>/dev/null || true
# wait for the processes to actually exit
for i in $(seq 1 90); do
  n=$(ps aux | grep -iE 'VLLM::|vllm.entrypoints.openai' | grep -v grep | wc -l)
  [ "$n" -eq 0 ] && break
  sleep 1
done
# wait for VRAM to drop below 1 GiB (workers can linger a moment after exit)
for i in $(seq 1 40); do
  u=$(cat /sys/class/drm/card0/device/mem_info_vram_used 2>/dev/null || echo 0)
  [ "$u" -lt 1073741824 ] && break
  sleep 1
done
sleep 2
exec bash "$DIR/qwen3_6-27b-quark-int8-graph-tp2-multimodal.sh"
