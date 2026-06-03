#!/bin/bash
# ============================================================================
# p2p-ib 27B eager — bottleneck profiler
#
# Answers "where do the 3.5 s/token go?" with two cheap, decisive signals:
#   A) GPU busy% (sysfs gpu_busy_percent) on BOTH ranks  -> compute-bound vs
#      idle/dispatch-bound vs NCCL-spin-waiting.
#   B) py-spy sampling of each Worker's MainThread        -> which Python frame
#      it sits in (kernel-launch path vs all_reduce vs compute).
#
# Run while the server (start_vm1_leader.sh + headless follower) is up. The
# script kicks its own sustained generation to create a steady decode window.
#
# REQUIRES: py-spy at $PYSPY on each guest (Python-only; --native is unsupported
#           on riscv64). Set VM_SUDO_PASS for the in-guest sudo py-spy needs.
#
# USAGE:  VM_SUDO_PASS=... bash profile.sh
# ============================================================================
set -u
: "${VM_SUDO_PASS:?set VM_SUDO_PASS to the in-guest sudo password (NOT committed)}"
SSH_OPT="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
V1_PORT="${V1_PORT:-2224}"; V2_PORT="${V2_PORT:-2225}"
GUEST="${GUEST:-ubuntu@127.0.0.1}"
PYSPY="${PYSPY:-/home/ubuntu/py-spy}"
MODEL="${MODEL:-qwen3_6-27b-int8}"
V1(){ ssh -p "$V1_PORT" $SSH_OPT "$GUEST" "$1" 2>/dev/null; }
V2(){ ssh -p "$V2_PORT" $SSH_OPT "$GUEST" "$1" 2>/dev/null; }

WP1=$(V1 "pgrep -f 'VLLM::Worker' | head -1")
WP2=$(V2 "pgrep -f 'VLLM::Worker' | head -1")
echo "VM1 Worker pid=$WP1   VM2 Worker pid=$WP2"

echo "=== kick sustained generation (160 tok) for a steady decode window ==="
V1 "nohup curl -s -m 900 http://127.0.0.1:8000/v1/completions -H 'Content-Type: application/json' \
    -d '{\"model\":\"$MODEL\",\"prompt\":\"Write a long detailed essay about computer architecture.\",\"max_tokens\":160,\"temperature\":0}' \
    >/tmp/gen_profile.out 2>&1 &"
echo "waiting 15 s for prefill to finish / decode to start..."; sleep 15
V1 "grep -E 'Avg generation throughput' /tmp/vllm_vm1.log 2>/dev/null | tail -1 | cut -c1-120"

KEYS='all_reduce|all_gather|communication_op|tensor_model_parallel|apply_weights|scaled_int8|triton_scaled_mm|RowParallel|ColumnParallel|GatedDeltaNet|gdn_attention|rms_norm|attention|sample|logits'

echo ""
echo "########## A) GPU busy% (both ranks, 20 samples @0.5s) ##########"
echo -n "VM1: "; V1 "for i in \$(seq 1 20); do cat /sys/class/drm/card0/device/gpu_busy_percent 2>/dev/null|tr '\n' ' '; sleep 0.5; done; echo"
echo -n "VM2: "; V2 "for i in \$(seq 1 20); do cat /sys/class/drm/card0/device/gpu_busy_percent 2>/dev/null|tr '\n' ' '; sleep 0.5; done; echo"
echo "  (interpretation: ~100% = NCCL all_reduce kernel spin-waiting for the"
echo "   dispatch-laggard peer; ~15-25% = GPU starved while CPU launches kernels."
echo "   Which rank is which OSCILLATES with host-CPU contention.)"

echo ""
echo "########## B) py-spy MainThread hotspots (each Worker, 10 dumps @2.5s) ##########"
for tag in VM1 VM2; do
  if [ "$tag" = VM1 ]; then WP=$WP1; RUN=V1; else WP=$WP2; RUN=V2; fi
  echo "--- $tag Worker $WP ---"
  $RUN "for i in \$(seq 1 10); do echo '$VM_SUDO_PASS' | sudo -S $PYSPY dump --pid $WP 2>/dev/null; echo '====DUMP===='; sleep 2.5; done > /tmp/dumps_$tag.txt 2>&1"
  echo -n "  MainThread state: "; $RUN "grep -E 'MainThread' /tmp/dumps_$tag.txt | sort | uniq -c | tr '\n' ' '; echo"
  echo "  top frames:"
  $RUN "grep -iE '$KEYS' /tmp/dumps_$tag.txt | sed -E 's/^[[:space:]]+//; s/\(.*$//' | sort | uniq -c | sort -rn | head -12" | sed 's/^/    /'
done
