#!/bin/bash
# Run inside each VM after fresh boot. Idempotent.
# Args: $1 = my IP/30 on Mellanox iface (10.99.0.1/30 for VM1, 10.99.0.2/30 for VM2)
# Args: $2 = Mellanox iface name (enP3p1s0np0 for VM1, enP3p1s0np1 for VM2)
set -eu
MY_IP=$1
MELL_IF=$2

# 1) Mount /data (read-only, skip journal replay since both VMs share it)
if ! mount | grep -q ' /data '; then
  sudo mount -o ro,noload /dev/sdb /data
fi
ls /data/vllm0.21-pt2.11/bin/python >/dev/null

# 2) Disable IPv6 everywhere (Gloo IPv4/IPv6 family-mismatch protection)
sudo sysctl -qw net.ipv6.conf.all.disable_ipv6=1
sudo sysctl -qw net.ipv6.conf.default.disable_ipv6=1
sudo sysctl -qw net.ipv6.conf.lo.disable_ipv6=1
for iface in $(ls /proc/sys/net/ipv6/conf/ | grep -v ^all$ | grep -v ^default$); do
  sudo sysctl -qw net.ipv6.conf.${iface}.disable_ipv6=1 2>/dev/null || true
done
# flush any addresses already learnt
sudo ip -6 addr flush scope link 2>/dev/null || true
sudo ip -6 addr flush scope global 2>/dev/null || true
sudo ip -6 addr flush scope site 2>/dev/null || true

# 3) Assign IPv4 to Mellanox iface (idempotent)
if ! ip addr show "$MELL_IF" | grep -q "${MY_IP%/*}/"; then
  sudo ip addr add "$MY_IP" dev "$MELL_IF"
fi
sudo ip link set "$MELL_IF" up

# 4) Show final state
echo === final ===
ip -br addr | grep -vE '^lo|^sit0'
echo === sysctl ipv6 ===
sysctl net.ipv6.conf.all.disable_ipv6
