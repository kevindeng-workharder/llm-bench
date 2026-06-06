#!/bin/bash
# Guest video-decode dependencies for the multimodal launcher. Run ON the riscv VM.
#
# Why pyav and not opencv: vLLM's default "opencv" video loader needs
# python3-opencv, which on this riscv apt mirror pulls libmysqlclient24 — which is
# unfetchable (mirror gap). So the launcher uses the pyav decode path instead
# (--media-io-kwargs '{"video":{"backend":"pyav"}}'), backed by python3-av here.
# python3-av in turn needs libcaca0, whose apt-pinned version 404s on the mirror
# (superseded point release), so we fetch the current .deb directly first.
#
# Verified working set (2026-06-06): python3-av 14.2.0-1ubuntu1,
# libcaca0 0.99.beta20-5ubuntu0.25.10.2. See docs/qwen3_6-27b-multimodal.md.
#
# Uses sudo — run as a user with sudo on the guest (do not hard-code credentials).
set -eu
VENV=/home/ubuntu/vllm-venv
PYSITE="$VENV/lib/python3.13/site-packages"

# 1. libcaca0 — apt's pinned version 404s on the riscv mirror; pull the current
#    .deb straight from ports.ubuntu.com (bump CACA_VER if it has moved on again).
CACA_VER=0.99.beta20-5ubuntu0.25.10.2
wget -O /tmp/libcaca0.deb \
  "http://ports.ubuntu.com/ubuntu-ports/pool/main/libc/libcaca/libcaca0_${CACA_VER}_riscv64.deb"
sudo dpkg -i /tmp/libcaca0.deb

# 2. python3-av — system package. --no-install-recommends keeps it off the
#    python3-opencv / libmysqlclient24 chain that can't be fetched.
sudo apt-get update -y
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends python3-av

# 3. Expose the system `av` to the vLLM venv — there is no riscv `av` wheel, so we
#    reuse apt's build via a symlink into site-packages.
ln -sfn /usr/lib/python3/dist-packages/av "$PYSITE/av"

# 4. Verify it imports inside the venv.
"$VENV/bin/python" -c "import av; print('pyav', av.__version__)"
