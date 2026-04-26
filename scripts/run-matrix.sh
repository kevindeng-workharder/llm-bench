#!/bin/bash
# Run the full matrix. Sync launchers first, then sweep all (server × N).
set -e
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"
./scripts/sync-launchers.sh
python3 -m runner.matrix "${1:-configs/bench-matrix.yaml}"
DAY=$(date +%F)
python3 -m runner.report results/raw/ > "results/${DAY}.md"
echo "OK: report at results/${DAY}.md"
