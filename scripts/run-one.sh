#!/bin/bash
# Run one server config × one N. Convenience wrapper around runner.matrix.
#
# Usage: ./scripts/run-one.sh <server-name> [<N>]
#   $ ./scripts/run-one.sh vllm-qwen3-30b-awq-graph-tp1
#   $ ./scripts/run-one.sh vllm-qwen3-30b-awq-graph-tp1 4
set -e
SERVER="${1:?server name (see configs/bench-matrix.yaml)}"
N="${2:-}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"

if [ -n "$N" ]; then
    # Override workload to a single N
    TMP=$(mktemp --suffix=.yaml)
    python3 -c "
import yaml, sys
plan = yaml.safe_load(open('configs/bench-matrix.yaml'))
plan['workloads'] = [{'name':'one','n_clients':[$N],'max_tokens':80,'temperature':0.0,'top_p':1.0,'unique_prompts':True}]
yaml.safe_dump(plan, open('$TMP','w'))
"
    python3 -m runner.matrix "$TMP" --only-server "$SERVER"
    rm -f "$TMP"
else
    python3 -m runner.matrix configs/bench-matrix.yaml --only-server "$SERVER"
fi
