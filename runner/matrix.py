"""Run a benchmark matrix from a YAML config.

Each `(server_config × workload × n_clients)` produces a JSON record under
results/raw/. Server lifecycle is automatic — the runner brings each server
up, runs all configured Ns against it, and tears down before moving on.

Usage:
    python -m runner.matrix configs/bench-matrix.yaml
    python -m runner.matrix configs/bench-matrix.yaml --only-server vllm-qwen3-30b-awq-graph-tp1
    python -m runner.matrix configs/bench-matrix.yaml --dry-run
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from runner.server import RemoteServer  # noqa: E402
from runner import bench  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results" / "raw"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path)
    ap.add_argument("--only-server", default=None,
                    help="Run just this server config (matches `name:` field)")
    ap.add_argument("--only-workload", default=None,
                    help="Run just this workload config")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    plan = yaml.safe_load(args.config.read_text())
    servers = plan["servers"]
    workloads = plan["workloads"]

    if args.only_server:
        servers = [s for s in servers if s["name"] == args.only_server]
    if args.only_workload:
        workloads = [w for w in workloads if w["name"] == args.only_workload]

    print(f"[matrix] {len(servers)} server(s) × {len(workloads)} workload(s)",
          file=sys.stderr)

    if args.dry_run:
        for s in servers:
            for w in workloads:
                for n in w["n_clients"]:
                    print(f"  WOULD RUN: {s['name']} × {w['name']}.N={n}")
        return

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for s in servers:
        print(f"\n[matrix] === server: {s['name']} ===", file=sys.stderr)
        srv = RemoteServer(
            name=s["name"],
            launch_script_remote_path=s["launch_remote_path"],
            served_model_name=s["served_model_name"],
            ready_timeout_s=s.get("ready_timeout_s", 900),
        )
        try:
            with srv:
                # warmup once (N=1, ignored for stats but loads CUDA graph caches etc)
                if s.get("warmup", True):
                    print(f"[matrix] warming up {s['name']}…", file=sys.stderr)
                    bench.run(srv.url, s["served_model_name"], n=1, max_tokens=20,
                              temperature=0.0, top_p=1.0, unique_prompts=False)

                for w in workloads:
                    for n in w["n_clients"]:
                        print(f"[matrix] {s['name']} × {w['name']}.N={n}", file=sys.stderr)
                        rec = bench.run(
                            url=srv.url,
                            model=s["served_model_name"],
                            n=n,
                            max_tokens=w.get("max_tokens", 80),
                            temperature=w.get("temperature", 0.0),
                            top_p=w.get("top_p", 1.0),
                            unique_prompts=w.get("unique_prompts", True),
                        )
                        rec["config_name"] = s["name"]
                        rec["workload"] = w["name"]
                        ts = time.strftime("%Y%m%d-%H%M%S")
                        out = RESULTS_DIR / f"{s['name']}.{w['name']}.N{n}.{ts}.json"
                        out.write_text(json.dumps(rec, indent=2, ensure_ascii=False))
                        sm = rec["summary"]
                        print(f"  -> {out.name}  agg={sm['agg_tps']}t/s  "
                              f"garbage={sm['garbage_clients']}/{n}", file=sys.stderr)
        except Exception as e:
            print(f"[matrix] {s['name']} FAILED: {e}", file=sys.stderr)
            srv.tail_log(80)
            continue


if __name__ == "__main__":
    main()
