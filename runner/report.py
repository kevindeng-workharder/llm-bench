"""Aggregate raw JSON bench results into a markdown report.

Usage:
    python -m runner.report results/raw/   > results/2026-04-26.md
    python -m runner.report results/raw/ --filter qwen3-30b
"""
from __future__ import annotations
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def collect(root: Path, name_filter: str | None) -> list[dict]:
    records = []
    for p in sorted(root.glob("*.json")):
        try:
            r = json.loads(p.read_text())
        except Exception as e:
            print(f"[skip] {p}: {e}", file=sys.stderr)
            continue
        if name_filter and name_filter not in r.get("config_name", ""):
            continue
        r["_path"] = str(p)
        records.append(r)
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", default="results/raw", nargs="?")
    ap.add_argument("--filter", default=None,
                    help="Only include records whose config_name contains this substring")
    args = ap.parse_args()

    root = Path(args.dir)
    records = collect(root, args.filter)
    if not records:
        print(f"No records in {root}", file=sys.stderr)
        return

    # Group by config_name → {n_clients: record}
    by_cfg: dict[str, dict[int, dict]] = defaultdict(dict)
    for r in records:
        cfg = r.get("config_name", "unknown")
        n = r["params"]["n_clients"]
        # If we have repeats, keep the latest (lex sort by ts).
        prev = by_cfg[cfg].get(n)
        if prev is None or r.get("ts", "") > prev.get("ts", ""):
            by_cfg[cfg][n] = r

    n_set = sorted({n for cfg in by_cfg.values() for n in cfg.keys()})

    print("# llm-bench results")
    print()
    print(f"Source: `{root}` — {len(records)} record(s) covering {len(by_cfg)} config(s).")
    print()

    # -- Aggregate throughput table
    print("## Aggregate throughput (tok/s)")
    print()
    header = "| config | " + " | ".join(f"N={n}" for n in n_set) + " |"
    print(header)
    print("|" + "---|" * (len(n_set) + 1))
    for cfg in sorted(by_cfg):
        cells = []
        for n in n_set:
            rec = by_cfg[cfg].get(n)
            if rec is None:
                cells.append("–")
            else:
                s = rec["summary"]
                tag = f"{s['agg_tps']}"
                g = s["garbage_clients"]
                if g:
                    tag = f"**{tag}**⚠️"
                cells.append(tag)
        print(f"| `{cfg}` | " + " | ".join(cells) + " |")
    print()

    # -- Garbage / correctness table
    print("## Correctness (ok / garbage clients)")
    print()
    print(header)
    print("|" + "---|" * (len(n_set) + 1))
    for cfg in sorted(by_cfg):
        cells = []
        for n in n_set:
            rec = by_cfg[cfg].get(n)
            if rec is None:
                cells.append("–")
            else:
                s = rec["summary"]
                cells.append(f"{s['ok_clients']-s['garbage_clients']}/{n}"
                             + (f" ({s['garbage_clients']} bad)" if s["garbage_clients"] else ""))
        print(f"| `{cfg}` | " + " | ".join(cells) + " |")
    print()

    # -- Per-client tok/s + TTFT detail
    print("## Per-config detail")
    print()
    for cfg in sorted(by_cfg):
        print(f"### `{cfg}`")
        print()
        print("| N | wall(s) | agg t/s | avg per-client t/s | ok/N | garbage |")
        print("|---|---|---|---|---|---|")
        for n in sorted(by_cfg[cfg]):
            rec = by_cfg[cfg][n]
            s = rec["summary"]
            print(f"| {n} | {s['wall_s']} | {s['agg_tps']} | "
                  f"{s['avg_per_client_tps']} | {s['ok_clients']}/{n} | "
                  f"{s['garbage_clients']} |")
        print()


if __name__ == "__main__":
    main()
