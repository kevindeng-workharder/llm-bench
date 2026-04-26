"""Concurrent benchmark client.

Hits an OpenAI-compatible /v1/chat/completions endpoint with N parallel
streaming requests, measures TTFT, per-request tok/s, aggregate tok/s, and
flags degenerate "garbage" outputs (single-token loops like `!!!!!`).

Output: one JSON record per (config, n_clients) suitable for aggregation.
"""
from __future__ import annotations
import argparse
import concurrent.futures as cf
import json
import os
import socket
import sys
import time
from pathlib import Path

import httpx

# Allow `python -m runner.bench ...` from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from workloads.prompts import PROMPTS  # noqa: E402


def is_garbage(text: str) -> bool:
    """Detect degenerate outputs: a single token (or a tiny set) repeated."""
    if not text or len(text) < 20:
        return False
    stripped = text.replace(" ", "").replace("\n", "")
    return len(set(stripped)) < 4


def one_request(url: str, model: str, client_id: int, prompt: str,
                max_tokens: int, temperature: float, top_p: float) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": True,
    }
    t0 = time.time()
    t_first = None
    n_chunks = 0
    full = []
    try:
        with httpx.stream("POST", f"{url}/v1/chat/completions",
                          json=body, timeout=600) as r:
            if r.status_code != 200:
                return {"id": client_id, "ok": False,
                        "err": f"HTTP {r.status_code}: {r.read().decode()[:200]}",
                        "elapsed": time.time() - t0}
            for line in r.iter_lines():
                if not line.startswith("data: "):
                    continue
                chunk = line[len("data: "):]
                if chunk.strip() == "[DONE]":
                    break
                try:
                    obj = json.loads(chunk)
                    delta = obj["choices"][0]["delta"]
                    txt = delta.get("content", "") or delta.get("reasoning_content", "")
                    if txt:
                        if t_first is None:
                            t_first = time.time() - t0
                        full.append(txt)
                        n_chunks += 1
                except Exception:
                    pass
    except Exception as e:
        return {"id": client_id, "ok": False, "err": str(e),
                "elapsed": time.time() - t0}

    elapsed = time.time() - t0
    output = "".join(full)
    return {
        "id": client_id, "ok": True,
        "prompt_short": prompt[:50],
        "n_chunks": n_chunks,
        "ttft_s": t_first,
        "elapsed_s": elapsed,
        "tps": n_chunks / elapsed if elapsed > 0 else 0,
        "garbage": is_garbage(output),
        "output_head": output[:80].replace("\n", " "),
    }


def run(url: str, model: str, n: int, max_tokens: int, temperature: float,
        top_p: float, unique_prompts: bool) -> dict:
    if unique_prompts:
        prompts = [PROMPTS[i % len(PROMPTS)] for i in range(n)]
    else:
        prompts = [PROMPTS[0]] * n  # all identical

    t_start = time.time()
    with cf.ThreadPoolExecutor(max_workers=n) as ex:
        futs = [ex.submit(one_request, url, model, i, prompts[i],
                          max_tokens, temperature, top_p) for i in range(n)]
        results = [f.result() for f in cf.as_completed(futs)]
    wall_s = time.time() - t_start
    results.sort(key=lambda r: r["id"])

    okish = [r for r in results if r["ok"]]
    total_chunks = sum(r.get("n_chunks", 0) for r in results)
    n_garbage = sum(1 for r in results if r.get("garbage"))
    return {
        "params": {
            "n_clients": n,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "unique_prompts": unique_prompts,
            "url": url,
            "model": model,
        },
        "host": socket.gethostname(),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "results": results,
        "summary": {
            "wall_s": round(wall_s, 2),
            "total_tokens": total_chunks,
            "agg_tps": round(total_chunks / wall_s if wall_s > 0 else 0, 2),
            "avg_per_client_tps": round(
                sum(r["tps"] for r in okish) / len(okish), 2) if okish else 0,
            "ok_clients": len(okish),
            "garbage_clients": n_garbage,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-u", "--url", default=os.environ.get("URL", "http://localhost:8000"))
    ap.add_argument("-m", "--model", default=os.environ.get("MODEL", "qwen3-30b-a3b"))
    ap.add_argument("-n", "--n-clients", type=int, default=4)
    ap.add_argument("-t", "--tokens", type=int, default=80)
    ap.add_argument("-T", "--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--unique", action="store_true",
                    help="Each client gets a different prompt (default: all clients send PROMPTS[0])")
    ap.add_argument("-o", "--out", default=None,
                    help="Path to write JSON; default stdout")
    ap.add_argument("--config-name", default="adhoc",
                    help="Server config tag, recorded in JSON for aggregation")
    args = ap.parse_args()

    print(f"[bench] url={args.url} model={args.model} N={args.n_clients} tokens={args.tokens} temp={args.temperature} unique={args.unique}",
          file=sys.stderr)
    record = run(args.url, args.model, args.n_clients, args.tokens,
                 args.temperature, args.top_p, args.unique)
    record["config_name"] = args.config_name
    s = record["summary"]
    print(f"[bench] wall={s['wall_s']}s  agg={s['agg_tps']}t/s  per_client={s['avg_per_client_tps']}t/s  garbage={s['garbage_clients']}/{args.n_clients}",
          file=sys.stderr)

    blob = json.dumps(record, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(blob)
        print(f"[bench] wrote {args.out}", file=sys.stderr)
    else:
        print(blob)


if __name__ == "__main__":
    main()
