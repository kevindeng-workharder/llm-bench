#!/usr/bin/env python3
"""Concurrent stress test for llama.cpp /v1/chat/completions endpoint.

Fires N parallel requests against the server, measures:
  - Per-request TTFT and tokens/s
  - Aggregate tokens/s across all clients
  - Total wall time

Usage:
    python3 concurrent-llamacpp.py [-n CLIENTS] [-t MAX_TOKENS] [URL]

Defaults: 4 clients, 100 tokens each, http://localhost:8000.
"""
import argparse
import concurrent.futures
import json
import os
import sys
import time

import httpx

PROMPTS = [
    "Write a short Python function that reverses a string.",
    "Explain in 3 sentences what Rust borrow checker does.",
    "Tell me a one-paragraph story about a robot learning to paint.",
    "Give 5 surprising facts about octopuses.",
    "What's the difference between TCP and UDP? Brief.",
    "Compose a 4-line poem about morning coffee.",
    "List 3 differences between SQL and NoSQL databases.",
    "How does a neural network learn? Explain like I'm 12.",
]


def one_request(args, client_id: int):
    """Send one streaming chat completion request, time it."""
    url = args.url
    prompt = PROMPTS[client_id % len(PROMPTS)]
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.tokens,
        "temperature": 0.7,
        "top_p": 0.95,
        "stream": True,
    }
    t0 = time.time()
    t_first = None
    n = 0
    full = []
    try:
        with httpx.stream("POST", f"{url}/v1/chat/completions",
                          json=body, timeout=600) as r:
            if r.status_code != 200:
                return {
                    "id": client_id, "ok": False,
                    "err": f"HTTP {r.status_code}: {r.read().decode()[:200]}",
                    "elapsed": time.time() - t0,
                }
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
                        n += 1
                except Exception:
                    pass
    except Exception as e:
        return {"id": client_id, "ok": False, "err": str(e), "elapsed": time.time() - t0}

    elapsed = time.time() - t0
    return {
        "id": client_id, "ok": True,
        "prompt_short": prompt[:40],
        "n_chunks": n,
        "ttft": t_first,
        "elapsed": elapsed,
        "tps": (n / elapsed) if elapsed > 0 else 0,
        "output_preview": "".join(full)[:80].replace("\n", " "),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--clients", type=int, default=4)
    ap.add_argument("-t", "--tokens",  type=int, default=100,
                    help="max tokens per request")
    ap.add_argument("-m", "--model", default=os.environ.get("MODEL", "qwen36-a3b"))
    ap.add_argument("url", nargs="?",
                    default=os.environ.get("URL", "http://localhost:8000"))
    args = ap.parse_args()

    print(f"Server: {args.url}")
    print(f"Clients: {args.clients}, max_tokens/req: {args.tokens}")
    print("=" * 70)

    t_start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.clients) as ex:
        futures = [ex.submit(one_request, args, i) for i in range(args.clients)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
    t_total = time.time() - t_start

    results.sort(key=lambda r: r["id"])
    total_chunks = sum(r.get("n_chunks", 0) for r in results)
    okish = [r for r in results if r["ok"]]
    print(f"\nResults ({len(okish)}/{len(results)} ok):\n")
    for r in results:
        if not r["ok"]:
            print(f"  [{r['id']}] FAIL: {r['err']}")
            continue
        ttft = f"{r['ttft']:.2f}s" if r['ttft'] else "-"
        print(f"  [{r['id']}] {r['n_chunks']:4d} chunks  {r['tps']:5.2f} tok/s  "
              f"ttft={ttft}  elapsed={r['elapsed']:5.1f}s  '{r['output_preview']}'")

    print()
    print("=" * 70)
    print(f"Total wall:        {t_total:.1f}s")
    print(f"Total chunks:      {total_chunks}")
    print(f"Aggregate tok/s:   {total_chunks/t_total:.2f}")
    if okish:
        avg_per = sum(r['tps'] for r in okish) / len(okish)
        print(f"Avg per-client tok/s: {avg_per:.2f}")
    print(f"Speedup vs serial: {(total_chunks/t_total) / (sum(r['tps'] for r in okish)/len(okish) if okish else 1):.2f}x")


if __name__ == "__main__":
    main()
