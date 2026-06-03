#!/usr/bin/env python3
# Single-stream decode throughput for the p2p-ib leader (VM1:8000).
# Mirrors the N=1, max_tokens=80, temperature=0 methodology used to land the
# 11.76 tok/s reference figure.
#
# RUN ON VM1 (the leader; the server listens on 0.0.0.0:8000):
#   ssh -p 2224 ubuntu@127.0.0.1 'python3 - < bench.py'
# or copy it over and: python3 bench.py [--host 127.0.0.1] [--port 8000]
#                                       [--tokens 80] [--runs 4] [--warmup 1]
#
# VERIFIED 2026-06-03: warm runs 8.5-12.5 tok/s (avg ~10, peak 12.52);
# 11.76 reference is within range. The spread is inherent to the
# GDR-disabled host-bounce IB path (every GPU<->NIC DMA stages through host
# memory, so per-step collective latency jitters). See RESULTS.md.

import argparse, json, time, urllib.request

PROMPTS = [
    "Explain how a CPU works.",
    "Describe the water cycle.",
    "What is machine learning?",
    "Tell me about the ocean.",
    "Write a short paragraph about the history of computers.",
]


def run_once(host, port, model, prompt, tokens):
    body = json.dumps({
        "model": model, "prompt": prompt,
        "max_tokens": tokens, "temperature": 0,
    }).encode()
    req = urllib.request.Request(
        f"http://{host}:{port}/v1/completions",
        body, {"Content-Type": "application/json"})
    t = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=120))
    dt = time.time() - t
    return r["usage"]["completion_tokens"], dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default="qwen3-4b")
    ap.add_argument("--tokens", type=int, default=80)
    ap.add_argument("--runs", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=1)
    a = ap.parse_args()

    for i in range(a.warmup):
        run_once(a.host, a.port, a.model, PROMPTS[i % len(PROMPTS)], a.tokens)
        print(f"warmup {i+1}/{a.warmup} done")

    rates = []
    for i in range(a.runs):
        n, dt = run_once(a.host, a.port, a.model,
                         PROMPTS[i % len(PROMPTS)], a.tokens)
        rate = n / dt
        rates.append(rate)
        print(f"run {i+1}: {n} tok in {dt:.2f}s = {rate:.2f} tok/s")

    if rates:
        print(f"\nsummary: avg={sum(rates)/len(rates):.2f}  "
              f"min={min(rates):.2f}  max={max(rates):.2f} tok/s  "
              f"(N=1, max_tokens={a.tokens})")


if __name__ == "__main__":
    main()
