#!/usr/bin/env python3
"""N=1 sequential stress test.

Hammer the local vLLM server with sequential single-stream requests,
varied prompts, varied max_tokens. Detect garbage outputs (repeated
'!', empty, NaN-shaped). Print pass/fail per request and summary.

Hypothesis: N=1 sequential should NEVER hit the bug because there's
no mixed prefill+decode batch in any kernel call. This script tests
that hypothesis empirically.
"""
import sys, time, json
import httpx

URL = "http://localhost:8000"
MODEL = "qwen3-4b"

# Variety of prompts (different content patterns) - replicate to fill 50 requests
SHORT_PROMPTS = [
    "Write a short Python function that reverses a string.",
    "Explain in 3 sentences what Rust borrow checker does.",
    "Tell me a one-paragraph story about a robot learning to paint.",
    "Give 5 surprising facts about octopuses.",
    "What's the difference between TCP and UDP? Brief.",
    "Write a haiku about morning coffee.",
    "List 3 differences between SQL and NoSQL.",
    "Explain neural networks like I'm 10.",
    "What does 'borrow checker' mean in Rust?",
    "Compose a 4-line poem about autumn.",
]

# Long prompts (~3000 tokens of repetitive content) — should trigger
# vLLM's internal chunked prefill if max_num_batched_tokens kicks in.
LONG_FILLER = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. "
    "How vexingly quick daft zebras jump! "
    "The five boxing wizards jump quickly. "
) * 35   # ~1400 tokens (fits in 4096 ctx after chat template + reply room)

LONG_PROMPTS = [
    f"Read this passage and summarize it in one sentence:\n\n{LONG_FILLER}",
    f"Count how many times 'quick' appears in this text:\n\n{LONG_FILLER}",
    f"Find any grammatical issues in this passage:\n\n{LONG_FILLER}",
]


def is_garbage(text: str) -> bool:
    """Heuristic: garbage outputs look like '!!!!!' or other repeats."""
    if not text or len(text.strip()) < 5:
        return True
    # >10 consecutive '!' or other single char
    for ch in "!@#$.,?":
        if ch * 10 in text:
            return True
    # All chars are the same one
    stripped = text.strip()
    if len(set(stripped[:50])) <= 2:
        return True
    return False


def req(prompt: str, max_tokens: int) -> dict:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": max_tokens,
    }
    t0 = time.time()
    try:
        r = httpx.post(f"{URL}/v1/chat/completions", json=body, timeout=300)
        elapsed = time.time() - t0
        if r.status_code != 200:
            return {"ok": False, "error": f"HTTP {r.status_code}",
                    "elapsed": elapsed}
        d = r.json()
        text = d["choices"][0]["message"]["content"]
        usage = d.get("usage", {})
        return {
            "ok": True,
            "text": text,
            "completion_tokens": usage.get("completion_tokens", 0),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "elapsed": elapsed,
            "garbage": is_garbage(text),
        }
    except Exception as e:
        return {"ok": False, "error": str(e), "elapsed": time.time() - t0}


def stress_pass(name: str, prompts: list[str], max_tokens: int, n_iters: int):
    print(f"\n=== Stress pass: {name} (n={n_iters}, max_tokens={max_tokens}) ===")
    print(f"{'i':>3} {'prompt_tokens':>14} {'completion':>11} {'tps':>6} {'elapsed':>9} status  text_head")
    print('-' * 100)
    results = []
    for i in range(n_iters):
        prompt = prompts[i % len(prompts)]
        r = req(prompt, max_tokens)
        if not r["ok"]:
            print(f"{i:>3} ERROR: {r.get('error')} after {r['elapsed']:.1f}s")
            results.append(r)
            continue
        head = r["text"][:60].replace("\n", " ").replace("<think>", "").strip()
        flag = "GARBAGE" if r["garbage"] else "ok"
        tps = r["completion_tokens"] / r["elapsed"] if r["elapsed"] > 0 else 0
        print(f"{i:>3} {r['prompt_tokens']:>14} {r['completion_tokens']:>11} "
              f"{tps:>6.1f} {r['elapsed']:>8.1f}s {flag:7s} \"{head}\"")
        results.append(r)
    n_garbage = sum(1 for r in results if r.get("garbage", False))
    n_ok = sum(1 for r in results if r.get("ok", False))
    n_err = len(results) - n_ok
    print(f"\n{name}: {len(results)} total, {n_garbage} garbage, {n_err} errors, "
          f"{n_ok - n_garbage} good")
    return {"name": name, "results": results, "n_garbage": n_garbage,
            "n_total": len(results), "n_err": n_err}


if __name__ == "__main__":
    # Wait for server
    print("Waiting for /v1/models to respond...")
    for i in range(120):
        try:
            r = httpx.get(f"{URL}/v1/models", timeout=3)
            if r.status_code == 200:
                print("Server ready.")
                break
        except Exception:
            pass
        time.sleep(2)
    else:
        print("Server never came up", file=sys.stderr)
        sys.exit(1)

    summary = []
    summary.append(stress_pass("short_50",  SHORT_PROMPTS,  max_tokens=200, n_iters=50))
    summary.append(stress_pass("long_5",    LONG_PROMPTS,   max_tokens=200, n_iters=5))

    total_garbage = sum(s["n_garbage"] for s in summary)
    print(f"\n========= OVERALL =========")
    for s in summary:
        print(f"  {s['name']}: {s['n_garbage']} garbage / {s['n_total']} total")
    print(f"  TOTAL garbage: {total_garbage}")

    # Save raw results
    out = {"ts": time.strftime("%Y%m%d-%H%M%S"), "passes": [
        {"name": s["name"], "n_total": s["n_total"], "n_garbage": s["n_garbage"],
         "n_err": s["n_err"],
         "results": [{"completion_tokens": r.get("completion_tokens"),
                      "prompt_tokens": r.get("prompt_tokens"),
                      "elapsed": r.get("elapsed"),
                      "garbage": r.get("garbage"),
                      "text_head": (r.get("text") or "")[:120],
                      "ok": r.get("ok"),
                      "error": r.get("error")}
                     for r in s["results"]]}
        for s in summary]}
    with open("/tmp/stress-n1-result.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nRaw results: /tmp/stress-n1-result.json")
    sys.exit(1 if total_garbage > 0 else 0)
