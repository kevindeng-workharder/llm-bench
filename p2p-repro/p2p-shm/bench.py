import time, json, urllib.request, sys
from concurrent.futures import ThreadPoolExecutor, as_completed

URL = "http://127.0.0.1:8000/v1/completions"
MODEL = "qwen3_6-27b-int8"
PROMPTS = [
    "Explain how a CPU pipeline works, stage by stage.",
    "Describe the water cycle in detail, step by step.",
    "What is machine learning and how does gradient descent train a model?",
    "Write a short story about a lighthouse keeper and a storm.",
    "Explain the causes and consequences of the French Revolution.",
    "How does the TCP three-way handshake establish a connection?",
    "Summarize the theory of plate tectonics and what it explains.",
    "Describe how photosynthesis converts sunlight into chemical energy.",
]
REQ_TIMEOUT = 240   # per-request; a 96-tok stream over >1 tok/s finishes well under this
SWEEP_TIMEOUT = 600 # overall per-N; exceeding => a request never completed => HANG

def stream_run(prompt, n):
    body = json.dumps({"model": MODEL, "prompt": prompt, "max_tokens": n,
                       "temperature": 0, "stream": True}).encode()
    req = urllib.request.Request(URL, body, {"Content-Type": "application/json"})
    t0 = time.time(); t_first = None; toks = 0
    with urllib.request.urlopen(req, timeout=REQ_TIMEOUT) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data:"): continue
            d = line[5:].strip()
            if d == "[DONE]": break
            try: obj = json.loads(d)
            except: continue
            if obj["choices"][0].get("text", ""):
                if t_first is None: t_first = time.time()
                toks += 1
    te = time.time()
    dec = (toks - 1) / (te - t_first) if (t_first and toks > 1) else None
    return {"toks": toks, "ttft": (t_first - t0) if t_first else None, "wall": te - t0, "dec": dec}

def run_N(N, max_tokens=96):
    t0 = time.time(); res = [None] * N; hang = False
    with ThreadPoolExecutor(max_workers=N) as ex:
        futs = {ex.submit(stream_run, PROMPTS[i % len(PROMPTS)], max_tokens): i for i in range(N)}
        try:
            for f in as_completed(futs, timeout=SWEEP_TIMEOUT):
                i = futs[f]
                try: res[i] = f.result()
                except Exception as e: res[i] = {"err": repr(e)}
        except TimeoutError:
            hang = True
    wall = time.time() - t0
    ok = [r for r in res if r and "toks" in r]
    err = [r for r in res if r and "err" in r]
    pending = sum(1 for r in res if r is None)
    tot = sum(r["toks"] for r in ok)
    agg = tot / wall if wall > 0 else 0
    per = [r["dec"] for r in ok if r["dec"]]
    ttfts = [r["ttft"] for r in ok if r["ttft"]]
    tag = "  *** HANG/DEADLOCK ***" if (hang or pending) else ""
    print(f"N={N:>2}: ok={len(ok)}/{N} err={len(err)} pending={pending} | "
          f"AGG={agg:5.2f} tok/s | per-req decode avg={ (sum(per)/len(per) if per else 0):4.2f} | "
          f"TTFT avg={ (sum(ttfts)/len(ttfts) if ttfts else 0):5.1f}s | wall={wall:5.1f}s | toks={tot}{tag}", flush=True)
    if err: print(f"      first error: {err[0]['err'][:160]}", flush=True)
    return not (hang or pending or err)

print("warmup...", flush=True); stream_run("Hello world", 4); print("warmup done\n", flush=True)
levels = [int(x) for x in sys.argv[1:]] or [1, 2, 4, 8]
allok = True
for N in levels:
    allok &= run_N(N)
print("\n>>> SWEEP " + ("CLEAN — no hangs/deadlocks" if allok else "had HANG/ERROR — see above"), flush=True)
