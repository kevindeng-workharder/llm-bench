import time, json, urllib.request, statistics

URL="http://127.0.0.1:8000/v1/completions"
MODEL="qwen3_6-27b-int8"

def stream_run(prompt, n):
    body=json.dumps({"model":MODEL,"prompt":prompt,"max_tokens":n,
                     "temperature":0,"stream":True}).encode()
    req=urllib.request.Request(URL, body, {"Content-Type":"application/json"})
    t0=time.time(); t_first=None; toks=0
    with urllib.request.urlopen(req, timeout=400) as r:
        for line in r:
            line=line.decode().strip()
            if not line.startswith("data:"): continue
            data=line[5:].strip()
            if data=="[DONE]": break
            try: obj=json.loads(data)
            except: continue
            txt=obj["choices"][0].get("text","")
            if txt:
                if t_first is None: t_first=time.time()
                toks+=1
    t_end=time.time()
    ttft=(t_first-t0) if t_first else float('nan')
    dec_s=(t_end-t_first) if (t_first and toks>1) else float('nan')
    dtps=(toks-1)/dec_s if (dec_s and dec_s>0) else float('nan')
    return toks, ttft, t_end-t0, dtps

print("warmup (JIT 预热)...", flush=True)
stream_run("Hello world", 4)
print("warmup done\n", flush=True)

prompts=["Explain how a CPU works in detail.",
         "Describe the water cycle step by step.",
         "What is machine learning and how does it work?"]
decs=[]
for i,p in enumerate(prompts):
    toks,ttft,total,dec=stream_run(p,32)
    decs.append(dec)
    print(f"run{i+1}: {toks} tok  TTFT(prefill)={ttft:.1f}s  total={total:.1f}s  "
          f"decode={dec:.3f} tok/s ({1/dec:.1f}s/tok)", flush=True)

decs=[d for d in decs if d==d]  # drop nan
if decs:
    print(f"\n>>> DECODE (eager, N=1, warm): avg={statistics.mean(decs):.3f}  "
          f"min={min(decs):.3f}  max={max(decs):.3f} tok/s", flush=True)
