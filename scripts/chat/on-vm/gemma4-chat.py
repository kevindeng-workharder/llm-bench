#!/usr/bin/env python3
"""Interactive chat with Gemma-4-E4B-it on riscv64 + gfx1100.

Pipeline: real triton 3.4 + TRITON_ATTN backend + FULL_DECODE_ONLY cudagraph
(see docs/triton-riscv-gfx1100.md in the rocm-riscv-build repo).

Commands (type at the `>` prompt):
    /quit | /exit | /q        exit
    /clear | /reset           clear conversation history
    /system <text>            set system prompt (also clears history)
    /temp <0..2>              set sampling temperature (default 0.7)
    /max <N>                  set max_tokens per reply (default 512)
    /show                     dump current history
    /help                     this help

The reply is printed after generation completes (no streaming yet). Each
reply shows timing + cumulative context length so you can see when to /clear.
"""
import os, sys, time

MODEL = "/data/gemma-4-E4B-it"
# Gemma4 natively supports 131072 token context. We cap at 8192 here so the
# KV cache (which grows with max_model_len because vLLM pre-allocates budget
# proportionally) fits comfortably on gfx1100 under gpu_memory_utilization=0.75.
# If you want more, raise MAX_MODEL_LEN (e.g. 16384) and accept that:
#   - first-request latency grows (cudagraph capture + KV-init)
#   - KV cache budget drops (at 16K, ~6K concurrent tokens across the single chat
#     vs ~13K at 8K). Fine for solo interactive use.
# Override at runtime: MAX_MODEL_LEN=16384 python3 gemma4-chat.py
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "8192"))
DEFAULT_TEMP = 0.7
DEFAULT_MAX_TOK = 2048   # per-reply cap; user can raise with /max (up to MAX_MODEL_LEN - ctx - 8)
DEFAULT_TOP_P = 0.9

print(f"loading {MODEL}", flush=True)
print(f"  (first run takes ~4 min: weights → triton JIT → cudagraph capture)", flush=True)
print(f"  (subsequent requests replay the captured graph → ~20 tok/s)", flush=True)
t0 = time.time()
from vllm import LLMEngine, EngineArgs, SamplingParams
from vllm.config.compilation import CompilationMode, CUDAGraphMode
from vllm.usage.usage_lib import UsageContext
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(MODEL)

# We drive the engine directly (step-loop) so we can stream tokens as they
# decode, rather than blocking on llm.generate() until the whole reply is
# finished. LLM.generate() in vLLM 0.19 has no streaming mode; LLMEngine does.
engine_args = EngineArgs(
    model=MODEL, dtype="bfloat16",
    compilation_config={
        "mode": CompilationMode.NONE,
        "cudagraph_mode": CUDAGraphMode.FULL_DECODE_ONLY,
        "cudagraph_capture_sizes": [1, 2, 4, 8],
        "max_cudagraph_capture_size": 8,
    },
    gpu_memory_utilization=0.75,
    max_model_len=MAX_MODEL_LEN,
    hf_overrides={"architectures": ["Gemma4ForCausalLM"]},
    disable_log_stats=True,
)
engine = LLMEngine.from_engine_args(engine_args, usage_context=UsageContext.LLM_CLASS)
print(f"loaded in {time.time()-t0:.1f}s", flush=True)
print("")
print("=" * 60)
print("  Gemma-4-E4B-it chat")
print("  /quit /clear /system /temp /max /show /help")
print("=" * 60)

history = []       # list of {"role": "user"|"assistant", "content": "..."}
system_prompt = None
cur_temp = DEFAULT_TEMP
cur_max_tok = DEFAULT_MAX_TOK

def render_prompt() -> str:
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.extend(history)
    return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)

def show_history():
    if system_prompt:
        print(f"  [system] {system_prompt[:100]}{'...' if len(system_prompt)>100 else ''}")
    if not history:
        print("  (no turns)")
    for i, m in enumerate(history):
        prefix = f"  [{i+1} {m['role'][0]}]"
        text = m['content'].replace("\n", " ")
        print(f"{prefix} {text[:120]}{'...' if len(text)>120 else ''}")

def show_help():
    print("  /quit /exit /q            exit")
    print("  /clear /reset             clear conversation")
    print(f"  /system <text>            set system prompt (also clears)")
    print(f"  /temp <0..2>              set temperature (current {cur_temp})")
    print(f"  /max <N>                  set max_tokens per reply (current {cur_max_tok})")
    print("  /show                     dump history")

# Default Ctrl-C handling: at the input prompt, clears the line (we catch
# KeyboardInterrupt and loop); during generation, aborts the request and
# keeps the REPL alive. Ctrl-D (EOF) exits cleanly.

while True:
    try:
        line = input("\n> ").strip()
    except EOFError:
        print()
        break
    except KeyboardInterrupt:
        print("  (Ctrl-C — type /quit or Ctrl-D to exit)")
        continue

    if not line:
        continue

    if line.lower() in ("/quit", "/exit", "/q"):
        break
    if line.lower() in ("/clear", "/reset"):
        history = []
        print("  (history cleared)")
        continue
    if line.lower() == "/help":
        show_help(); continue
    if line.lower() == "/show":
        show_history(); continue
    if line.lower().startswith("/system "):
        system_prompt = line[len("/system "):].strip() or None
        history = []
        print(f"  (system prompt {'set' if system_prompt else 'cleared'}; history cleared)")
        continue
    if line.lower().startswith("/temp "):
        try:
            cur_temp = float(line.split()[1]); print(f"  (temperature = {cur_temp})")
        except Exception:
            print("  usage: /temp <0..2>")
        continue
    if line.lower().startswith("/max "):
        try:
            cur_max_tok = int(line.split()[1]); print(f"  (max_tokens = {cur_max_tok})")
        except Exception:
            print("  usage: /max <N>")
        continue
    if line.startswith("/"):
        print(f"  unknown command: {line.split()[0]}")
        continue

    history.append({"role": "user", "content": line})
    prompt = render_prompt()
    prompt_tokens = len(tok(prompt).input_ids)

    # Protect against overflow — if ctx + max_tok would exceed window,
    # auto-shrink max_tok for this turn instead of rejecting the request.
    available = MAX_MODEL_LEN - prompt_tokens - 8
    if available <= 32:
        print(f"  (context {prompt_tokens} tok leaves only {available} for reply in {MAX_MODEL_LEN} window; /clear first)")
        history.pop()
        continue
    this_turn_max_tok = min(cur_max_tok, available)
    if this_turn_max_tok < cur_max_tok:
        print(f"  (auto-shrinking this reply to {this_turn_max_tok} tok — ctx near window)")

    params = SamplingParams(
        max_tokens=this_turn_max_tok,
        temperature=cur_temp,
        top_p=DEFAULT_TOP_P,
    )

    # --- streaming via LLMEngine.step() loop ---
    # step() returns all active RequestOutputs; RequestOutput.outputs[0].text
    # is the cumulative decoded text (not a delta). We track prev_text length
    # and print only the newly-added suffix each tick → token-by-token print.
    print()
    req_id = f"turn-{int(time.time() * 1000)}"
    engine.add_request(req_id, prompt, params)

    prev_text = ""
    n_gen = 0
    t0 = time.time()
    t_first = None
    try:
        while engine.has_unfinished_requests():
            step_outputs = engine.step()
            for ro in step_outputs:
                if ro.request_id != req_id:
                    continue
                new_text = ro.outputs[0].text
                n_gen = len(ro.outputs[0].token_ids)
                if len(new_text) > len(prev_text):
                    if t_first is None:
                        t_first = time.time()
                    delta = new_text[len(prev_text):]
                    print(delta, end="", flush=True)
                    prev_text = new_text
                if ro.finished:
                    break
    except KeyboardInterrupt:
        # Abort this request, keep the engine alive for the next turn
        print("\n  (interrupted)", flush=True)
        try:
            engine.abort_request([req_id])
        except Exception:
            pass
        # Still append partial reply to history so the model sees it next turn
        history.append({"role": "assistant", "content": prev_text})
        continue

    elapsed = time.time() - t0
    ttft = (t_first - t0) if t_first else elapsed
    history.append({"role": "assistant", "content": prev_text})

    tps = (n_gen / elapsed) if elapsed > 0 else 0.0
    print(f"\n\n  [{n_gen} tok in {elapsed:.1f}s = {tps:.2f} tok/s | "
          f"TTFT {ttft:.2f}s | ctx {prompt_tokens}→{prompt_tokens + n_gen}/{MAX_MODEL_LEN}]",
          flush=True)

print("bye.")
# Avoid torch interpreter-shutdown segfault on riscv64
sys.stdout.flush()
os._exit(0)
