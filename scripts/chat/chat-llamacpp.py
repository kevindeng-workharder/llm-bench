#!/usr/bin/env python3
"""Interactive chat client for llama-server (Qwen3.6-35B-A3B MXFP4 on riscv64 + ROCm).

Two modes — toggle with /chat or /raw:
  - chat:  uses /v1/chat/completions (OpenAI-compat). Streams via SSE.
           Subject to the GGUF's embedded chat template (may have quirks
           with thinking models — see /raw if outputs go weird).
  - raw:   uses /completion. Builds the Qwen chatml prompt manually, streams.
           Bypasses server-side template parsing but still wraps in chatml.
  - comp:  uses /completion with NO wrapping. Your input is sent verbatim
           and the model continues from there. Use for non-chat completion
           tests (e.g. "The capital of France is" -> " Paris.").

Commands at the `>` prompt:
    /quit /exit /q          exit
    /clear /reset           clear conversation
    /system <text>          set system prompt (clears history)
    /chat                   switch to /v1/chat/completions endpoint
    /raw                    switch to /completion endpoint (manual chatml)
    /comp                   switch to /completion with NO wrapping (pure)
    /think on|off           include `<think>\\n` priming (raw mode only)
    /temp <0..2>            sampling temperature (default 0.7)
    /top-p <0..1>           top-p (default 0.95)
    /top-k <N>              top-k (default 40)
    /max <N>                per-reply max tokens (default 1024)
    /show                   dump conversation
    /info                   server config
    /help                   this help

Defaults to RAW mode since the chat-template path is unreliable on this build.
Tokens stream live; each reply prints TTFT + tok/s.
"""
import json
import os
import sys
import time

import httpx

URL = os.environ.get("URL", "http://localhost:8000")
MODEL = os.environ.get("MODEL", "qwen36-a3b")
DEFAULT_TEMP = 0.7
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 40
DEFAULT_MAX = 1024


def get_server_info():
    try:
        r = httpx.get(f"{URL}/v1/models", timeout=10)
        m = r.json()["data"][0]
        return m.get("id", MODEL), int(m.get("meta", {}).get("n_ctx_train", 4096))
    except Exception as e:
        print(f"  could not reach {URL}: {e}")
        sys.exit(1)


def show_help():
    print(__doc__)


def show_history(history, system_prompt):
    if system_prompt:
        sp = system_prompt.replace("\n", " ")
        print(f"  [system] {sp[:140]}{'...' if len(sp) > 140 else ''}")
    if not history:
        print("  (no turns)")
        return
    for i, m in enumerate(history):
        text = str(m["content"]).replace("\n", " ")
        print(f"  [{i+1} {m['role'][0]}] {text[:140]}{'...' if len(text) > 140 else ''}")


def show_info(model, n_ctx, mode, params, think):
    print(f"  url:       {URL}")
    print(f"  model:     {model}")
    print(f"  n_ctx:     {n_ctx}")
    print(f"  mode:      {mode}")
    print(f"  thinking:  {think} (raw mode only)")
    for k, v in params.items():
        print(f"  {k:9s}: {v}")


def build_chatml_prompt(history, system_prompt, think):
    """Build the literal Qwen chatml prompt for /completion endpoint."""
    parts = []
    if system_prompt:
        parts.append(f"<|im_start|>system\n{system_prompt}<|im_end|>\n")
    for m in history:
        role = m["role"]
        content = m["content"]
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    if think:
        parts.append("<think>\n")
    return "".join(parts)


def stream_chat(history, system_prompt, params):
    """POST to /v1/chat/completions, stream SSE deltas."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history)
    body = {
        "model": MODEL,
        "messages": messages,
        "stream": True,
        **params,
    }
    with httpx.stream("POST", f"{URL}/v1/chat/completions",
                      json=body, timeout=600) as resp:
        if resp.status_code != 200:
            print(f"\n  HTTP {resp.status_code}: {resp.read().decode()[:300]}")
            return
        for line in resp.iter_lines():
            if not line.startswith("data: "):
                continue
            chunk = line[len("data: "):]
            if chunk.strip() == "[DONE]":
                return
            try:
                obj = json.loads(chunk)
                delta_obj = obj["choices"][0]["delta"]
                # llama-server in chat mode may stream both
                # `reasoning_content` (the <think> block) and `content`.
                rc = delta_obj.get("reasoning_content")
                if rc:
                    yield rc
                ct = delta_obj.get("content")
                if ct:
                    yield ct
            except Exception:
                pass


def stream_raw(history, system_prompt, params, think):
    """POST to /completion, stream tokens (llama.cpp native format)."""
    prompt = build_chatml_prompt(history, system_prompt, think)
    body = {
        "prompt": prompt,
        "stream": True,
        "stop": ["<|im_end|>", "<|endoftext|>"],
        "n_predict": params.get("max_tokens", DEFAULT_MAX),
        "temperature": params.get("temperature", DEFAULT_TEMP),
        "top_p": params.get("top_p", DEFAULT_TOP_P),
        "top_k": params.get("top_k", DEFAULT_TOP_K),
        "repeat_penalty": params.get("repeat_penalty", 1.15),
        "repeat_last_n": 256,
        "frequency_penalty": 0.5,
        "presence_penalty": 0.2,
        "dry_multiplier": 0.8,
        "dry_base": 1.75,
        "dry_allowed_length": 2,
    }
    yield from _stream_completion(body)


def stream_comp(text, params):
    """POST to /completion with verbatim text — no chatml wrapping."""
    body = {
        "prompt": text,
        "stream": True,
        "n_predict": params.get("max_tokens", DEFAULT_MAX),
        "temperature": params.get("temperature", DEFAULT_TEMP),
        "top_p": params.get("top_p", DEFAULT_TOP_P),
        "top_k": params.get("top_k", DEFAULT_TOP_K),
        "repeat_penalty": params.get("repeat_penalty", 1.05),
    }
    yield from _stream_completion(body)


def _stream_completion(body):
    with httpx.stream("POST", f"{URL}/completion",
                      json=body, timeout=600) as resp:
        if resp.status_code != 200:
            print(f"\n  HTTP {resp.status_code}: {resp.read().decode()[:300]}")
            return
        for line in resp.iter_lines():
            if not line.startswith("data: "):
                continue
            chunk = line[len("data: "):]
            try:
                obj = json.loads(chunk)
                content = obj.get("content", "")
                if content:
                    yield content
                if obj.get("stop"):
                    return
            except Exception:
                pass


def main():
    model, n_ctx = get_server_info()
    print(f"connected: {URL}  model={model}  n_ctx={n_ctx}")
    print("=" * 60)
    print("  llama.cpp chat — raw=manual chatml, chat=/v1/chat/completions")
    print("  /chat /raw /think on|off /system /temp /top-p /top-k /max /help")
    print("=" * 60)

    history: list = []
    system_prompt: str | None = None
    mode = "raw"
    think = False
    params = {
        "max_tokens": DEFAULT_MAX,
        "temperature": DEFAULT_TEMP,
        "top_p": DEFAULT_TOP_P,
        "top_k": DEFAULT_TOP_K,
        "repeat_penalty": 1.05,
    }

    while True:
        try:
            label = f"[{mode}{'+think' if think else ''}]> "
            line = input(f"\n{label}").strip()
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print("  (Ctrl-C — type /quit to exit)")
            continue

        if not line:
            continue

        # commands
        if line.lower() in ("/quit", "/exit", "/q"):
            break
        if line.lower() in ("/clear", "/reset"):
            history = []
            print("  (history cleared)")
            continue
        if line.lower() == "/help":
            show_help()
            continue
        if line.lower() == "/show":
            show_history(history, system_prompt)
            continue
        if line.lower() == "/info":
            show_info(model, n_ctx, mode, params, think)
            continue
        if line.lower() == "/chat":
            mode = "chat"
            print("  (mode: /v1/chat/completions)")
            continue
        if line.lower() == "/raw":
            mode = "raw"
            print("  (mode: /completion with manual chatml)")
            continue
        if line.lower() == "/comp":
            mode = "comp"
            print("  (mode: /completion verbatim, no wrapping, no history)")
            continue
        if line.lower().startswith("/think "):
            arg = line.split()[1].lower()
            think = arg in ("on", "true", "1", "yes")
            print(f"  (think prefix = {think})")
            continue
        if line.lower().startswith("/system "):
            system_prompt = line[len("/system "):].strip() or None
            history = []
            print(f"  (system prompt {'set' if system_prompt else 'cleared'}; history cleared)")
            continue
        if line.lower().startswith("/temp "):
            try:
                params["temperature"] = float(line.split()[1])
                print(f"  (temperature = {params['temperature']})")
            except Exception:
                print("  usage: /temp <0..2>")
            continue
        if line.lower().startswith("/top-p "):
            try:
                params["top_p"] = float(line.split()[1])
                print(f"  (top_p = {params['top_p']})")
            except Exception:
                print("  usage: /top-p <0..1>")
            continue
        if line.lower().startswith("/top-k "):
            try:
                params["top_k"] = int(line.split()[1])
                print(f"  (top_k = {params['top_k']})")
            except Exception:
                print("  usage: /top-k <N>")
            continue
        if line.lower().startswith("/max "):
            try:
                params["max_tokens"] = int(line.split()[1])
                print(f"  (max_tokens = {params['max_tokens']})")
            except Exception:
                print("  usage: /max <N>")
            continue
        if line.startswith("/"):
            print(f"  unknown command: {line.split()[0]}; /help")
            continue

        # comp mode: verbatim, no history tracking
        if mode == "comp":
            print()
            t_start = time.time()
            t_first = None
            n_chunks = 0
            try:
                for delta in stream_comp(line, params):
                    if t_first is None:
                        t_first = time.time() - t_start
                    print(delta, end="", flush=True)
                    n_chunks += 1
            except KeyboardInterrupt:
                print("\n  (interrupted)")
            elapsed = time.time() - t_start
            ttft_str = f"TTFT {t_first:.2f}s" if t_first else "TTFT -"
            rate = (n_chunks / elapsed) if elapsed > 0 else 0
            print(f"\n\n  [{n_chunks} chunks in {elapsed:.1f}s = {rate:.2f} chunk/s | {ttft_str}]")
            continue

        # chat / raw modes: track history
        history.append({"role": "user", "content": line})

        print()
        t_start = time.time()
        t_first = None
        n_chunks = 0
        full = []
        try:
            stream = stream_chat(history, system_prompt, params) if mode == "chat" \
                else stream_raw(history, system_prompt, params, think)
            for delta in stream:
                if t_first is None:
                    t_first = time.time() - t_start
                print(delta, end="", flush=True)
                full.append(delta)
                n_chunks += 1
        except KeyboardInterrupt:
            print("\n  (interrupted, partial reply kept in history)")

        elapsed = time.time() - t_start
        text = "".join(full)
        # Strip <think>...</think> blocks before saving to history.
        # Qwen thinking models will get confused by their own previous
        # reasoning showing up in subsequent turns.
        clean_text = text
        while "<think>" in clean_text and "</think>" in clean_text:
            start = clean_text.find("<think>")
            end = clean_text.find("</think>", start) + len("</think>")
            clean_text = (clean_text[:start] + clean_text[end:]).lstrip("\n")
        history.append({"role": "assistant", "content": clean_text or text})

        ttft_str = f"TTFT {t_first:.2f}s" if t_first else "TTFT -"
        rate = (n_chunks / elapsed) if elapsed > 0 else 0
        print(f"\n\n  [{n_chunks} chunks in {elapsed:.1f}s = {rate:.2f} chunk/s | {ttft_str}]")

    print("bye.")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
