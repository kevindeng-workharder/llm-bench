#!/usr/bin/env python3
"""Interactive chat client for the Gemma4 mm + TP=2 + graph server running on VM.

Server side: localhost:8000 (host SSH-forwarded → VM:8000) with the
launch-gemma4-tp2-graph.sh server. Model serves both text and images.

Commands at the `>` prompt:
    /quit | /exit | /q       exit
    /clear | /reset          clear conversation history
    /system <text>           set system prompt (also clears history)
    /img <path>              attach a local image to the NEXT user message
    /url <https-url>         attach an image URL (or already-base64 data URI)
    /temp <0..2>             sampling temperature (default 0.7)
    /max <N>                 per-reply max_tokens (default 4096)
    /show                    dump current history
    /info                    server max_model_len + KV cache info
    /help                    this help

Tokens are streamed as they arrive (no waiting for full reply). Each
reply prints TTFT + tok/s + cumulative context consumption.
"""
import base64
import json
import mimetypes
import os
import sys
import time
from pathlib import Path

import httpx

URL = os.environ.get("URL", "http://localhost:8000")
MODEL = os.environ.get("MODEL", "gemma4")
DEFAULT_TEMP = 0.7
DEFAULT_TOP_P = 0.9
DEFAULT_MAX_TOK = 4096   # large by default; auto-shrinks if context near limit


def get_server_info():
    """Fetch model + max_model_len from /v1/models."""
    try:
        r = httpx.get(f"{URL}/v1/models", timeout=10)
        m = r.json()["data"][0]
        return m.get("id", MODEL), int(m.get("max_model_len", 4096))
    except Exception as e:
        print(f"  (could not reach {URL}: {e})")
        sys.exit(1)


def load_image(path_or_url: str) -> str:
    """Return a `data:image/...;base64,...` URI."""
    if path_or_url.startswith(("http://", "https://", "data:")):
        return path_or_url
    p = Path(path_or_url).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(p)
    mime = mimetypes.guess_type(p.name)[0] or "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(p.read_bytes()).decode()}"


def show_help(cur_temp, cur_max):
    print("  /quit /exit /q              exit")
    print("  /clear /reset               clear conversation")
    print("  /system <text>              set system prompt (clears history)")
    print("  /img <path>                 attach local image to NEXT user message")
    print("  /url <url>                  attach image by URL or data URI")
    print(f"  /temp <0..2>                sampling temperature (current {cur_temp})")
    print(f"  /max <N>                    per-reply max_tokens (current {cur_max})")
    print("  /show                       dump history (truncated)")
    print("  /info                       server config (max_model_len, etc.)")


def show_history(history, system_prompt):
    if system_prompt:
        sp = system_prompt.replace("\n", " ")
        print(f"  [system] {sp[:140]}{'...' if len(sp) > 140 else ''}")
    if not history:
        print("  (no turns)")
        return
    for i, m in enumerate(history):
        if isinstance(m["content"], list):
            text = " ".join(
                p.get("text", f"[{p['type']}]") if isinstance(p, dict) else str(p)
                for p in m["content"]
            )
        else:
            text = str(m["content"])
        text = text.replace("\n", " ")
        prefix = f"  [{i+1} {m['role'][0]}]"
        print(f"{prefix} {text[:140]}{'...' if len(text) > 140 else ''}")


def show_info(model, max_model_len):
    print(f"  url:           {URL}")
    print(f"  model:         {model}")
    print(f"  max_model_len: {max_model_len}")


def stream_chat(messages, params):
    """POST + stream SSE; yield delta tokens; return (n_tokens, ttft, total_text)."""
    body = {**params, "model": MODEL, "messages": messages, "stream": True}
    n = 0
    t0 = time.time()
    t_first = None
    chunks = []
    with httpx.stream("POST", f"{URL}/v1/chat/completions",
                      json=body, timeout=600) as resp:
        if resp.status_code != 200:
            print(f"\n  HTTP {resp.status_code}: {resp.read().decode()[:300]}")
            return 0, None, ""
        for line in resp.iter_lines():
            if not line.startswith("data: "):
                continue
            chunk = line[len("data: "):]
            if chunk.strip() == "[DONE]":
                break
            try:
                obj = json.loads(chunk)
                delta = obj["choices"][0]["delta"].get("content", "") or ""
                if delta:
                    if t_first is None:
                        t_first = time.time() - t0
                    yield delta
                    chunks.append(delta)
                    n += 1
            except Exception:
                pass
    return n, t_first, "".join(chunks)


def main():
    model, max_model_len = get_server_info()
    print(f"connected: {URL} model={model} max_model_len={max_model_len}")
    print("=" * 60)
    print(f"  Gemma4 mm + TP=2 chat (HTTP streaming)")
    print(f"  /img <path>  /url <url>  /system  /temp  /max  /show  /info  /help")
    print("=" * 60)

    history: list = []
    system_prompt: str | None = None
    cur_temp = DEFAULT_TEMP
    cur_max = DEFAULT_MAX_TOK
    pending_image: str | None = None

    while True:
        try:
            prompt_label = "(+img)> " if pending_image else "> "
            line = input(f"\n{prompt_label}").strip()
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print("  (Ctrl-C — type /quit or Ctrl-D to exit)")
            continue

        if not line:
            continue

        # commands
        if line.lower() in ("/quit", "/exit", "/q"):
            break
        if line.lower() in ("/clear", "/reset"):
            history = []
            pending_image = None
            print("  (history cleared)")
            continue
        if line.lower() == "/help":
            show_help(cur_temp, cur_max)
            continue
        if line.lower() == "/show":
            show_history(history, system_prompt)
            continue
        if line.lower() == "/info":
            show_info(model, max_model_len)
            continue
        if line.lower().startswith("/system "):
            system_prompt = line[len("/system "):].strip() or None
            history = []
            print(f"  (system prompt {'set' if system_prompt else 'cleared'}; history cleared)")
            continue
        if line.lower().startswith("/temp "):
            try:
                cur_temp = float(line.split()[1])
                print(f"  (temperature = {cur_temp})")
            except Exception:
                print("  usage: /temp <0..2>")
            continue
        if line.lower().startswith("/max "):
            try:
                cur_max = int(line.split()[1])
                print(f"  (max_tokens = {cur_max})")
            except Exception:
                print("  usage: /max <N>")
            continue
        if line.lower().startswith("/img "):
            path = line[len("/img "):].strip()
            try:
                pending_image = load_image(path)
                print(f"  (attached: {path}, will go with next message)")
            except Exception as e:
                print(f"  ERROR loading {path}: {e}")
            continue
        if line.lower().startswith("/url "):
            url = line[len("/url "):].strip()
            try:
                pending_image = load_image(url)
                print(f"  (attached URL, will go with next message)")
            except Exception as e:
                print(f"  ERROR: {e}")
            continue
        if line.startswith("/"):
            print(f"  unknown command: {line.split()[0]}; /help")
            continue

        # build user content (text + optional image)
        if pending_image:
            user_content = [
                {"type": "image_url", "image_url": {"url": pending_image}},
                {"type": "text", "text": line},
            ]
            pending_image = None
        else:
            user_content = line

        history.append({"role": "user", "content": user_content})

        # construct messages with optional system
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(history)

        # send + stream reply
        params = {
            "max_tokens": cur_max,
            "temperature": cur_temp,
            "top_p": DEFAULT_TOP_P,
        }
        print()
        t_total_start = time.time()
        n_tokens = 0
        t_first = None
        full_reply = []
        try:
            for delta in stream_chat(messages, params):
                if t_first is None:
                    t_first = time.time() - t_total_start
                print(delta, end="", flush=True)
                full_reply.append(delta)
                n_tokens += 1
        except KeyboardInterrupt:
            print("\n  (interrupted, partial reply kept in history)")

        elapsed = time.time() - t_total_start
        text = "".join(full_reply)
        history.append({"role": "assistant", "content": text})

        ttft_str = f"TTFT {t_first:.2f}s" if t_first else "TTFT -"
        tps = (n_tokens / elapsed) if elapsed > 0 else 0
        print(f"\n\n  [{n_tokens} chunks in {elapsed:.1f}s = {tps:.2f} chunk/s | {ttft_str}]")

    print("bye.")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
