#!/usr/bin/env python3
"""LangGraph demo: parallel multi-image album comparison via fan-out.

Pipeline:

    [user supplies N images]
              │
              ▼
   ┌──────────────────────┐
   │ fanout (Send × N)    │
   └──────────┬───────────┘
              │
       ┌──────┼──────┐
       ▼      ▼      ▼      (N parallel branches; vLLM Gemma processes
   describe describe describe up to 2 concurrently — that's the server's
       │      │      │      `--max-num-seqs 2` setting kicking in.)
       └──────┬──────┘
              ▼
      Annotated[list, operator.add]  ← descriptions accumulate
              │
              ▼
   ┌──────────────────────┐
   │ synthesize           │  Qwen3.6 (single call) reads ALL descriptions
   │  (single Qwen call)  │  and writes a comparative markdown report.
   └──────────┬───────────┘
              ▼
            save → END

Demonstrates:
  - LangGraph's `Send` API for fan-out
  - Both servers used concurrently: Gemma on card1 :8002 (vision),
    Qwen on card0 :8001 (reasoning)
  - Per-server N=2 concurrent capacity exercised end-to-end
  - operator.add reducer to merge parallel branch outputs

Usage:
    /home/kevin/.local/langgraph-venv/bin/python parallel-album.py [<img1> ...]

Default: downloads 4 different cat photos to /tmp/album/ and runs.
"""
import base64
import operator
import re
import sys
import time
from pathlib import Path
from typing import Annotated

import httpx
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.types import Send
from typing_extensions import TypedDict


CARD0_URL = "http://localhost:8001/v1"   # Qwen3.6-35B (--jinja --reasoning-format none)
CARD1_URL = "http://localhost:8002/v1"   # gemma-4-E2B (multimodal)


# ─────────── LLMs ───────────
def make_vision_llm():
    return ChatOpenAI(
        base_url=CARD1_URL, api_key="dummy",
        model="gemma4-e2b", temperature=0.0, max_tokens=300,
    )


def make_reasoner_llm():
    return ChatOpenAI(
        base_url=CARD0_URL, api_key="dummy",
        model="qwen36-a3b", temperature=0.0, max_tokens=2500,
    )


# ─────────── State ───────────
class State(TypedDict):
    image_paths: list[str]
    output_path: str
    descriptions: Annotated[list[dict], operator.add]   # accumulator
    summary: str


# ─────────── Nodes ───────────
def fanout(state: State):
    """Emit Send objects, one per image. Each goes to `describe` in parallel."""
    n = len(state["image_paths"])
    print(f"\n[1/3 fanout] dispatching {n} parallel describe() calls to Gemma…")
    return [Send("describe", {"image_path": p}) for p in state["image_paths"]]


def describe(payload: dict) -> dict:
    """Single Gemma call. Runs in parallel with sibling describe() invocations.
    The vLLM server's `--max-num-seqs 2` will batch up to 2 simultaneously."""
    img_path = Path(payload["image_path"]).expanduser()
    if not img_path.exists():
        return {"descriptions": [{"path": str(img_path),
                                   "description": f"[error] file not found"}]}
    img_b64 = base64.b64encode(img_path.read_bytes()).decode()
    mime = "image/jpeg" if img_path.suffix.lower() in (".jpg", ".jpeg") else "image/png"

    t0 = time.time()
    resp = make_vision_llm().invoke([
        SystemMessage(content="You are a precise visual analyst."),
        HumanMessage(content=[
            {"type": "text",
             "text": "Describe this image in 2-3 sentences. Focus on subject, "
                     "mood, dominant colors, and any notable details."},
            {"type": "image_url",
             "image_url": {"url": f"data:{mime};base64,{img_b64}"}},
        ]),
    ])
    elapsed = time.time() - t0
    desc = resp.content or ""
    print(f"  ✓ {img_path.name} ({elapsed:.1f}s)  → {desc[:100]}{'…' if len(desc)>100 else ''}")
    return {"descriptions": [{"path": str(img_path), "description": desc}]}


def synthesize(state: State) -> dict:
    """Qwen reads all parallel-collected descriptions, writes comparison."""
    n = len(state["descriptions"])
    print(f"\n[2/3 synthesize] Qwen3.6 comparing {n} descriptions…")
    rows = "\n".join(
        f"### {Path(d['path']).name}\n{d['description']}"
        for d in state["descriptions"]
    )
    sys_msg = SystemMessage(content=(
        "You write concise comparative analysis reports. Reply with ONLY "
        "the markdown report, no <think> blocks, no preamble, no fences."))
    user_msg = HumanMessage(content=(
        f"You have {n} image descriptions from a vision model:\n\n"
        f"{rows}\n\n"
        f"Compose a markdown report with this skeleton:\n\n"
        f"# Image album comparison ({n} images)\n\n"
        f"## Common themes\n<2-3 bullets shared across the set>\n\n"
        f"## Notable contrasts\n<2-3 bullets where images differ>\n\n"
        f"## Per-image one-liner\n"
        f"| Image | One-liner |\n|---|---|\n"
        f"<one row per image, name | tight one-line summary>\n\n"
        f"Reply with ONLY the markdown."))
    t0 = time.time()
    resp = make_reasoner_llm().invoke([sys_msg, user_msg])
    md = resp.content or ""
    md = re.sub(r"<think>.*?</think>\s*", "", md, flags=re.DOTALL).strip()
    if "<think>" in md and "</think>" not in md:
        idx = md.find("# ")
        md = md[idx:] if idx >= 0 else md.split("<think>", 1)[0].strip()
    if md.startswith("```"):
        md = re.sub(r"^```(?:markdown|md)?\s*\n", "", md)
        md = re.sub(r"\n```\s*$", "", md)
    md = md.strip()
    print(f"  ✓ synthesised in {time.time()-t0:.1f}s, {len(md)} chars")
    return {"summary": md}


def save(state: State) -> dict:
    out = Path(state["output_path"]).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(state["summary"])
    print(f"\n[3/3 save] {out} ({out.stat().st_size} bytes)")
    return {}


# ─────────── Graph ───────────
def build_graph():
    g = StateGraph(State)
    g.add_node("describe", describe)
    g.add_node("synthesize", synthesize)
    g.add_node("save", save)
    # `fanout` is a conditional edge that dispatches Send objects,
    # not a node — so we use add_conditional_edges from START.
    g.set_conditional_entry_point(fanout, ["describe"])
    g.add_edge("describe", "synthesize")
    g.add_edge("synthesize", "save")
    g.add_edge("save", END)
    return g.compile()


# ─────────── Default image set ───────────
DEFAULT_URLS = [
    ("https://images.unsplash.com/photo-1574158622682-e40e69881006?w=400", "cat-tabby"),
    ("https://images.unsplash.com/photo-1543852786-1cf6624b9987?w=400",   "cat-gray"),
    ("https://images.unsplash.com/photo-1518791841217-8f162f1e1131?w=400", "cat-ginger"),
    ("https://images.unsplash.com/photo-1592194996308-7b43878e84a6?w=400", "cat-window"),
]


def ensure_default_album():
    album_dir = Path("/tmp/album"); album_dir.mkdir(exist_ok=True)
    paths = []
    for url, name in DEFAULT_URLS:
        p = album_dir / f"{name}.jpg"
        if not p.exists():
            try:
                r = httpx.get(url, timeout=15, follow_redirects=True)
                r.raise_for_status()
                p.write_bytes(r.content)
                print(f"  downloaded {name}.jpg ({len(r.content)} bytes)")
            except Exception as e:
                print(f"  WARN: download {name} failed: {e}")
                continue
        paths.append(str(p))
    return paths


# ─────────── Main ───────────
if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_paths = sys.argv[1:]
    else:
        print("=== seeding default 4-image cat album ===")
        image_paths = ensure_default_album()
    out = "/tmp/album-report.md"

    if not image_paths:
        print("no images to process", file=sys.stderr); sys.exit(1)

    print("\n=== LangGraph parallel-album ===")
    print(f"  N images: {len(image_paths)}")
    for p in image_paths:
        print(f"    - {p}")
    print(f"  output:    {out}")
    print(f"  vision:    card1 :8002 / gemma4-e2b  (server max-num-seqs=2)")
    print(f"  reasoner:  card0 :8001 / qwen36-a3b")

    t0 = time.time()
    final = build_graph().invoke(
        {"image_paths": image_paths, "output_path": out,
         "descriptions": [], "summary": ""},
        # max_concurrency=2 matches the server's batch capacity. Setting
        # it higher just queues at the server; setting it lower is wasteful.
        config={"max_concurrency": 2},
    )
    elapsed = time.time() - t0

    print(f"\n=== TOTAL ELAPSED: {elapsed:.1f}s ===")
    out_path = Path(out)
    if out_path.exists():
        print(f"--- {out_path} ({out_path.stat().st_size} bytes) ---")
        print(out_path.read_text())
