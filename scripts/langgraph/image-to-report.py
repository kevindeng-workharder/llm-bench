#!/usr/bin/env python3
"""LangGraph demo: image → vision LLM → reasoning LLM → save markdown.

Pipeline:

    [user provides image path + topic + output path]
              │
              ▼
   ┌──────────────────────┐
   │ describe_image       │  Gemma-4-E2B (vision) on card1 :8002
   │  (multimodal call)   │  emits a detailed image description
   └──────────────────────┘
              │
              ▼
   ┌──────────────────────┐
   │ compose_report       │  Qwen3.6-35B-A3B (reasoning) on card0 :8001
   │  (text generation)   │  composes a markdown report from the description
   └──────────────────────┘
              │
              ▼
   ┌──────────────────────┐
   │ save_report          │  pure Python — strips fences, writes to disk
   └──────────────────────┘
              │
              ▼
            END

Why no LLM tool-calling? Local Qwen3.6 on llama.cpp with the chatml chat
template emits broken function-call format under load (the model template
was built for inference text, not strict tool-calling JSON). Keeping the
file write in plain Python keeps the demo deterministic.

Usage:
    /home/kevin/.local/langgraph-venv/bin/python image-to-report.py
        [<image>] [<topic>] [<out_md>]

Defaults: /tmp/cat.jpg, "Daily image catalog entry", /tmp/image-report.md
"""
import base64
import re
import sys
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict


# ─────────── Endpoints ───────────
CARD0_URL = "http://localhost:8001/v1"   # llama.cpp Qwen3.6-35B-A3B
CARD1_URL = "http://localhost:8002/v1"   # vLLM gemma-4-E2B (multimodal)


def make_vision_llm():
    return ChatOpenAI(
        base_url=CARD1_URL,
        api_key="dummy",
        model="gemma4-e2b",
        temperature=0.0,
        max_tokens=400,
    )


def make_reasoner_llm():
    # Qwen3.6 likes to think extensively before answering. Give it room
    # for both the reasoning <think>…</think> block AND the markdown reply.
    return ChatOpenAI(
        base_url=CARD0_URL,
        api_key="dummy",
        model="qwen36-a3b",
        temperature=0.0,
        max_tokens=3000,
    )


# ─────────── State ───────────
class State(TypedDict):
    image_path: str
    topic: str
    output_path: str
    description: str   # filled by describe_image
    report_md: str     # filled by compose_report
    saved_to: str      # filled by save_report


# ─────────── Nodes ───────────
def describe_image(state: State) -> dict:
    """Send image to Gemma, get a description."""
    img_path = Path(state["image_path"]).expanduser()
    if not img_path.exists():
        raise FileNotFoundError(f"image not found: {img_path}")
    img_b64 = base64.b64encode(img_path.read_bytes()).decode()
    mime = "image/jpeg" if img_path.suffix.lower() in (".jpg", ".jpeg") else "image/png"

    print(f"[1/3 describe_image] sending {img_path.name} "
          f"({len(img_b64)//1024} KB b64) to Gemma-E2B…")
    resp = make_vision_llm().invoke([
        SystemMessage(content="You are a precise visual analyst."),
        HumanMessage(content=[
            {"type": "text",
             "text": f"Describe the image in 3-5 sentences. Focus on subject, "
                     f"setting, colors, lighting and notable details. Topic "
                     f"context: {state['topic']}"},
            {"type": "image_url",
             "image_url": {"url": f"data:{mime};base64,{img_b64}"}},
        ]),
    ])
    desc = resp.content
    print(f"  → {len(desc)} chars\n  {desc[:300]}{'…' if len(desc)>300 else ''}\n")
    return {"description": desc}


def compose_report(state: State) -> dict:
    """Qwen turns the description into a structured markdown report."""
    print(f"[2/3 compose_report] handing description to Qwen3.6-35B-A3B…")
    sys_msg = SystemMessage(content=(
        "You are a report writer. Reply with ONLY the markdown content of "
        "the report — no <think> blocks, no preamble, no code fences. The "
        "first line must be the H1 heading."))
    user_msg = HumanMessage(content=(
        f"Compose a markdown report titled '{state['topic']}'. Use this "
        f"exact skeleton:\n\n"
        f"# {state['topic']}\n\n"
        f"## Image\n<one paragraph describing what is shown, based on the description below>\n\n"
        f"## Observations\n- bullet 1\n- bullet 2\n- bullet 3 (3-5 bullets)\n\n"
        f"## Suggested next steps\n- bullet 1\n- bullet 2 (2-3 bullets)\n\n"
        f"Image description from the vision model:\n\n"
        f"{state['description']}\n\n"
        f"Reply with the markdown only, starting with `# {state['topic']}`."))
    resp = make_reasoner_llm().invoke([sys_msg, user_msg])
    md = resp.content
    # Strip <think>…</think> if Qwen leaks it (closed form)
    md = re.sub(r"<think>.*?</think>", "", md, flags=re.DOTALL)
    # Truncated thinking (no </think>): drop everything before the first H1
    if "<think>" in md and "</think>" not in md:
        m = re.search(r"^# ", md, flags=re.MULTILINE)
        if m:
            md = md[m.start():]
        else:
            # Couldn't find the report — return what we have minus the <think>
            md = md.split("<think>", 1)[0].strip() or md
    md = md.strip()
    # Strip ``` fences if Qwen wrapped the whole thing in a code block
    if md.startswith("```"):
        md = re.sub(r"^```(?:markdown|md)?\n", "", md)
        md = re.sub(r"\n```\s*$", "", md)
    md = md.strip()
    print(f"  → {len(md)} chars\n  {md[:400]}{'…' if len(md)>400 else ''}\n")
    return {"report_md": md}


def save_report(state: State) -> dict:
    """Plain Python — write the report to disk."""
    out = Path(state["output_path"]).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(state["report_md"])
    print(f"[3/3 save_report] wrote {out.stat().st_size} bytes to {out}")
    return {"saved_to": str(out)}


# ─────────── Graph ───────────
def build_graph():
    g = StateGraph(State)
    g.add_node("describe", describe_image)
    g.add_node("compose",  compose_report)
    g.add_node("save",     save_report)
    g.set_entry_point("describe")
    g.add_edge("describe", "compose")
    g.add_edge("compose",  "save")
    g.add_edge("save",     END)
    return g.compile()


# ─────────── Main ───────────
if __name__ == "__main__":
    img   = sys.argv[1] if len(sys.argv) > 1 else "/tmp/cat.jpg"
    topic = sys.argv[2] if len(sys.argv) > 2 else "Daily image catalog entry"
    out   = sys.argv[3] if len(sys.argv) > 3 else "/tmp/image-report.md"

    if not Path(img).expanduser().exists():
        print(f"image not found: {img}", file=sys.stderr)
        print("usage: image-to-report.py [<image>] [<topic>] [<out_md>]",
              file=sys.stderr)
        sys.exit(1)

    print("=== LangGraph demo: image → markdown report ===")
    print(f"  image:  {img}")
    print(f"  topic:  {topic}")
    print(f"  output: {out}")
    print(f"  vision: card1 :8002 / gemma4-e2b")
    print(f"  writer: card0 :8001 / qwen36-a3b\n")

    final = build_graph().invoke({
        "image_path":  img,
        "topic":       topic,
        "output_path": out,
        "description": "",
        "report_md":   "",
        "saved_to":    "",
    })

    print(f"=== FINAL ===")
    saved = Path(final["saved_to"])
    if saved.exists():
        print(f"--- {saved} ({saved.stat().st_size} bytes) ---")
        print(saved.read_text())
    else:
        print("WARN: nothing saved")
        sys.exit(2)
