#!/usr/bin/env python3
"""LangGraph demo: image → vision LLM → reasoning agent (tool-calling).

Like image-to-report.py, but the reasoning model uses real tool calls
to save the file (and can choose to read other files, list dirs, etc.)
instead of having Python do it deterministically.

Requires the llama.cpp launcher to use ``--jinja`` so Qwen3.6's native
chat template (with proper OpenAI function-calling support) is loaded.
With the default ``--chat-template chatml`` the function-call format
is broken and this script will loop / fail.

Pipeline:

    [user provides image path + topic + output path]
              │
              ▼
   describe_image (Gemma-4-E2B vision)
              │
              ▼
   reason_and_act (Qwen3.6-35B with tools)  ◄────┐
        │                                          │
        ├── if .tool_calls present ────► tools ────┘  (loop until LLM stops)
        │
        └── final answer ────► END

Usage:
    /home/kevin/.local/langgraph-venv/bin/python image-to-report-agent.py
        [<image>] [<topic>] [<out_md>]
"""
import base64
import re
import sys
from pathlib import Path
from typing import Annotated

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict


CARD0_URL = "http://localhost:8001/v1"   # llama.cpp Qwen3.6-35B-A3B (--jinja)
CARD1_URL = "http://localhost:8002/v1"   # vLLM gemma-4-E2B (multimodal)


# ─────────── Tools ───────────
@tool
def write_file(path: str, content: str) -> str:
    """Write `content` to the file at `path`. Overwrites if it exists."""
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return f"wrote {len(content)} bytes to {p}"


@tool
def read_file(path: str) -> str:
    """Read the file at `path` and return its contents."""
    return Path(path).expanduser().read_text()


@tool
def list_files(path: str) -> str:
    """List the files in directory `path`."""
    p = Path(path).expanduser()
    return "\n".join(sorted(c.name for c in p.iterdir()))


tools = [write_file, read_file, list_files]


# ─────────── LLMs ───────────
def make_vision_llm():
    return ChatOpenAI(
        base_url=CARD1_URL, api_key="dummy",
        model="gemma4-e2b", temperature=0.0, max_tokens=400,
    )


def make_agent_llm():
    return ChatOpenAI(
        base_url=CARD0_URL, api_key="dummy",
        model="qwen36-a3b", temperature=0.0, max_tokens=3000,
    ).bind_tools(tools)


# ─────────── State ───────────
class State(TypedDict):
    image_path: str
    topic: str
    output_path: str
    description: str
    messages: Annotated[list, add_messages]


# ─────────── Nodes ───────────
def describe_image(state: State) -> dict:
    img_path = Path(state["image_path"]).expanduser()
    if not img_path.exists():
        raise FileNotFoundError(f"image not found: {img_path}")
    img_b64 = base64.b64encode(img_path.read_bytes()).decode()
    mime = "image/jpeg" if img_path.suffix.lower() in (".jpg", ".jpeg") else "image/png"

    print(f"[1/N describe_image] sending {img_path.name} to Gemma-E2B…")
    resp = make_vision_llm().invoke([
        SystemMessage(content="You are a precise visual analyst."),
        HumanMessage(content=[
            {"type": "text",
             "text": f"Describe the image in 3-5 sentences. Topic: {state['topic']}"},
            {"type": "image_url",
             "image_url": {"url": f"data:{mime};base64,{img_b64}"}},
        ]),
    ])
    desc = resp.content
    print(f"  → {len(desc)} chars\n  {desc[:240]}{'…' if len(desc)>240 else ''}\n")

    # Seed the agent loop with a system prompt + the task description
    sys_msg = SystemMessage(content=(
        "You are a report-writing agent. You have these tools available: "
        "write_file, read_file, list_files. To complete the user's request "
        "you MUST call write_file with the markdown report as `content`. "
        "After the tool returns successfully, reply with ONE short "
        "confirmation sentence. Never paste the markdown into your final "
        "reply — that's a failure mode. Strip any <think> blocks before "
        "your final reply."))
    user_msg = HumanMessage(content=(
        f"Image description from the vision model:\n\n{desc}\n\n"
        f"Now compose a markdown report with this exact skeleton:\n\n"
        f"# {state['topic']}\n\n"
        f"## Image\n<one paragraph based on the description>\n\n"
        f"## Observations\n- 3-5 bullets\n\n"
        f"## Suggested next steps\n- 2-3 bullets\n\n"
        f"Save it to {state['output_path']!r} via the write_file tool, "
        f"then confirm in one sentence."))
    return {"description": desc, "messages": [sys_msg, user_msg]}


def reason_and_act(state: State) -> dict:
    n_calls = sum(1 for m in state["messages"]
                  if getattr(m, "tool_calls", None))
    print(f"[2/N reason_and_act] turn={len(state['messages'])} tool_calls_so_far={n_calls}")
    resp = make_agent_llm().invoke(state["messages"])
    tc = getattr(resp, "tool_calls", None) or []
    if tc:
        for c in tc:
            args_str = str(c.get("args", c.get("function", {}).get("arguments", "")))[:80]
            print(f"  → tool_call: {c.get('name', c.get('function', {}).get('name'))}({args_str})")
    else:
        # Strip <think>…</think> from final reply for clean console output
        content = resp.content or ""
        clean = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        print(f"  → final reply: {clean[:200]}{'…' if len(clean)>200 else ''}")
    return {"messages": [resp]}


def should_continue(state: State) -> str:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END


# ─────────── Graph ───────────
def build_graph():
    g = StateGraph(State)
    g.add_node("describe", describe_image)
    g.add_node("reason",   reason_and_act)
    g.add_node("tools",    ToolNode(tools))
    g.set_entry_point("describe")
    g.add_edge("describe", "reason")
    g.add_conditional_edges("reason", should_continue)
    g.add_edge("tools", "reason")
    return g.compile()


# ─────────── Main ───────────
if __name__ == "__main__":
    img   = sys.argv[1] if len(sys.argv) > 1 else "/tmp/cat.jpg"
    topic = sys.argv[2] if len(sys.argv) > 2 else "Daily image catalog entry"
    out   = sys.argv[3] if len(sys.argv) > 3 else "/tmp/image-report-agent.md"

    if not Path(img).expanduser().exists():
        print(f"image not found: {img}", file=sys.stderr); sys.exit(1)

    print("=== LangGraph agent demo: image → tool-calling reasoner ===")
    print(f"  image:  {img}")
    print(f"  topic:  {topic}")
    print(f"  output: {out}")
    print(f"  vision: card1 :8002 / gemma4-e2b")
    print(f"  agent : card0 :8001 / qwen36-a3b (--jinja for tool calls)\n")

    final = build_graph().invoke({
        "image_path":  img,
        "topic":       topic,
        "output_path": out,
        "description": "",
        "messages":    [],
    }, config={"recursion_limit": 25})

    print(f"\n=== FINAL ===")
    out_path = Path(out).expanduser()
    if out_path.exists():
        print(f"--- {out_path} ({out_path.stat().st_size} bytes) ---")
        print(out_path.read_text())
    else:
        print(f"WARN: {out_path} doesn't exist — agent never called write_file")
        sys.exit(2)
