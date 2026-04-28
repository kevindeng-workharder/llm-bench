#!/usr/bin/env python3
"""LangGraph dev-agent: 7-tool autonomous developer assistant.

Built on the local Qwen3.6-35B-A3B server (--jinja for tool calling).
Demonstrates a real dev workflow: agent explores files, runs shell
commands, optionally fetches URLs, and writes reports.

Tools available to the agent:
  read_file(path)             read file content (capped at 20 KB)
  write_file(path, content)   overwrite a file
  list_files(path)            ls a directory with file sizes
  glob_files(root, pattern)   recursive glob, e.g. '*.py'
  grep(pattern, path)         grep -rIn for a regex
  run_shell(cmd)              free-form shell with 60s timeout (DANGEROUS)
  fetch_url(url)              http(s) GET, body capped at 20 KB

Usage:
    /home/kevin/.local/langgraph-venv/bin/python dev-agent.py [<task>]

Default task: summarise the llm-bench repo into /tmp/repo-summary.md
"""
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import httpx
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict


CARD0_URL = "http://localhost:8001/v1"   # llama.cpp Qwen3.6 (--jinja)
MAX_TURNS = 30


# ─────────── Tools ───────────
@tool
def read_file(path: str) -> str:
    """Read file at `path`; return up to 20 KB of contents."""
    p = Path(path).expanduser()
    if not p.exists():
        return f"file not found: {p}"
    if p.is_dir():
        return f"is a directory, not a file: {p}"
    try:
        data = p.read_text(errors="replace")
    except Exception as e:
        return f"read failed: {e}"
    if len(data) > 20_000:
        return data[:20_000] + f"\n\n[truncated, file is {len(data)} bytes total]"
    return data


@tool
def write_file(path: str, content: str) -> str:
    """Overwrite the file at `path` with `content`."""
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return f"wrote {len(content)} bytes to {p}"


@tool
def list_files(path: str) -> str:
    """List entries in directory `path` with file sizes."""
    p = Path(path).expanduser()
    if not p.is_dir():
        return f"not a directory: {p}"
    rows = []
    for c in sorted(p.iterdir()):
        if c.is_dir():
            rows.append(f"  {c.name}/  (dir)")
        else:
            try:
                rows.append(f"  {c.name}  ({c.stat().st_size}B)")
            except Exception:
                rows.append(f"  {c.name}  (?)")
    return "\n".join(rows) or "(empty)"


@tool
def glob_files(root: str, pattern: str) -> str:
    """Recursive glob `pattern` (e.g. '*.py') under `root`. Cap 50 results."""
    p = Path(root).expanduser()
    if not p.is_dir():
        return f"not a directory: {p}"
    matches = sorted(p.rglob(pattern))
    if len(matches) > 50:
        return "\n".join(str(m) for m in matches[:50]) + \
               f"\n[+ {len(matches) - 50} more matches not shown]"
    return "\n".join(str(m) for m in matches) or "no matches"


@tool
def grep(pattern: str, path: str) -> str:
    """`grep -rIn <pattern> <path>` for a regex. Cap 30 lines."""
    try:
        r = subprocess.run(
            ["grep", "-rIn", "--include=*", pattern, str(Path(path).expanduser())],
            capture_output=True, text=True, timeout=30,
        )
    except subprocess.TimeoutExpired:
        return "grep timed out"
    except Exception as e:
        return f"grep failed: {e}"
    out = r.stdout.strip().splitlines()
    if len(out) > 30:
        return "\n".join(out[:30]) + f"\n[+ {len(out) - 30} more]"
    return "\n".join(out) or "no matches"


@tool
def run_shell(cmd: str) -> str:
    """Run a shell command (60s timeout). DANGEROUS — full shell access.
    Output capped at 10 KB."""
    try:
        r = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=60,
            env={**os.environ, "PAGER": "cat", "GIT_PAGER": "cat"},
        )
    except subprocess.TimeoutExpired:
        return "command timed out (60s)"
    out = (
        f"[exit={r.returncode}]\n"
        f"--- stdout ---\n{r.stdout}\n"
        f"--- stderr ---\n{r.stderr}"
    )
    return out if len(out) <= 10_000 else out[:10_000] + "\n[truncated]"


@tool
def fetch_url(url: str) -> str:
    """HTTP(S) GET — return body (capped at 20 KB)."""
    if not url.startswith(("http://", "https://")):
        return f"only http/https URLs supported, got: {url}"
    try:
        r = httpx.get(url, timeout=15, follow_redirects=True)
    except Exception as e:
        return f"fetch failed: {e}"
    body = r.text
    head = (
        f"[status={r.status_code} "
        f"content-type={r.headers.get('content-type','?')} "
        f"len={len(r.text)}]\n"
    )
    if len(body) > 20_000:
        body = body[:20_000] + f"\n\n[truncated, {len(r.text)} total bytes]"
    return head + body


tools = [read_file, write_file, list_files, glob_files, grep, run_shell, fetch_url]


# ─────────── LLM ───────────
def make_agent_llm():
    return ChatOpenAI(
        base_url=CARD0_URL, api_key="dummy",
        model="qwen36-a3b", temperature=0.0, max_tokens=3000,
    ).bind_tools(tools)


# ─────────── State + nodes ───────────
class State(TypedDict):
    messages: Annotated[list, add_messages]


def call_llm(state: State) -> dict:
    print(f"\n--- agent turn (msgs={len(state['messages'])}) ---")
    resp = make_agent_llm().invoke(state["messages"])
    tcs = getattr(resp, "tool_calls", None) or []
    if tcs:
        for tc in tcs:
            args_str = str(tc.get("args", ""))[:140]
            print(f"  tool_call: {tc.get('name')}({args_str})")
    else:
        clean = re.sub(r"<think>.*?</think>", "", resp.content or "", flags=re.DOTALL).strip()
        print(f"  reply: {clean[:240]}{'…' if len(clean)>240 else ''}")
    # CRITICAL: strip <think>…</think> BEFORE appending to state — otherwise
    # every subsequent turn re-ingests the full reasoning trace and the
    # context window blows up after ~5 turns. We keep the original response
    # content but rewrite it without the thinking block.
    if resp.content:
        clean_content = re.sub(r"<think>.*?</think>\s*", "", resp.content,
                               flags=re.DOTALL).strip()
        # Also handle truncated <think> (no closing tag) — just delete the
        # opener and everything after it that doesn't look like real content
        if "<think>" in clean_content and "</think>" not in clean_content:
            clean_content = clean_content.split("<think>", 1)[0].strip()
        # Mutate the response. langchain's AIMessage is a pydantic model and
        # supports attribute assignment via the validator.
        try:
            resp.content = clean_content
        except Exception:
            # Some langchain versions need model_copy; fall through if mutation fails
            pass
    return {"messages": [resp]}


def should_continue(state: State) -> str:
    return "tools" if (getattr(state["messages"][-1], "tool_calls", None) or []) else END


def build_graph():
    g = StateGraph(State)
    g.add_node("llm", call_llm)
    g.add_node("tools", ToolNode(tools))
    g.set_entry_point("llm")
    g.add_conditional_edges("llm", should_continue)
    g.add_edge("tools", "llm")
    return g.compile()


# ─────────── Default task ───────────
DEFAULT_TASK = (
    "Explore the directory /home/kevin/llm-bench. Workflow:\n"
    "  1. list_files /home/kevin/llm-bench to see top-level layout\n"
    "  2. read_file /home/kevin/llm-bench/README.md for the headline overview\n"
    "  3. list_files /home/kevin/llm-bench/scripts to see the script categories\n"
    "  4. glob_files /home/kevin/llm-bench/servers '*.sh' to count launchers\n"
    "  5. run_shell 'cd /home/kevin/llm-bench && git log --oneline -8' for recent commits\n"
    "  6. write_file /tmp/repo-summary.md with a markdown report containing:\n"
    "       # llm-bench summary\n"
    "       ## What it is\n       ## Layout\n       ## Recent activity\n"
    "       ## Notable artifacts\n\n"
    "Stop after the file is written. Reply with one short confirmation."
)


if __name__ == "__main__":
    task = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TASK

    sys_msg = SystemMessage(content=(
        "You are an autonomous developer assistant with these tools: "
        "read_file, write_file, list_files, glob_files, grep, run_shell, "
        "fetch_url. Use them step by step to complete the user's task. "
        "Be concise — don't over-explore; once you have enough info to "
        "write the report, do it and stop. NEVER produce <think> blocks "
        "in your final reply (only in intermediate reasoning if needed). "
        "After completing, give a 1-line confirmation and stop calling "
        "tools."))

    print("=== LangGraph dev-agent ===")
    print(f"Task: {task[:200]}{'…' if len(task)>200 else ''}")
    print(f"Tools: {[t.name for t in tools]}\n")

    final = build_graph().invoke(
        {"messages": [sys_msg, HumanMessage(content=task)]},
        config={"recursion_limit": MAX_TURNS},
    )

    print("\n=== STATS ===")
    n_tool_calls = sum(
        len(getattr(m, "tool_calls", None) or [])
        for m in final["messages"]
    )
    print(f"  total messages    : {len(final['messages'])}")
    print(f"  total tool calls  : {n_tool_calls}")

    print("\n=== FINAL REPLY ===")
    last = final["messages"][-1]
    clean = re.sub(r"<think>.*?</think>", "", last.content or "", flags=re.DOTALL).strip()
    print(clean or "(empty reply)")
