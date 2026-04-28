#!/usr/bin/env python3
"""LangGraph demo: image with text → Gemma OCR → Qwen3.6 structured JSON.

Pipeline:

   [user supplies an image with text in it]
              │
              ▼
   ┌──────────────────────┐
   │ ocr_stage            │  Gemma-4-E2B (vision) on card1 :8002
   │  multimodal call     │  transcribes every visible line of text.
   │  trailing-hallucin.  │  Uses a `<DONE>` sentinel so we can chop off
   │  guard               │  the model's tendency to keep generating
   │                      │  after the real text ends.
   └──────────────────────┘
              │
              ▼
   ┌──────────────────────┐
   │ structure_stage      │  Qwen3.6-35B-A3B on card0 :8001
   │  text → JSON         │  detects document type, extracts fields,
   │                      │  emits a single JSON object.
   └──────────────────────┘
              │
              ▼
   structured JSON dict (printed + optionally written to disk)

Usage:
    /home/kevin/.local/langgraph-venv/bin/python ocr-to-json.py [<image>] [<out_json>]

Example:
    ocr-to-json.py /tmp/ocr-test-mixed.png /tmp/ocr-result.json
"""
import base64
import json
import re
import sys
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict


CARD0_URL = "http://localhost:8001/v1"   # llama.cpp Qwen3.6-35B-A3B (--jinja)
CARD1_URL = "http://localhost:8002/v1"   # vLLM gemma-4-E2B (multimodal)


def make_vision_llm():
    return ChatOpenAI(
        base_url=CARD1_URL,
        api_key="dummy",
        model="gemma4-e2b",
        temperature=0.0,
        max_tokens=500,
        # Gemma 知道遇到 <DONE> 该停;若没出现 <DONE>，我们会在代码里再 split 一次
        stop=["<DONE>"],
    )


def make_reasoner_llm():
    return ChatOpenAI(
        base_url=CARD0_URL,
        api_key="dummy",
        model="qwen36-a3b",
        temperature=0.0,
        max_tokens=2500,
    )


# ─────────── State ───────────
class State(TypedDict):
    image_path: str
    raw_ocr: str
    structured: dict
    output_path: str  # optional


# ─────────── Nodes ───────────
def ocr_stage(state: State) -> dict:
    img_path = Path(state["image_path"]).expanduser()
    if not img_path.exists():
        raise FileNotFoundError(f"image not found: {img_path}")
    img_b64 = base64.b64encode(img_path.read_bytes()).decode()
    mime = {
        ".jpg":  "image/jpeg",  ".jpeg": "image/jpeg",
        ".png":  "image/png",   ".gif":  "image/gif",
        ".webp": "image/webp",
    }.get(img_path.suffix.lower(), "image/png")

    print(f"[1/2 ocr_stage] sending {img_path.name} to Gemma-E2B…")
    resp = make_vision_llm().invoke([
        HumanMessage(content=[
            {"type": "text",
             "text":
                "Transcribe ALL visible text in this image, line by line, "
                "exactly as written. Preserve every character including "
                "punctuation, symbols, numbers, and non-Latin scripts "
                "(e.g. Chinese). Maintain the original line breaks. "
                "After the LAST visible line, output the literal token "
                "`<DONE>` and stop. Do NOT add commentary."},
            {"type": "image_url",
             "image_url": {"url": f"data:{mime};base64,{img_b64}"}},
        ]),
    ])
    text = resp.content or ""
    # Belt + braces: even if `stop=[<DONE>]` worked at the API, also strip
    # programmatically. Some inference engines emit the stop token before
    # cutting off, others don't.
    text = text.split("<DONE>")[0].strip()
    print(f"  → {len(text)} chars\n  {text[:400]}{'…' if len(text)>400 else ''}\n")
    return {"raw_ocr": text}


def structure_stage(state: State) -> dict:
    print(f"[2/2 structure_stage] Qwen3.6 → JSON…")
    sys_msg = SystemMessage(content=(
        "You convert raw OCR text into a single structured JSON object. "
        "Reply with ONLY the JSON — no <think> blocks, no preamble, no "
        "markdown fences, no commentary, no trailing notes. Start with `{` "
        "and end with `}`. Detect the document type from the content and "
        "pick descriptive field names that match what is actually present "
        "(do not invent fields that aren't in the OCR)."))
    user_msg = HumanMessage(content=(
        f"Raw OCR text:\n\n```\n{state['raw_ocr']}\n```\n\n"
        f"Produce a JSON object with at least these top-level keys:\n"
        f"  - document_type: short string identifying the document kind "
        f"(e.g. \"invoice\", \"contact_card\", \"note\", \"receipt\")\n"
        f"  - extracted: nested object containing the extracted fields. "
        f"Use sensible nested structure (e.g. dollar amounts as "
        f"{{\"currency\": ..., \"amount\": ...}}, names with original "
        f"script preserved).\n"
        f"  - raw_lines: array of strings, the OCR lines as-is.\n\n"
        f"Reply with the JSON object only."))
    resp = make_reasoner_llm().invoke([sys_msg, user_msg])
    text = resp.content or ""
    # Strip <think>…</think> (closed) — uncloseed-think handled below
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # If only opening <think> survived (truncation), drop the prefix up to
    # the first `{`.
    if "<think>" in text and "</think>" not in text:
        idx = text.find("{")
        text = text[idx:] if idx >= 0 else text
    # Strip ``` code fences
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n", "", text)
        text = re.sub(r"\n```\s*$", "", text)
    text = text.strip()

    try:
        parsed = json.loads(text)
        print(f"  → JSON OK\n{json.dumps(parsed, indent=2, ensure_ascii=False)[:1000]}")
    except json.JSONDecodeError as e:
        print(f"  → JSON parse FAILED at line {e.lineno} col {e.colno}: {e.msg}")
        print(f"  → raw output:\n{text[:500]}{'…' if len(text)>500 else ''}")
        parsed = {"_parse_error": str(e), "_raw": text}

    return {"structured": parsed}


def maybe_save(state: State) -> dict:
    if not state.get("output_path"):
        return {}
    out = Path(state["output_path"]).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(state["structured"], indent=2, ensure_ascii=False))
    print(f"\n[3/3 saved] {out} ({out.stat().st_size} bytes)")
    return {}


# ─────────── Graph ───────────
def build_graph():
    g = StateGraph(State)
    g.add_node("ocr",        ocr_stage)
    g.add_node("structure",  structure_stage)
    g.add_node("save",       maybe_save)
    g.set_entry_point("ocr")
    g.add_edge("ocr",        "structure")
    g.add_edge("structure",  "save")
    g.add_edge("save",       END)
    return g.compile()


# ─────────── Main ───────────
if __name__ == "__main__":
    img = sys.argv[1] if len(sys.argv) > 1 else "/tmp/ocr-test-mixed.png"
    out = sys.argv[2] if len(sys.argv) > 2 else ""

    if not Path(img).expanduser().exists():
        print(f"image not found: {img}", file=sys.stderr); sys.exit(1)

    print("=== LangGraph: image → OCR → structured JSON ===")
    print(f"  image:  {img}")
    if out:
        print(f"  output: {out}")
    print(f"  vision: card1 :8002 / gemma4-e2b")
    print(f"  parser: card0 :8001 / qwen36-a3b\n")

    final = build_graph().invoke({
        "image_path":  img,
        "raw_ocr":     "",
        "structured":  {},
        "output_path": out,
    })

    print(f"\n=== STRUCTURED ===")
    print(json.dumps(final["structured"], indent=2, ensure_ascii=False))
