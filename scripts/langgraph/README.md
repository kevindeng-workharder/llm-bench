# `scripts/langgraph/` — multi-engine pipelines via LangGraph

LangGraph workflows that call our two local OpenAI-compatible servers
(see `servers/llamacpp/` and `servers/vllm/`) as graph nodes. Treats
the boundary between vision-capable and reasoning-capable models as
just two endpoints to compose.

Pairs naturally with the **dual-card** server setup:
- card0 (RX 7900 XTX, 24 GB) → llama.cpp Qwen3.6-35B-A3B-MXFP4 on `:8001`
- card1 (RX 7900 XT,  20 GB) → vLLM gemma-4-E2B (multimodal)  on `:8002`

## One-time setup (host)

```bash
# uv (single-binary installer for python + venvs)
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# python 3.12 venv with langgraph + langchain-openai
uv venv --python 3.12 ~/.local/langgraph-venv
VIRTUAL_ENV=~/.local/langgraph-venv uv pip install \
    langgraph langchain-openai langchain-core httpx
```

## Files

| Script | What it does |
|---|---|
| `image-to-report.py` | image → Gemma vision describes it → Qwen3.6 composes a markdown report → Python saves to disk. Three-node graph, deterministic file write. |
| `image-to-report-agent.py` | Same pipeline, but the reasoner uses real **OpenAI-style tool calls** (`write_file`/`read_file`/`list_files`) to drive the file write itself. Requires the llama.cpp launcher with `--jinja` (loads Qwen3.6's native chat template, which has tool-calling support). |
| `ocr-to-json.py` | image with text → Gemma OCR (with `<DONE>` sentinel to suppress trailing hallucinations) → Qwen3.6 detects document type and extracts structured JSON (currency split into `{currency, amount}`, names split into native + latin, quantity strings parsed as ints, etc.). Three-node graph (ocr → structure → save). |
| `dev-agent.py` | 7-tool autonomous developer agent: `read_file` / `write_file` / `list_files` / `glob_files` / `grep` / `run_shell` / `fetch_url`. Free-form natural-language task, ReAct-style loop until LLM stops calling tools. Default task auto-summarises this very repo into `/tmp/repo-summary.md` (verifies tool integration end-to-end with multiple parallel tool_calls + sequential follow-ups). Strips `<think>` blocks from message history per turn so context doesn't blow up after a few turns. |

## Run

Make sure both servers are up (see `../../servers/` launchers and
`scripts/sync-launchers.sh`). Then:

```bash
~/.local/langgraph-venv/bin/python scripts/langgraph/image-to-report.py \
    /tmp/cat.jpg "Cat photo analysis" /tmp/cat-report.md
```

You should see three node prints:

```
[1/3 describe_image] sending cat.jpg (45 KB b64) to Gemma-E2B…
  → 455 chars
  This is a close-up portrait of a tabby cat...
[2/3 compose_report] handing description to Qwen3.6-35B-A3B…
  → 1220 chars
  # Cat photo analysis...
[3/3 save_report] wrote 1220 bytes to /tmp/cat-report.md
```

## Tool calling: chatml vs --jinja

llama.cpp can be launched two ways:

  --chat-template chatml   → simple ChatML, NO tool calling support
  --jinja                  → load the model's native jinja template

The `qwen36-35b-mxfp4-card0-dual.sh` launcher uses `--jinja` so
Qwen3.6's native chat template (with proper OpenAI function-calling
protocol) is active. Verified end-to-end: model emits real OpenAI
`tool_calls` JSON, LangGraph's `ToolNode` executes the function, and
the result feeds back to the LLM until it answers without further
tool calls.

The deterministic non-tool-call variant (`image-to-report.py`) is
still kept around for two reasons:
1. It's strictly more predictable (no agent loop, no recursion limit).
2. If you swap the reasoner to a model with worse tool-call support
   (e.g. gemma-4-E2B), the python-driven version still works.

Note: Gemma-4-E2B's OpenAI function-calling compatibility is known to
be weaker than Qwen3.6 — Google trained Gemma for its own tool format,
not OpenAI's. Keep tool-calling on the Qwen side, file ops there too.

## Caveats specific to this hardware path

- **Qwen3.6 thinking blocks**: the model emits long `<think>…</think>`
  reasoning. The script strips closed blocks; if generation runs out of
  budget mid-thinking it falls back to "find the first H1 in the
  output and start from there". Keep `max_tokens` ≥ 3000 for the
  reasoner to give thinking + output room.
- **Vision encoder on RX 7900 XT (gfx1100)**: Gemma-4 multimodal works
  but lives at the edge of the riscv-cross-compile compatibility
  surface — large images (>2k×2k) may crash the vision tower; resize
  upstream of this script.
- **Concurrency**: each server is configured `--parallel 2`. Don't
  fan-out beyond 2 calls per server in any single graph step.
