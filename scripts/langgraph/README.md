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
| `image-to-report.py` | image → Gemma vision describes it → Qwen3.6 composes a markdown report → Python saves to disk. Three-node graph. |

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

## Why no LLM-driven `write_file` tool calling

The earliest version of `image-to-report.py` had Qwen3.6 invoke a
`write_file(path, content)` tool to save the report itself. Reality:
Qwen3.6 on llama.cpp's chatml template emits broken tool-call format
(unmatched `</parameter>` tags, ChatML/Hermes mix), and the small
gemma-4-E2B is even worse at OpenAI-style function calling. Keeping
the file write in pure Python (`save_report` node) is deterministic
and side-steps all of that. If you need the LLM to drive arbitrary
tool calls, run llama.cpp with `--jinja` to use the model's own
function-calling chat template, or switch to a vLLM-served reasoner
that handles tool calling reliably (e.g. Qwen3-30B-AWQ).

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
