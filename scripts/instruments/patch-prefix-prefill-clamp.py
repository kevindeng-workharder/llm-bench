"""Patch context_attention_fwd's _fwd_kernel epilogue to clamp acc to
fp16 range before tl.store. This bypasses the fp16 overflow when
acc / (l_i + 1e-10) blows up for tokens with very small l_i.

Two clamp sites:
  1. Main _fwd_kernel epilogue (around line 336)
  2. _fwd_kernel_alibi epilogue (around line 616)
"""
import sys, re
from pathlib import Path

PATH = Path("/home/ubuntu/ai-2.10/lib/python3.13/site-packages/vllm/v1/attention/ops/prefix_prefill.py")
src = PATH.read_text()

# Site 1: main _fwd_kernel
NEEDLE_1 = """    acc = acc / (l_i[:, None] + 1e-10)

    # initialize pointers to output"""

REPL_1 = """    acc = acc / (l_i[:, None] + 1e-10)
    # ---- llm-bench fp16 fix: clamp acc to fp16 range before store ----
    acc = tl.minimum(tl.maximum(acc, -65504.0), 65504.0)
    # ---- end ----

    # initialize pointers to output"""

if NEEDLE_1 not in src:
    print("ERROR site 1 not found", file=sys.stderr); sys.exit(1)
src = src.replace(NEEDLE_1, REPL_1, 1)

# Site 2: _fwd_kernel_alibi
NEEDLE_2 = """    acc = acc / l_i[:, None]"""

REPL_2 = """    acc = acc / l_i[:, None]
    # ---- llm-bench fp16 fix: clamp acc to fp16 range before store ----
    acc = tl.minimum(tl.maximum(acc, -65504.0), 65504.0)
    # ---- end ----"""

if NEEDLE_2 not in src:
    print("ERROR site 2 not found", file=sys.stderr); sys.exit(1)
src = src.replace(NEEDLE_2, REPL_2, 1)

PATH.write_text(src)
print("ok prefix_prefill clamp patch (2 sites)")
