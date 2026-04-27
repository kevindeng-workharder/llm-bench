"""v2 clamp: when l_i is degenerate (very small), zero out the row.
Also use tl.where (NaN-safe) instead of tl.minimum/tl.maximum.

Hypothesis:
  - When l_i for a row is tiny (e.g. 1e-30), acc/(l_i+1e-10) blows up
    despite the +1e-10 epsilon, producing fp16-clamped 65504.
  - That 65504 in attn_out then makes o_proj (Linear matmul with
    fp16 weights summed over 4096 dims) overflow to fp16 inf in
    o_proj's output.
  - Real fix: when l_i is degenerate, the softmax denominator is
    meaningless, so set the row's attn output to 0.

Patches the 4 sites with our existing block clamp:
  - prefix_prefill.py _fwd_kernel + _fwd_kernel_alibi
  - triton_prefill_attention.py _fwd_kernel
  - chunked_prefill_paged_decode.py kernel_paged_attention_2d
"""
import sys, re
from pathlib import Path

OLD = """    # ---- llm-bench fp16 fix: clamp acc to fp16 range before store ----
    acc = tl.minimum(tl.maximum(acc, -65504.0), 65504.0)
    # ---- end ----
"""

# Two flavors of the new fix depending on the variable holding the
# softmax denominator:
NEW_LI = """    # ---- llm-bench fp16 fix v2: zero degenerate rows + NaN-safe clamp ----
    # When the softmax denominator is tiny, the row was effectively masked
    # out. Set its attn output to 0 instead of letting acc/(l+eps) blow up.
    acc = tl.where(l_i[:, None] < 1e-3, 0.0, acc)
    acc = tl.where(acc != acc, 0.0, acc)              # NaN -> 0
    acc = tl.where(acc > 65504.0, 65504.0, acc)
    acc = tl.where(acc < -65504.0, -65504.0, acc)
    # ---- end ----
"""

NEW_L = """    # ---- llm-bench fp16 fix v2: zero degenerate rows + NaN-safe clamp ----
    acc = tl.where(L[:, None] < 1e-3, 0.0, acc)
    acc = tl.where(acc != acc, 0.0, acc)
    acc = tl.where(acc > 65504.0, 65504.0, acc)
    acc = tl.where(acc < -65504.0, -65504.0, acc)
    # ---- end ----
"""

PATHS_LI = [
    "/home/ubuntu/ai-2.10/lib/python3.13/site-packages/vllm/v1/attention/ops/prefix_prefill.py",
    "/home/ubuntu/ai-2.10/lib/python3.13/site-packages/vllm/v1/attention/ops/triton_prefill_attention.py",
]

PATHS_L = [
    "/home/ubuntu/ai-2.10/lib/python3.13/site-packages/vllm/v1/attention/ops/chunked_prefill_paged_decode.py",
]

for p in PATHS_LI:
    pp = Path(p)
    src = pp.read_text()
    if OLD not in src:
        print(f"SKIP {pp.name}: old clamp not found"); continue
    n = src.count(OLD)
    pp.write_text(src.replace(OLD, NEW_LI))
    print(f"OK   {pp.name}: replaced {n} site(s) with v2 (l_i variant)")

for p in PATHS_L:
    pp = Path(p)
    src = pp.read_text()
    if OLD not in src:
        print(f"SKIP {pp.name}: old clamp not found"); continue
    n = src.count(OLD)
    pp.write_text(src.replace(OLD, NEW_L))
    print(f"OK   {pp.name}: replaced {n} site(s) with v2 (L variant)")

print("done")
