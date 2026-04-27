"""Apply fp16-safe clamp before tl.store in ALL triton attention kernel
epilogues across vLLM 0.19's v1 attention ops directory.

Targets:
  prefix_prefill.py (already done in earlier patch — skip if already there)
  triton_prefill_attention.py
  triton_decode_attention.py
  chunked_prefill_paged_decode.py (kernel_paged_attention_2d)
"""
import sys, re
from pathlib import Path

CLAMP = "    # ---- llm-bench fp16 fix: clamp acc to fp16 range before store ----\n    acc = tl.minimum(tl.maximum(acc, -65504.0), 65504.0)\n    # ---- end ----\n"
INDENT8 = "        "

PATCHES = [
    # (path, needle, repl_after_pattern_or_none)
    (
        "/home/ubuntu/ai-2.10/lib/python3.13/site-packages/vllm/v1/attention/ops/triton_prefill_attention.py",
        "    acc = acc / l_i[:, None]\n",
        "    acc = acc / l_i[:, None]\n" + CLAMP,
    ),
    # chunked_prefill_paged_decode.py: kernel_paged_attention_2d epilogue
    # line 233: "    acc = acc / (L[:, None] + 1e-10)"
    (
        "/home/ubuntu/ai-2.10/lib/python3.13/site-packages/vllm/v1/attention/ops/chunked_prefill_paged_decode.py",
        "    acc = acc / (L[:, None] + 1e-10)\n",
        "    acc = acc / (L[:, None] + 1e-10)\n" + CLAMP,
    ),
]

# triton_decode_attention.py has 3 sites with `acc / e_sum`. They appear
# inline as args to tl.store(... acc / e_sum ...) which is harder to clamp
# inline. We'll use a regex replacement: `acc / e_sum,` followed by a comma
# (it's a tl.store arg). Replace with a clamped expression.

DECODE_PATH = "/home/ubuntu/ai-2.10/lib/python3.13/site-packages/vllm/v1/attention/ops/triton_decode_attention.py"

for path, needle, repl in PATCHES:
    p = Path(path)
    if not p.exists():
        print(f"SKIP {p}: not found")
        continue
    src = p.read_text()
    if "llm-bench fp16 fix" in src:
        print(f"SKIP {p}: already patched")
        continue
    if needle not in src:
        print(f"WARN {p}: needle not found")
        continue
    p.write_text(src.replace(needle, repl, 1))
    print(f"OK {p}")

# triton_decode_attention.py — replace `acc / e_sum` with clamped expr inline
p = Path(DECODE_PATH)
if p.exists():
    src = p.read_text()
    if "llm-bench fp16 fix" not in src:
        # Pattern: acc / e_sum,    (single line, used inside tl.store)
        # Pattern: acc / e_sum[:, None],
        new = src.replace(
            "acc / e_sum,",
            "tl.minimum(tl.maximum(acc / e_sum, -65504.0), 65504.0),  # llm-bench fp16 fix",
        )
        new = new.replace(
            "acc / e_sum[:, None],",
            "tl.minimum(tl.maximum(acc / e_sum[:, None], -65504.0), 65504.0),  # llm-bench fp16 fix",
        )
        if new != src:
            p.write_text(new)
            print(f"OK {p} (inline replacements)")
        else:
            print(f"WARN {p}: no inline match")
    else:
        print(f"SKIP {p}: already patched")

print("done")
