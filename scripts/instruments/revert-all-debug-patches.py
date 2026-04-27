"""Revert ALL debug patches applied during the N=8 fp16 batched-NaN
investigation, restoring the venv to its post-riscv-cross-compile state
(i.e. the state before any of our clamp / force-triton / probe patches).

Patches removed:
  1. qwen3.py             N8PROBE / N8STATS instrumentation, layer-idx capture
  2. prefix_prefill.py    fp16 clamp before tl.store (2 sites, v1 or v2)
  3. triton_prefill_attention.py  fp16 clamp before tl.store (1 site)
  4. triton_decode_attention.py   fp16 clamp wrapping `acc / e_sum` (3 sites)
  5. chunked_prefill_paged_decode.py
        - fp16 clamp before tl.store
        - `use_custom = False` (force-triton path)

Both clamp variants are recognized:
  - v1 (initial): tl.minimum(tl.maximum(acc, -65504.0), 65504.0) — fixed N=1/2/4
                  but NOT N=8 (degenerate-row overflow not caught).
  - v2 (the actual fix): zero rows where l_i < 1e-3 + NaN-safe tl.where clamp
                  — fixes N=1/2/4/8 cleanly.

Strategy:
  - Where a `.before-clamp` / `.bak-before-fix-20260424` /
    `.before-instrument` backup exists, restore from it.
  - For triton_prefill_attention.py and triton_decode_attention.py
    (no pre-clamp backup), strip the well-marked clamp blocks/wrappers
    in place.

Idempotent: safe to re-run.

Usage:
    python3 scripts/instruments/revert-all-debug-patches.py

After running, the venv is back to the riscv-cross-compile baseline.
The pre-existing `.pre-riscv-patch` files are NOT touched — those are
the upstream-original files and reverting to them would re-break the
riscv VM.
"""
import re
import shutil
import sys
from pathlib import Path

VENV = Path("/home/ubuntu/ai-2.10/lib/python3.13/site-packages/vllm")

OPS_DIR = VENV / "v1/attention/ops"
QWEN3 = VENV / "model_executor/models/qwen3.py"

# (target, backup_suffix) — restore by `cp backup target` if backup exists
RESTORE_FROM_BACKUP = [
    (OPS_DIR / "prefix_prefill.py",                 ".before-clamp"),
    (OPS_DIR / "chunked_prefill_paged_decode.py",   ".bak-before-fix-20260424"),
    (QWEN3,                                          ".before-instrument"),
]

# Files where we have to strip in place because no useful backup exists
STRIP_IN_PLACE = [
    OPS_DIR / "triton_prefill_attention.py",
    OPS_DIR / "triton_decode_attention.py",
]


def restore_from_backup() -> None:
    for target, suffix in RESTORE_FROM_BACKUP:
        backup = target.with_suffix(target.suffix + suffix)
        if not backup.exists():
            print(f"  SKIP {target.name}: backup {backup.name} missing")
            continue
        if not target.exists():
            print(f"  SKIP {target.name}: target missing")
            continue
        shutil.copy2(backup, target)
        print(f"  OK   {target.name} <- {backup.name}")


def strip_clamp_block(src: str) -> str:
    """Remove our clamp blocks (both v1 minimum/maximum form and v2
    zero-degenerate-row + NaN-safe tl.where form).

    v1 (3 lines):
        # ---- llm-bench fp16 fix: clamp acc to fp16 range before store ----
        acc = tl.minimum(tl.maximum(acc, -65504.0), 65504.0)
        # ---- end ----

    v2 (5 lines, with l_i or L variant):
        # ---- llm-bench fp16 fix v2: zero degenerate rows + NaN-safe clamp ----
        acc = tl.where(l_i[:, None] < 1e-3, 0.0, acc)   # or L[:, None]
        acc = tl.where(acc != acc, 0.0, acc)
        acc = tl.where(acc > 65504.0, 65504.0, acc)
        acc = tl.where(acc < -65504.0, -65504.0, acc)
        # ---- end ----

    Tolerant to leading whitespace and the v2 comment-only line variant
    that includes "When the softmax denominator is tiny..." comment.
    """
    # v2 form (with optional comment line + variable-named threshold guard)
    v2_pattern = (
        r"[ \t]*# ---- llm-bench fp16 fix v2:[^\n]*\n"
        r"(?:[ \t]*# [^\n]*\n){0,3}"
        r"[ \t]*acc = tl\.where\(\w+\[:, None\] < 1e-3, 0\.0, acc\)\n"
        r"[ \t]*acc = tl\.where\(acc != acc, 0\.0, acc\)[^\n]*\n"
        r"[ \t]*acc = tl\.where\(acc > 65504\.0, 65504\.0, acc\)\n"
        r"[ \t]*acc = tl\.where\(acc < -65504\.0, -65504\.0, acc\)\n"
        r"[ \t]*# ---- end ----\n"
    )
    src = re.sub(v2_pattern, "", src)
    # v1 form
    v1_pattern = (
        r"[ \t]*# ---- llm-bench fp16 fix:[^\n]*\n"
        r"[ \t]*acc = tl\.minimum\(tl\.maximum\(acc, -65504\.0\), 65504\.0\)\n"
        r"[ \t]*# ---- end ----\n"
    )
    return re.sub(v1_pattern, "", src)


def strip_inline_clamp(src: str) -> str:
    """Undo the inline clamp wrappers we added in triton_decode_attention.py:

        tl.minimum(tl.maximum(acc / e_sum, -65504.0), 65504.0),  # llm-bench fp16 fix
        ->  acc / e_sum,
        tl.minimum(tl.maximum(acc / e_sum[:, None], -65504.0), 65504.0),  # llm-bench fp16 fix
        ->  acc / e_sum[:, None],
    """
    src = src.replace(
        "tl.minimum(tl.maximum(acc / e_sum, -65504.0), 65504.0),  # llm-bench fp16 fix",
        "acc / e_sum,",
    )
    src = src.replace(
        "tl.minimum(tl.maximum(acc / e_sum[:, None], -65504.0), 65504.0),  # llm-bench fp16 fix",
        "acc / e_sum[:, None],",
    )
    return src


def strip_in_place() -> None:
    for path in STRIP_IN_PLACE:
        if not path.exists():
            print(f"  SKIP {path.name}: missing")
            continue
        before = path.read_text()
        after = strip_clamp_block(before)
        after = strip_inline_clamp(after)
        if after == before:
            print(f"  OK   {path.name}: already clean")
            continue
        path.write_text(after)
        n_block = before.count("llm-bench fp16 fix:")
        n_inline = before.count("# llm-bench fp16 fix")
        print(f"  OK   {path.name}: stripped {n_block} block(s) + "
              f"{n_inline - n_block} inline clamp(s)")


def main() -> int:
    if not VENV.exists():
        print(f"ERROR: venv not found at {VENV}", file=sys.stderr)
        return 1
    print("[revert] restoring from backups…")
    restore_from_backup()
    print("[revert] stripping clamp blocks in place where no backup exists…")
    strip_in_place()
    print("[revert] done. Verify:")
    print("  grep -c 'llm-bench fp16 fix\\|force triton path probe\\|N8STATS\\|N8PROBE' "
          "/home/ubuntu/ai-2.10/lib/python3.13/site-packages/vllm/v1/attention/ops/*.py "
          "/home/ubuntu/ai-2.10/lib/python3.13/site-packages/vllm/model_executor/models/qwen3.py")
    print("  → all lines should report 0.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
