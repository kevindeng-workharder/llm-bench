"""Upgrade the 3 inline v1-style clamps in triton_decode_attention.py
(MLA decode kernels) to v2 style: zero-degenerate-rows + NaN-safe
tl.where clamp. Same logic as the prefix_prefill / chunked_prefill
v2 patch.

Sites:
  - _fwd_kernel_stage1          (e_sum is 1D)
  - _fwd_grouped_kernel_stage1  (e_sum broadcast as e_sum[:, None])
  - _fwd_kernel_stage2          (e_sum is 1D, stores final O)

Each `tl.store(... acc/e_sum ..., mask=...)` is rewritten to compute
the clamped output as a separate variable before the store.

Idempotent: bails out if the v1 inline marker is already gone.
"""
import sys, re
from pathlib import Path

PATH = Path("/home/ubuntu/ai-2.10/lib/python3.13/site-packages/vllm/v1/attention/ops/triton_decode_attention.py")
src = PATH.read_text()

# Site 1: _fwd_kernel_stage1 — 1D e_sum, mask=(mask_dv)
SITE1_OLD = """        tl.store(
            Att_Out + offs_mid_o,
            tl.minimum(tl.maximum(acc / e_sum, -65504.0), 65504.0),  # llm-bench fp16 fix
            mask=(mask_dv),
        )"""

SITE1_NEW = """        # ---- llm-bench fp16 fix v2: zero degenerate rows + NaN-safe clamp ----
        _v2_out = acc / e_sum
        _v2_out = tl.where(e_sum < 1e-3, 0.0, _v2_out)
        _v2_out = tl.where(_v2_out != _v2_out, 0.0, _v2_out)
        _v2_out = tl.where(_v2_out > 65504.0, 65504.0, _v2_out)
        _v2_out = tl.where(_v2_out < -65504.0, -65504.0, _v2_out)
        tl.store(
            Att_Out + offs_mid_o,
            _v2_out,
            mask=(mask_dv),
        )
        # ---- end ----"""

# Site 2: _fwd_grouped_kernel_stage1 — 2D e_sum[:, None]
SITE2_OLD = """        tl.store(
            Att_Out + offs_mid_o,
            tl.minimum(tl.maximum(acc / e_sum[:, None], -65504.0), 65504.0),  # llm-bench fp16 fix
            mask=(mask_h[:, None]) & (mask_dv[None, :]),
        )"""

SITE2_NEW = """        # ---- llm-bench fp16 fix v2: zero degenerate rows + NaN-safe clamp ----
        _v2_out = acc / e_sum[:, None]
        _v2_out = tl.where(e_sum[:, None] < 1e-3, 0.0, _v2_out)
        _v2_out = tl.where(_v2_out != _v2_out, 0.0, _v2_out)
        _v2_out = tl.where(_v2_out > 65504.0, 65504.0, _v2_out)
        _v2_out = tl.where(_v2_out < -65504.0, -65504.0, _v2_out)
        tl.store(
            Att_Out + offs_mid_o,
            _v2_out,
            mask=(mask_h[:, None]) & (mask_dv[None, :]),
        )
        # ---- end ----"""

# Site 3: _fwd_kernel_stage2 — 1D e_sum, stores final O with mask=mask_d
SITE3_OLD = """    tl.store(
        o + cur_batch * stride_obs + cur_head * stride_oh + offs_d,
        tl.minimum(tl.maximum(acc / e_sum, -65504.0), 65504.0),  # llm-bench fp16 fix
        mask=mask_d,
    )"""

SITE3_NEW = """    # ---- llm-bench fp16 fix v2: zero degenerate rows + NaN-safe clamp ----
    _v2_out = acc / e_sum
    _v2_out = tl.where(e_sum < 1e-3, 0.0, _v2_out)
    _v2_out = tl.where(_v2_out != _v2_out, 0.0, _v2_out)
    _v2_out = tl.where(_v2_out > 65504.0, 65504.0, _v2_out)
    _v2_out = tl.where(_v2_out < -65504.0, -65504.0, _v2_out)
    tl.store(
        o + cur_batch * stride_obs + cur_head * stride_oh + offs_d,
        _v2_out,
        mask=mask_d,
    )
    # ---- end ----"""

n = 0
for old, new, name in [
    (SITE1_OLD, SITE1_NEW, "_fwd_kernel_stage1"),
    (SITE2_OLD, SITE2_NEW, "_fwd_grouped_kernel_stage1"),
    (SITE3_OLD, SITE3_NEW, "_fwd_kernel_stage2"),
]:
    if old not in src:
        print(f"SKIP {name}: v1 inline clamp not found (already v2 maybe?)")
        continue
    src = src.replace(old, new, 1)
    n += 1
    print(f"OK   {name}: replaced v1 inline -> v2 multi-line clamp")

if n == 0:
    print("nothing changed")
else:
    PATH.write_text(src)
    # quick verify: no v1-style inline left
    remaining = src.count("tl.minimum(tl.maximum(acc / e_sum")
    if remaining:
        print(f"WARN  {remaining} v1-inline clamp(s) still present — manual review needed", file=sys.stderr)
    else:
        print(f"done — all {n} sites upgraded to v2; no v1 inline clamps remain")
