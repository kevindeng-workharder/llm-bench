"""Probe which path chunked_prefill_paged_decode takes (custom C++ vs triton)."""
import sys, re
from pathlib import Path
PATH = Path("/home/ubuntu/ai-2.10/lib/python3.13/site-packages/vllm/v1/attention/ops/chunked_prefill_paged_decode.py")
src = PATH.read_text()
# Strip prior probes (zeros + sync left from earlier patches)
NEEDLE = "    if use_custom:\n        _PARTITION_SIZE_ROCM = 256"
HOOK = """    import os as _os
    _os.write(2, ("[CPPD] use_custom=" + str(use_custom) + " query.shape=" + str(list(query.shape)) + " query.dtype=" + str(query.dtype) + " block_size=" + str(block_size) + " max_seq_len=" + str(max_seq_len) + "\\n").encode())
    if use_custom:
        _PARTITION_SIZE_ROCM = 256"""
if NEEDLE not in src:
    print("ERROR not found", file=sys.stderr); sys.exit(1)
PATH.write_text(src.replace(NEEDLE, HOOK, 1))
print("ok")
