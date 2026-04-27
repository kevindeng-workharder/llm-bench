"""Force use_custom = False to bypass C++ kernel and test if Triton path is correct."""
import sys, re
from pathlib import Path
PATH = Path("/home/ubuntu/ai-2.10/lib/python3.13/site-packages/vllm/v1/attention/ops/chunked_prefill_paged_decode.py")
src = PATH.read_text()
NEEDLE = '    use_custom = use_rocm_custom_paged_attention('
HOOK = '    use_custom = False  # ---- force triton path probe ----\n    _ignored = use_rocm_custom_paged_attention('
if NEEDLE not in src:
    print("ERROR not found", file=sys.stderr); sys.exit(1)
PATH.write_text(src.replace(NEEDLE, HOOK, 1))
print("ok force-triton")
