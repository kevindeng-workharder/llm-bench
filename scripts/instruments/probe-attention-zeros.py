"""Probe: replace torch.empty(output_shape) with torch.zeros to test
whether the bug is "kernel skips rows" (→ zeros stay zero) or
"kernel writes NaN" (→ NaN replaces zeros).

Patches Attention.forward to use zeros + log per-row max of output BEFORE
and AFTER the unified_attention_with_output call.
"""
import sys, re
from pathlib import Path

PATH = Path("/home/ubuntu/ai-2.10/lib/python3.13/site-packages/vllm/model_executor/layers/attention/attention.py")
src = PATH.read_text()

NEEDLE = """            output = torch.empty(output_shape, dtype=output_dtype, device=query.device)
            hidden_size = output_shape[-1]"""

REPL = """            # ---- bench probe: zeros instead of empty ----
            output = torch.zeros(output_shape, dtype=output_dtype, device=query.device)
            hidden_size = output_shape[-1]
            import os as _os
            _enable = (output.dim() >= 1 and 2 <= output.size(0) <= 16
                       and getattr(self, "_a_dbg_cnt", 0) < 4)
            if _enable:
                self._a_dbg_cnt = getattr(self, "_a_dbg_cnt", 0) + 1
                try:
                    _b = output.view(output.size(0), -1).max(dim=-1).values.float().cpu().tolist()
                    _os.write(2, ("[A_BEFORE " + self.layer_name + "] before_max=" + str(["NaN" if x!=x else round(x,3) for x in _b]) + "\\n").encode())
                except Exception as _ex:
                    _os.write(2, ("[A_BEFORE] err: " + str(_ex) + "\\n").encode())
            # ---- end ----"""

if NEEDLE not in src:
    print("ERROR needle not found", file=sys.stderr); sys.exit(1)
src = src.replace(NEEDLE, REPL, 1)

# After unified_attention_with_output, log post
NEEDLE2 = """                unified_attention_with_output(
                    query,
                    key,
                    value,
                    output,
                    self.layer_name,
                    kv_cache_dummy_dep=kv_cache_dummy_dep,
                )
            return output.view(-1, hidden_size)"""

REPL2 = """                unified_attention_with_output(
                    query,
                    key,
                    value,
                    output,
                    self.layer_name,
                    kv_cache_dummy_dep=kv_cache_dummy_dep,
                )
            # ---- bench probe: post log ----
            if _enable:
                try:
                    _a = output.view(output.size(0), -1).max(dim=-1).values.float().cpu().tolist()
                    _ami = output.view(output.size(0), -1).min(dim=-1).values.float().cpu().tolist()
                    _has_nan = any(x!=x for x in _a) or any(x!=x for x in _ami)
                    _os.write(2, ("[A_AFTER " + self.layer_name + "] after_max=" + str(["NaN" if x!=x else round(x,3) for x in _a]) + " after_min=" + str(["NaN" if x!=x else round(x,3) for x in _ami]) + (" HAS_NAN" if _has_nan else "") + "\\n").encode())
                except Exception as _ex:
                    _os.write(2, ("[A_AFTER] err: " + str(_ex) + "\\n").encode())
            # ---- end ----
            return output.view(-1, hidden_size)"""

# This needle appears twice (use_direct_call branch + else branch); both should match.
# Use a more unique anchor — find the second occurrence (else branch) which is what runs for our setup
# Actually let me check both work — replace_all carefully.
count = src.count(NEEDLE2)
if count != 2:
    # Try without the comment block that may differ
    NEEDLE2_alt = """            return output.view(-1, hidden_size)"""
    cnt2 = src.count(NEEDLE2_alt)
    print(f"NEEDLE2 count={count}, NEEDLE2_alt count={cnt2}", file=sys.stderr)

new = src.replace(NEEDLE2, REPL2)  # replace ALL occurrences
PATH.write_text(new)
print(f"ok zeros probe (replaced {count} occurrence(s))")
