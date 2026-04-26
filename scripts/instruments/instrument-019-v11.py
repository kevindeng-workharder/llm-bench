"""v11: log hidden_states / sample_hidden_states / logits to localize the NaN."""
import sys, re
from pathlib import Path
PATH = Path("/home/ubuntu/ai-2.10/lib/python3.13/site-packages/vllm/v1/worker/gpu_model_runner.py")
src = PATH.read_text()
src = re.sub(r"\n        # ---- v[0-9]+[^\n]*\n(?:.*\n)*?        # ---- end ----\n", "\n", src)
src = re.sub(r"\n                # ---- v[0-9]+[^\n]*\n(?:.*\n)*?                # ---- end ----\n", "\n", src)
src = re.sub(r"\n            # ---- v[0-9]+[^\n]*\n(?:.*\n)*?            # ---- end ----\n", "\n", src)

NEEDLE = """                sample_hidden_states = hidden_states[logits_indices]
                logits = self.model.compute_logits(sample_hidden_states)"""

HOOK = """                sample_hidden_states = hidden_states[logits_indices]
                # ---- v11 ----
                import os as _os
                try:
                    _hs_shape = list(sample_hidden_states.shape)
                    if sample_hidden_states.size(0) >= 2 and sample_hidden_states.size(0) <= 16:
                        _hs_max = sample_hidden_states.max(dim=-1).values.cpu().tolist()
                        _hs_sum = sample_hidden_states.float().sum(dim=-1).cpu().tolist()
                        _hs_first = sample_hidden_states[:, :3].cpu().tolist()
                        _os.write(2, ("[V11-HS] shape=" + str(_hs_shape) + " max_per_row=" + str([round(x,3) if x==x else 'NaN' for x in _hs_max]) + " sum_per_row=" + str([round(x,3) if x==x else 'NaN' for x in _hs_sum]) + " first3=" + str(_hs_first) + "\\n").encode())
                except Exception as _ex:
                    _os.write(2, ("[V11-HS] err: " + str(_ex) + "\\n").encode())
                # ---- end ----
                logits = self.model.compute_logits(sample_hidden_states)"""

if NEEDLE not in src:
    print("ERROR not found", file=sys.stderr); sys.exit(1)
new = src.replace(NEEDLE, HOOK, 1)
PATH.write_text(new)
print("ok v11")
