"""Instrument sampler output: print sampled_token_ids right before it's
stored as prev_sampled_token_ids."""
import sys, re
from pathlib import Path
PATH = Path("/home/ubuntu/ai-2.10/lib/python3.13/site-packages/vllm/v1/worker/gpu_model_runner.py")
src = PATH.read_text()
# Strip prior instruments first
src = re.sub(r"\n        # ---- v8[^\n]*\n(?:.*\n)*?        # ---- end ----\n", "\n", src)
src = re.sub(r"\n        # ---- llm-bench[^\n]*\n(?:.*\n)*?        # ---- end ----\n", "\n", src)

NEEDLE = "                self.input_batch.prev_sampled_token_ids = sampled_token_ids"
HOOK = """                # ---- v9 sampler output ----
                import os as _os
                try:
                    _shp = list(sampled_token_ids.shape)
                    _vals = sampled_token_ids.cpu().tolist()
                    _os.write(2, ("[V9-SAMPLER-OUT] shape=" + str(_shp) + " vals=" + str(_vals)[:200] + "\\n").encode())
                except Exception as _ex:
                    _os.write(2, ("[V9-SAMPLER-OUT] err: " + str(_ex) + "\\n").encode())
                # ---- end ----
                self.input_batch.prev_sampled_token_ids = sampled_token_ids"""

if NEEDLE not in src:
    print("ERROR not found", file=sys.stderr); sys.exit(1)
new = src.replace(NEEDLE, HOOK, 1)
PATH.write_text(new)
print("ok v9")
