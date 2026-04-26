"""v10: log logits stats before sampler, sampled tokens after."""
import sys, re
from pathlib import Path
PATH = Path("/home/ubuntu/ai-2.10/lib/python3.13/site-packages/vllm/v1/worker/gpu_model_runner.py")
src = PATH.read_text()
src = re.sub(r"\n        # ---- v[0-9]+[^\n]*\n(?:.*\n)*?        # ---- end ----\n", "\n", src)
src = re.sub(r"\n                # ---- v[0-9]+[^\n]*\n(?:.*\n)*?                # ---- end ----\n", "\n", src)

# Hook at start of _sample
NEEDLE = """        if spec_decode_metadata is None:
            return self.sampler(
                logits=logits,
                sampling_metadata=sampling_metadata,
            )"""

HOOK = """        if spec_decode_metadata is None:
            # ---- v10 logits + sampler output ----
            import os as _os
            try:
                _shp = list(logits.shape) if logits is not None else None
                if logits is not None and logits.size(0) >= 2 and logits.size(0) <= 16:
                    _argmax = logits.argmax(dim=-1).cpu().tolist()
                    _max = logits.max(dim=-1).values.cpu().tolist()
                    _sums = logits.float().sum(dim=-1).cpu().tolist()
                    _os.write(2, ("[V10-PRE-SAMP] shape=" + str(_shp) + " argmax_per_row=" + str(_argmax) + " max_per_row=" + str([round(x,3) for x in _max]) + " sum_per_row=" + str([round(x,3) for x in _sums]) + "\\n").encode())
            except Exception as _ex:
                _os.write(2, ("[V10-PRE-SAMP] err: " + str(_ex) + "\\n").encode())
            _so = self.sampler(
                logits=logits,
                sampling_metadata=sampling_metadata,
            )
            try:
                if _so.sampled_token_ids.size(0) >= 2:
                    _os.write(2, ("[V10-POST-SAMP] sampled=" + str(_so.sampled_token_ids.cpu().tolist()) + "\\n").encode())
            except Exception:
                pass
            return _so
            # ---- end ----
            return self.sampler(
                logits=logits,
                sampling_metadata=sampling_metadata,
            )"""

if NEEDLE not in src:
    print("ERROR not found", file=sys.stderr); sys.exit(1)
new = src.replace(NEEDLE, HOOK, 1)
PATH.write_text(new)
print("ok v10")
