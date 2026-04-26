"""Re-instrument 0.11 right before model() call.
Same INSTR-011 line for diff with 0.19.
"""
import sys
from pathlib import Path

PATH = Path("/home/ubuntu/ai/lib/python3.13/site-packages/vllm/v1/worker/gpu_model_runner.py")
src = PATH.read_text()

NEEDLE = "            model_output = self.model("
if NEEDLE not in src:
    print("ERROR: needle not found", file=sys.stderr); sys.exit(1)

INSTRUMENT = """            # ---- llm-bench instrument BEGIN ----
            try:
                if num_reqs >= 2:
                    cnt = getattr(self, "_bench_iter", 0) + 1
                    self._bench_iter = cnt
                    if cnt <= 30:
                        _ids = input_ids[:num_input_tokens].cpu().tolist()
                        _pos = positions[:num_input_tokens].cpu().tolist() if positions is not None else []
                        _ns = num_scheduled_tokens.tolist() if hasattr(num_scheduled_tokens, "tolist") else list(num_scheduled_tokens)
                        _nct = list(self.input_batch.num_computed_tokens_cpu[:num_reqs])
                        lines = []
                        start = 0
                        for r in range(num_reqs):
                            n = int(_ns[r])
                            head = _ids[start:start+min(8, n)]
                            pos = _pos[start:start+min(4, n)] if _pos else []
                            lines.append(f"row{r}: nct={_nct[r]} num_sched={n} pos={pos} ids={head}")
                            start += n
                        print(f"[INSTR-011 #{cnt}] num_reqs={num_reqs} total={num_input_tokens}", flush=True)
                        for l in lines:
                            print(f"  {l}", flush=True)
            except Exception as _e:
                print(f"[INSTR-011] err: {_e}", flush=True)
            # ---- llm-bench instrument END ----
"""

new = src.replace(NEEDLE, INSTRUMENT + NEEDLE, 1)
PATH.write_text(new)
print(f"OK: instrumented {PATH}")
