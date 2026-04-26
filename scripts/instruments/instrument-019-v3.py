"""Add a SECOND instrument to 0.19: print what _bookkeeping_sync is
writing back to token_ids_cpu, per req.

Inserts right after the line:
    self.input_batch.token_ids_cpu[req_idx, start_idx:end_idx] = sampled_ids
"""
import sys
from pathlib import Path

PATH = Path("/home/ubuntu/ai-2.10/lib/python3.13/site-packages/vllm/v1/worker/gpu_model_runner.py")
src = PATH.read_text()

NEEDLE = """            self.input_batch.token_ids_cpu[req_idx, start_idx:end_idx] = sampled_ids
            self.input_batch.is_token_ids[req_idx, start_idx:end_idx] = True
            self.input_batch.num_tokens_no_spec[req_idx] = end_idx"""

if NEEDLE not in src:
    print("ERROR: needle not found", file=sys.stderr); sys.exit(1)

INSTRUMENT = """
            # ---- llm-bench WRITEBACK instrument BEGIN ----
            try:
                if num_sampled_tokens >= 2:
                    cnt = getattr(self, "_bench_wb_iter", 0)
                    if req_idx == 0:
                        cnt = cnt + 1
                        self._bench_wb_iter = cnt
                    if cnt <= 6:
                        nct = self.input_batch.num_computed_tokens_cpu[req_idx]
                        print(f"[INSTR-019-WB #{cnt}] req_idx={req_idx} start={start_idx} end={end_idx} sampled={sampled_ids} nct={nct} (nct-vs-start diff={int(start_idx)-int(nct)})", flush=True)
            except Exception as _e:
                print(f"[INSTR-019-WB] err: {_e}", flush=True)
            # ---- llm-bench WRITEBACK instrument END ----
"""

new = src.replace(NEEDLE, NEEDLE + INSTRUMENT, 1)
if new == src:
    print("ERROR: replacement no-op", file=sys.stderr); sys.exit(1)
PATH.write_text(new)
print(f"OK: writeback-instrumented {PATH}")
