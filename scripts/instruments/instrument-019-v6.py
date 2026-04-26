import sys
from pathlib import Path
PATH = Path("/home/ubuntu/ai-2.10/lib/python3.13/site-packages/vllm/v1/worker/gpu_model_runner.py")
src = PATH.read_text()
NEEDLE_ENTRY = "        if self.input_batch.prev_sampled_token_ids is None:"
HOOK = """        # ---- llm-bench v6: ALWAYS print on entry ----
        import os
        os.write(2, ("[INSTR-019-PI-V6] enter num_reqs=" + str(num_reqs) + " has_prev=" + str(self.input_batch.prev_sampled_token_ids is not None) + "\\n").encode())
        # ---- end ----
        if self.input_batch.prev_sampled_token_ids is None:"""
new = src.replace(NEEDLE_ENTRY, HOOK, 1)
if new == src:
    print("ERROR not replaced", file=sys.stderr); sys.exit(1)
PATH.write_text(new)
print("ok v6")
