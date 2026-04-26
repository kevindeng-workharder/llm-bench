import sys
from pathlib import Path
PATH = Path("/home/ubuntu/ai-2.10/lib/python3.13/site-packages/vllm/v1/worker/gpu_model_runner.py")
src = PATH.read_text()

# Insert at top of _prepare_inputs body, just after num_reqs assignment
NEEDLE = """        num_reqs = self.input_batch.num_reqs
        assert num_reqs > 0"""

HOOK = """        num_reqs = self.input_batch.num_reqs
        assert num_reqs > 0
        # ---- llm-bench v7 hook on _prepare_inputs entry ----
        import os as _os
        _os.write(2, ("[INSTR-019-PINPUTS-V7] enter num_reqs=" + str(num_reqs) + " total_sched=" + str(total_num_scheduled_tokens) + "\\n").encode())
        # ---- end ----"""

if NEEDLE not in src:
    print("ERROR not found", file=sys.stderr); sys.exit(1)
new = src.replace(NEEDLE, HOOK, 1)
PATH.write_text(new)
print("ok v7")
