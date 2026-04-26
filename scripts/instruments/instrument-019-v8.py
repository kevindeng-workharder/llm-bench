import sys
from pathlib import Path
PATH = Path("/home/ubuntu/ai-2.10/lib/python3.13/site-packages/vllm/v1/worker/gpu_model_runner.py")
src = PATH.read_text()

# Restore first by stripping any prior instruments
import re
src = re.sub(r"\n        # ---- llm-bench[^\n]*\n(?:.*\n)*?        # ---- end ----\n", "\n", src)
# Also remove any leftover print stubs
src = re.sub(r"\n        # ---- llm-bench[^\n]*\n        import os.*\n        os\.write\(2.*\n        # ---- end ----\n", "\n", src)

# Hook 1: at start of _prepare_input_ids
ENTRY_NEEDLE = "        if self.input_batch.prev_sampled_token_ids is None:\n            # Normal scheduling case"
ENTRY_HOOK = """        # ---- v8 entry ----
        import os as _os
        _os.write(2, ("[V8-PI-ENTRY] num_reqs=" + str(num_reqs) + " total=" + str(total_num_scheduled_tokens) + " has_prev=" + str(self.input_batch.prev_sampled_token_ids is not None) + "\\n").encode())
        # ---- end ----
        if self.input_batch.prev_sampled_token_ids is None:
            # Normal scheduling case"""

# Hook 2: in fast-opt branch
FASTOPT_NEEDLE = """        if common_indices_match and max_flattened_index == (num_common_tokens - 1):
            # Common-case optimization: the batch is unchanged
            # and no reordering happened.
            # The indices are both the same permutation of 0..N-1 so
            # we can copy directly using a single slice.
            self.input_ids.gpu[:num_common_tokens].copy_("""

FASTOPT_HOOK = """        if common_indices_match and max_flattened_index == (num_common_tokens - 1):
            # Common-case optimization: the batch is unchanged
            # and no reordering happened.
            # The indices are both the same permutation of 0..N-1 so
            # we can copy directly using a single slice.
            # ---- v8 fast-opt ----
            try:
                _src8 = self.input_batch.prev_sampled_token_ids[:num_common_tokens, 0].cpu().tolist()
                _os.write(2, ("[V8-FAST] num_common=" + str(num_common_tokens) + " src=" + str(_src8) + "\\n").encode())
            except Exception as _ex:
                _os.write(2, ("[V8-FAST] err: " + str(_ex) + "\\n").encode())
            # ---- end ----
            self.input_ids.gpu[:num_common_tokens].copy_("""

# Hook 3: in scatter branch
SCATTER_NEEDLE = """        self.input_ids.gpu.scatter_(
            dim=0,
            index=sampled_tokens_index_tensor,
            src=self.input_batch.prev_sampled_token_ids[
                prev_common_req_indices_tensor, 0
            ],
        )"""

SCATTER_HOOK = """        # ---- v8 scatter pre ----
        try:
            _idx8 = sampled_tokens_index_tensor.cpu().tolist()
            _previdx8 = prev_common_req_indices_tensor.cpu().tolist()
            _src8 = self.input_batch.prev_sampled_token_ids[prev_common_req_indices_tensor, 0].cpu().tolist()
            _before8 = self.input_ids.gpu[:max(_idx8)+1].cpu().tolist() if _idx8 else []
            _os.write(2, ("[V8-SCAT-PRE] num_reqs=" + str(num_reqs) + " sample_flat=" + str(_idx8) + " prev_idx=" + str(_previdx8) + " src=" + str(_src8) + " before=" + str(_before8) + "\\n").encode())
        except Exception as _ex:
            _os.write(2, ("[V8-SCAT-PRE] err: " + str(_ex) + "\\n").encode())
        # ---- end ----
        self.input_ids.gpu.scatter_(
            dim=0,
            index=sampled_tokens_index_tensor,
            src=self.input_batch.prev_sampled_token_ids[
                prev_common_req_indices_tensor, 0
            ],
        )
        # ---- v8 scatter post ----
        try:
            _after8 = self.input_ids.gpu[:max(_idx8)+1].cpu().tolist() if _idx8 else []
            _os.write(2, ("[V8-SCAT-POST] after=" + str(_after8) + "\\n").encode())
        except Exception as _ex:
            _os.write(2, ("[V8-SCAT-POST] err: " + str(_ex) + "\\n").encode())
        # ---- end ----"""

if ENTRY_NEEDLE not in src:
    print("ENTRY needle missing", file=sys.stderr); sys.exit(1)
if FASTOPT_NEEDLE not in src:
    print("FASTOPT needle missing", file=sys.stderr); sys.exit(1)
if SCATTER_NEEDLE not in src:
    print("SCATTER needle missing", file=sys.stderr); sys.exit(1)

src = src.replace(ENTRY_NEEDLE, ENTRY_HOOK, 1)
src = src.replace(FASTOPT_NEEDLE, FASTOPT_HOOK, 1)
src = src.replace(SCATTER_NEEDLE, SCATTER_HOOK, 1)
PATH.write_text(src)
print("ok v8")
