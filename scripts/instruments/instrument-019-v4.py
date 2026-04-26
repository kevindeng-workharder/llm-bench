"""Add instrument to _prepare_input_ids common-case optimization branch.
Print prev_sampled_token_ids[:N, 0] right before the copy."""
import sys
from pathlib import Path

PATH = Path("/home/ubuntu/ai-2.10/lib/python3.13/site-packages/vllm/v1/worker/gpu_model_runner.py")
src = PATH.read_text()

NEEDLE = """        if common_indices_match and max_flattened_index == (num_common_tokens - 1):
            # Common-case optimization: the batch is unchanged
            # and no reordering happened.
            # The indices are both the same permutation of 0..N-1 so
            # we can copy directly using a single slice.
            self.input_ids.gpu[:num_common_tokens].copy_(
                self.input_batch.prev_sampled_token_ids[:num_common_tokens, 0],
                non_blocking=True,
            )"""

if NEEDLE not in src:
    print("ERROR: needle not found", file=sys.stderr); sys.exit(1)

REPLACEMENT = """        if common_indices_match and max_flattened_index == (num_common_tokens - 1):
            # Common-case optimization: the batch is unchanged
            # and no reordering happened.
            # The indices are both the same permutation of 0..N-1 so
            # we can copy directly using a single slice.
            # ---- llm-bench instrument: print pre-copy state ----
            try:
                cnt = getattr(self, "_bench_pinp", 0) + 1
                self._bench_pinp = cnt
                if cnt <= 8 and num_common_tokens >= 2:
                    _src = self.input_batch.prev_sampled_token_ids[:num_common_tokens, 0].cpu().tolist()
                    _dst_before = self.input_ids.gpu[:num_common_tokens].cpu().tolist()
                    print(f"[INSTR-019-PINP-OPT #{cnt}] num_common={num_common_tokens} prev_sampled_src={_src} input_ids_before={_dst_before}", flush=True)
            except Exception as _e:
                print(f"[INSTR-019-PINP-OPT] err: {_e}", flush=True)
            # ---- end instrument ----
            self.input_ids.gpu[:num_common_tokens].copy_(
                self.input_batch.prev_sampled_token_ids[:num_common_tokens, 0],
                non_blocking=True,
            )
            # ---- llm-bench instrument: print post-copy state ----
            try:
                if cnt <= 8 and num_common_tokens >= 2:
                    _dst_after = self.input_ids.gpu[:num_common_tokens].cpu().tolist()
                    print(f"[INSTR-019-PINP-OPT #{cnt}] AFTER copy input_ids={_dst_after}", flush=True)
            except Exception as _e:
                pass
            # ---- end instrument ----"""

new = src.replace(NEEDLE, REPLACEMENT, 1)
if new == src:
    print("ERROR: replacement no-op", file=sys.stderr); sys.exit(1)
PATH.write_text(new)
print(f"OK: PINP-OPT-instrumented {PATH}")
