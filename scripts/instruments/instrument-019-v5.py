"""Instrument _prepare_input_ids entry to log which branch is taken each
iteration: prev_None / fast-opt / fall-through scatter / no-op.
"""
import sys
from pathlib import Path

PATH = Path("/home/ubuntu/ai-2.10/lib/python3.13/site-packages/vllm/v1/worker/gpu_model_runner.py")
src = PATH.read_text()

# Instrument 1: at entry — log num_reqs, total_num_scheduled_tokens, prev_sampled present
NEEDLE_ENTRY = """        if self.input_batch.prev_sampled_token_ids is None:
            # Normal scheduling case
            self.input_ids.copy_to_gpu(total_num_scheduled_tokens)"""

ENTRY_HOOK = """        # ---- llm-bench instrument: branch logging ----
        try:
            cnt = getattr(self, "_bench_pi", 0) + 1
            self._bench_pi = cnt
            _has_prev = self.input_batch.prev_sampled_token_ids is not None
            if cnt <= 12 and num_reqs >= 2:
                print(f"[INSTR-019-PI #{cnt}] num_reqs={num_reqs} total_sched={total_num_scheduled_tokens} has_prev={_has_prev}", flush=True)
        except Exception as _e:
            print(f"[INSTR-019-PI] err: {_e}", flush=True)
        # ---- end ----
        if self.input_batch.prev_sampled_token_ids is None:
            # Normal scheduling case
            self.input_ids.copy_to_gpu(total_num_scheduled_tokens)"""

if NEEDLE_ENTRY not in src:
    print("ERROR: ENTRY needle not found", file=sys.stderr); sys.exit(1)

src1 = src.replace(NEEDLE_ENTRY, ENTRY_HOOK, 1)

# Instrument 2: log scatter inputs (the fall-through path)
NEEDLE_SCATTER = """        self.input_ids.gpu.scatter_(
            dim=0,
            index=sampled_tokens_index_tensor,
            src=self.input_batch.prev_sampled_token_ids[
                prev_common_req_indices_tensor, 0
            ],
        )"""

SCATTER_HOOK = """        # ---- llm-bench instrument: scatter inputs ----
        try:
            if cnt <= 12 and num_reqs >= 2:
                _idx = sampled_tokens_index_tensor.cpu().tolist()
                _prev_idx = prev_common_req_indices_tensor.cpu().tolist()
                _src = self.input_batch.prev_sampled_token_ids[prev_common_req_indices_tensor, 0].cpu().tolist()
                _before = self.input_ids.gpu[:max(_idx)+1].cpu().tolist() if _idx else []
                print(f"[INSTR-019-PI-SCAT #{cnt}] sample_flat={_idx} prev_idx={_prev_idx} src={_src} input_ids_before={_before}", flush=True)
        except Exception as _e:
            print(f"[INSTR-019-PI-SCAT] err: {_e}", flush=True)
        # ---- end ----
        self.input_ids.gpu.scatter_(
            dim=0,
            index=sampled_tokens_index_tensor,
            src=self.input_batch.prev_sampled_token_ids[
                prev_common_req_indices_tensor, 0
            ],
        )
        # ---- llm-bench instrument: post-scatter ----
        try:
            if cnt <= 12 and num_reqs >= 2:
                _after = self.input_ids.gpu[:max(_idx)+1].cpu().tolist() if _idx else []
                print(f"[INSTR-019-PI-SCAT #{cnt}] AFTER input_ids={_after}", flush=True)
        except Exception:
            pass
        # ---- end ----"""

if NEEDLE_SCATTER not in src1:
    print("ERROR: SCATTER needle not found", file=sys.stderr); sys.exit(1)

src2 = src1.replace(NEEDLE_SCATTER, SCATTER_HOOK, 1)
PATH.write_text(src2)
print(f"OK: branch+scatter-instrumented {PATH}")
