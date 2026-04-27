"""Print metadata tensors at entry of chunked_prefill_paged_decode:
  - slot_mapping (where to write new K, V for each token)
  - block_table[seq] (which physical KV blocks the seq owns)
  - seq_lens[seq]
  - query_start_loc

Limit to first 6 calls where num_actual_tokens >= 2 to focus on decode
batches.

NOTE: chunked_prefill_paged_decode does NOT receive slot_mapping
directly (KV write is done before this in 0.19). It receives
block_table, seq_lens, query_start_loc, query, key_cache, value_cache.
We'll log those.
"""
import sys, re
from pathlib import Path
PATH = Path("/home/ubuntu/ai-2.10/lib/python3.13/site-packages/vllm/v1/attention/ops/chunked_prefill_paged_decode.py")
src = PATH.read_text()

# Insert at TOP of the function body, right after function entry.
# Find a stable anchor: "is_pow2 = block_size > 0 and (block_size & (block_size - 1) == 0)"
NEEDLE = "    is_pow2 = block_size > 0 and (block_size & (block_size - 1) == 0)"
HOOK = """    # ---- bench probe: metadata trace ----
    import os as _os
    try:
        _ts = getattr(chunked_prefill_paged_decode, "_bench_meta_cnt", 0)
        if query.shape[0] >= 2 and query.shape[0] <= 16 and _ts < 8:
            chunked_prefill_paged_decode._bench_meta_cnt = _ts + 1
            _bt = block_table[:query.shape[0]].cpu().tolist()
            _sl = seq_lens[:query.shape[0]].cpu().tolist()
            _qsl = query_start_loc.cpu().tolist() if query_start_loc is not None else None
            _os.write(2, ("[META #" + str(_ts+1) + "] q.shape=" + str(list(query.shape)) + " seq_lens=" + str(_sl) + " query_start_loc=" + str(_qsl) + "\\n").encode())
            for s in range(min(query.shape[0], 4)):
                _bt_row = _bt[s][:6]  # first 6 block ids per seq
                _os.write(2, ("    seq" + str(s) + " block_table[:6]=" + str(_bt_row) + "\\n").encode())
    except Exception as _ex:
        _os.write(2, ("[META] err: " + str(_ex) + "\\n").encode())
    # ---- end ----
    is_pow2 = block_size > 0 and (block_size & (block_size - 1) == 0)"""

if NEEDLE not in src:
    print("ERROR not found", file=sys.stderr); sys.exit(1)
PATH.write_text(src.replace(NEEDLE, HOOK, 1))
print("ok metadata probe")
