"""v2: only print when num_seqs >= 2 (i.e., true batched decode/prefill).
Use len(seq_lens) as the num_seqs check."""
import sys, re
from pathlib import Path
PATH = Path("/home/ubuntu/ai-2.10/lib/python3.13/site-packages/vllm/v1/attention/ops/chunked_prefill_paged_decode.py")
src = PATH.read_text()
# Strip prior probe
src = re.sub(r"\n    # ---- bench probe[^\n]*\n(?:.*\n)*?    # ---- end ----\n", "\n", src)
NEEDLE = "    is_pow2 = block_size > 0 and (block_size & (block_size - 1) == 0)"
HOOK = """    # ---- bench probe v2: metadata trace, only when num_seqs >= 2 ----
    import os as _os
    try:
        _num_seqs = seq_lens.shape[0]
        _ts = getattr(chunked_prefill_paged_decode, "_bench_meta_cnt", 0)
        if _num_seqs >= 2 and _num_seqs <= 16 and _ts < 6:
            chunked_prefill_paged_decode._bench_meta_cnt = _ts + 1
            _bt = block_table[:_num_seqs].cpu().tolist()
            _sl = seq_lens[:_num_seqs].cpu().tolist()
            _qsl = query_start_loc.cpu().tolist() if query_start_loc is not None else None
            _os.write(2, ("[META2 #" + str(_ts+1) + "] num_seqs=" + str(_num_seqs) + " q.shape=" + str(list(query.shape)) + " seq_lens=" + str(_sl) + " query_start_loc=" + str(_qsl) + "\\n").encode())
            for s in range(min(_num_seqs, 4)):
                _bt_row = _bt[s][:6]
                _os.write(2, ("    seq" + str(s) + " block_table[:6]=" + str(_bt_row) + "\\n").encode())
    except Exception as _ex:
        _os.write(2, ("[META2] err: " + str(_ex) + "\\n").encode())
    # ---- end ----
    is_pow2 = block_size > 0 and (block_size & (block_size - 1) == 0)"""
if NEEDLE not in src:
    print("ERROR not found", file=sys.stderr); sys.exit(1)
PATH.write_text(src.replace(NEEDLE, HOOK, 1))
print("ok v2")
