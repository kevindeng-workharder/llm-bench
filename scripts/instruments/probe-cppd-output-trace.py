"""Trace output tensor at chunked_prefill_paged_decode entry, after prefill,
after decode kernel.

Pre-fill output with zeros to distinguish "kernel didn't write" from
"kernel wrote NaN".
"""
import sys, re
from pathlib import Path
PATH = Path("/home/ubuntu/ai-2.10/lib/python3.13/site-packages/vllm/v1/attention/ops/chunked_prefill_paged_decode.py")
src = PATH.read_text()
src = re.sub(r"\n    # ---- bench probe[^\n]*\n(?:.*\n)*?    # ---- end ----\n", "\n", src)

# 1. Add zero-prefill + entry log + after context_attention_fwd log
NEEDLE_PRE = """    if max_query_len > 1:
        context_attention_fwd("""

REPL_PRE = """    # ---- bench probe: zero output + entry log ----
    import os as _os
    _bench_active = (seq_lens.shape[0] >= 2 and seq_lens.shape[0] <= 16
                     and getattr(chunked_prefill_paged_decode, "_bench_cnt", 0) < 4)
    if _bench_active:
        chunked_prefill_paged_decode._bench_cnt = getattr(chunked_prefill_paged_decode, "_bench_cnt", 0) + 1
        output.zero_()
        try:
            _qsl = query_start_loc.cpu().tolist()
            _sl = seq_lens.cpu().tolist()
            _os.write(2, ("[PROBE #" + str(chunked_prefill_paged_decode._bench_cnt) + "] num_seqs=" + str(seq_lens.shape[0]) + " query_start_loc=" + str(_qsl) + " seq_lens=" + str(_sl) + " max_query_len=" + str(max_query_len) + "\\n").encode())
        except Exception as _ex:
            _os.write(2, ("[PROBE] err: " + str(_ex) + "\\n").encode())
    # ---- end ----
    if max_query_len > 1:
        # ---- bench probe: pre-context_attention_fwd ----
        if _bench_active:
            try:
                for _s in range(seq_lens.shape[0]):
                    _start = int(query_start_loc[_s].item())
                    _end = int(query_start_loc[_s+1].item())
                    if _end > _start:
                        _o = output[_start:_end].view(_end-_start, -1).max(dim=-1).values.float().cpu().tolist()
                        _o_min = output[_start:_end].view(_end-_start, -1).min(dim=-1).values.float().cpu().tolist()
                        _has_nan = any(x!=x for x in _o)
                        _os.write(2, ("    BEFORE_CTX seq" + str(_s) + " out[" + str(_start) + ":" + str(_end) + "] max=" + str(["NaN" if x!=x else round(x,3) for x in _o]) + " min=" + str(["NaN" if x!=x else round(x,3) for x in _o_min]) + (" HAS_NAN" if _has_nan else "") + "\\n").encode())
            except Exception as _ex:
                _os.write(2, ("[PROBE BEFORE_CTX] err: " + str(_ex) + "\\n").encode())
        # ---- end ----
        context_attention_fwd("""

if NEEDLE_PRE not in src:
    print("ERROR pre needle not found", file=sys.stderr); sys.exit(1)
src = src.replace(NEEDLE_PRE, REPL_PRE, 1)

# 2. After context_attention_fwd block ends, log output again
NEEDLE_POST = """    block_size = value_cache.shape[3]
    num_seqs = len(seq_lens)"""

REPL_POST = """    # ---- bench probe: after context_attention_fwd ----
    if _bench_active:
        try:
            for _s in range(seq_lens.shape[0]):
                _start = int(query_start_loc[_s].item())
                _end = int(query_start_loc[_s+1].item())
                if _end > _start:
                    _o = output[_start:_end].view(_end-_start, -1).max(dim=-1).values.float().cpu().tolist()
                    _has_nan = any(x!=x for x in _o)
                    _os.write(2, ("    AFTER_CTX seq" + str(_s) + " out[" + str(_start) + ":" + str(_end) + "] max=" + str(["NaN" if x!=x else round(x,3) for x in _o]) + (" HAS_NAN" if _has_nan else "") + "\\n").encode())
        except Exception as _ex:
            _os.write(2, ("[PROBE AFTER_CTX] err: " + str(_ex) + "\\n").encode())
    # ---- end ----
    block_size = value_cache.shape[3]
    num_seqs = len(seq_lens)"""

if NEEDLE_POST not in src:
    print("ERROR post needle not found", file=sys.stderr); sys.exit(1)
src = src.replace(NEEDLE_POST, REPL_POST, 1)

PATH.write_text(src)
print("ok output-trace probe")
