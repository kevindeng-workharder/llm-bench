"""Apply [riscv-patch] suggested fix: add stream-aware torch.cuda.synchronize()
between unified_kv_cache_update and unified_attention_with_output, guarded
to skip during cudagraph capture (which is the only thing that broke last time).

The 0.19 split between KV-cache write and attention compute relies on the
fake `kv_cache_dummy_dep` tensor as an op dependency, but on RDNA3 +
ROCm the actual GPU stream ordering doesn't always wait for the cache
write to complete before attention reads it. For some seqs, attention
reads stale/zero KV → softmax of (Q @ stale_K) → NaN.

Instead of synchronizing AFTER unified_kv_cache_update, we just force
the call-site to wait. Only synchronize when NOT capturing a graph.
"""
import sys, re
from pathlib import Path

PATH = Path("/home/ubuntu/ai-2.10/lib/python3.13/site-packages/vllm/model_executor/layers/attention/attention.py")
src = PATH.read_text()

# Locate the kv_cache_update -> unified_attention_with_output sequence in the
# direct-call branch (use_direct_call). There are two call sites (use_direct_call
# True and False). Patch BOTH.

NEEDLE_DIRECT = """                if (
                    not self.attn_backend.forward_includes_kv_cache_update
                    and self.kv_sharing_target_layer_name is None
                    and key is not None
                    and value is not None
                ):
                    kv_cache_dummy_dep = unified_kv_cache_update(
                        key, value, self.layer_name
                    )
                unified_attention_with_output(
                    query,
                    key,
                    value,
                    output,
                    self.layer_name,
                    kv_cache_dummy_dep=kv_cache_dummy_dep,
                )"""

REPL_DIRECT = """                if (
                    not self.attn_backend.forward_includes_kv_cache_update
                    and self.kv_sharing_target_layer_name is None
                    and key is not None
                    and value is not None
                ):
                    kv_cache_dummy_dep = unified_kv_cache_update(
                        key, value, self.layer_name
                    )
                # ---- llm-bench fix: stream-aware sync to enforce KV write happens-before attn read
                # Required on RDNA3 gfx1100 where the kv_cache_dummy_dep tensor doesn't
                # actually create a stream dependency under the V1 split-op design.
                if not torch.cuda.is_current_stream_capturing():
                    torch.cuda.synchronize()
                # ---- end fix ----
                unified_attention_with_output(
                    query,
                    key,
                    value,
                    output,
                    self.layer_name,
                    kv_cache_dummy_dep=kv_cache_dummy_dep,
                )"""

if NEEDLE_DIRECT not in src:
    print("ERROR direct needle not found", file=sys.stderr); sys.exit(1)
src = src.replace(NEEDLE_DIRECT, REPL_DIRECT, 1)

# Same for the non-direct (torch.ops.vllm) branch
NEEDLE_OPS = """                if (
                    not self.attn_backend.forward_includes_kv_cache_update
                    and self.kv_sharing_target_layer_name is None
                    and key is not None
                    and value is not None
                ):
                    kv_cache_dummy_dep = torch.ops.vllm.unified_kv_cache_update(
                        key, value, self.layer_name
                    )
                torch.ops.vllm.unified_attention_with_output(
                    query,
                    key,
                    value,
                    output,
                    self.layer_name,
                    kv_cache_dummy_dep=kv_cache_dummy_dep,
                )"""

REPL_OPS = """                if (
                    not self.attn_backend.forward_includes_kv_cache_update
                    and self.kv_sharing_target_layer_name is None
                    and key is not None
                    and value is not None
                ):
                    kv_cache_dummy_dep = torch.ops.vllm.unified_kv_cache_update(
                        key, value, self.layer_name
                    )
                # ---- llm-bench fix: same as above ----
                if not torch.cuda.is_current_stream_capturing():
                    torch.cuda.synchronize()
                # ---- end fix ----
                torch.ops.vllm.unified_attention_with_output(
                    query,
                    key,
                    value,
                    output,
                    self.layer_name,
                    kv_cache_dummy_dep=kv_cache_dummy_dep,
                )"""

if NEEDLE_OPS in src:
    src = src.replace(NEEDLE_OPS, REPL_OPS, 1)
    print("ok patched both branches")
else:
    print("ok patched direct branch only")

PATH.write_text(src)
