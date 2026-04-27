"""Hot patch: wrap Attention.forward to do attention compute in bf16 when
input is fp16, then cast back. Sidesteps the fp16 NaN-overflow in
paged_attention_rocm without rebuilding the C++ kernel.

Modifies vllm/model_executor/layers/attention/attention.py.
"""
import sys, re
from pathlib import Path

PATH = Path("/home/ubuntu/ai-2.10/lib/python3.13/site-packages/vllm/model_executor/layers/attention/attention.py")
src = PATH.read_text()

NEEDLE = """    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        # For some alternate attention backends like MLA the attention output
        # shape does not match the query shape, so we optionally let the model
        # definition specify the output tensor shape.
        output_shape: torch.Size | None = None,
    ) -> torch.Tensor:"""

REPL = """    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        # For some alternate attention backends like MLA the attention output
        # shape does not match the query shape, so we optionally let the model
        # definition specify the output tensor shape.
        output_shape: torch.Size | None = None,
    ) -> torch.Tensor:
        # ---- llm-bench fp16-NaN workaround ----
        # paged_attention_rocm on RDNA3 gfx1100 produces NaN for some rows
        # when query.dtype == fp16 and batch > 1. Casting to bf16 (same
        # exponent range as fp32) sidesteps the overflow. Cast output back
        # to caller's dtype.
        _bench_orig_dtype = query.dtype
        if _bench_orig_dtype == torch.float16:
            query = query.to(torch.bfloat16)
            if key is not None:
                key = key.to(torch.bfloat16)
            if value is not None:
                value = value.to(torch.bfloat16)
        try:
            result = self._bench_orig_forward(query, key, value, output_shape)
        finally:
            pass
        if _bench_orig_dtype == torch.float16 and result is not None:
            result = result.to(torch.float16)
        return result

    def _bench_orig_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output_shape: torch.Size | None = None,
    ) -> torch.Tensor:"""

if NEEDLE not in src:
    print("ERROR not found", file=sys.stderr); sys.exit(1)
src = src.replace(NEEDLE, REPL, 1)
PATH.write_text(src)
print("ok bf16-wrap patch applied")
