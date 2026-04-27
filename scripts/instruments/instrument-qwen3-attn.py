"""Instrument Qwen3Attention.forward — print per-row max after each sub-op
to find which one introduces NaN.

Sub-ops:
  P_qkv:  after qkv_proj
  P_qn:   after q_norm
  P_kn:   after k_norm
  P_rope: after rotary_emb
  P_attn: after self.attn
  P_o:    after o_proj
"""
import sys, re
from pathlib import Path

PATH = Path("/home/ubuntu/ai-2.10/lib/python3.13/site-packages/vllm/model_executor/models/qwen3.py")
src = PATH.read_text()

# Patch Qwen3Attention __init__ to capture prefix as _attn_name (before super-class super().__init__())
ATTN_INIT_NEEDLE = """        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()"""

ATTN_INIT_REPL = """        super().__init__()
        self._attn_name = prefix
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()"""

if ATTN_INIT_NEEDLE not in src:
    print("ERROR attn init needle not found", file=sys.stderr); sys.exit(1)
src = src.replace(ATTN_INIT_NEEDLE, ATTN_INIT_REPL, 1)

# Patch Qwen3Attention.forward
ATTN_FWD_NEEDLE = """    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # Add qk-norm
        q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim)
        q_by_head = self.q_norm(q_by_head)
        q = q_by_head.view(q.shape)
        k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.view(k.shape)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output"""

ATTN_FWD_REPL = """    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        import os as _os
        _enable = (hidden_states is not None and hidden_states.dim() >= 1
                   and 2 <= hidden_states.size(0) <= 16
                   and getattr(self, "_attn_dbg_cnt", 0) < 3)
        def _dbg2(stage, t):
            if not _enable or t is None:
                return
            try:
                # Reduce to per-row by treating any non-row dims as flattened
                t2 = t.view(t.size(0), -1)
                _max = t2.max(dim=-1).values.float().cpu().tolist()
                _min = t2.min(dim=-1).values.float().cpu().tolist()
                _has_nan = any(x != x for x in _max) or any(x != x for x in _min)
                _flag = " HAS_NAN" if _has_nan else ""
                _an = self._attn_name
                _os.write(2, ("[QA " + _an + "-" + stage + "] max=" + str(["NaN" if x != x else round(x,3) for x in _max]) + " min=" + str(["NaN" if x != x else round(x,3) for x in _min]) + _flag + "\\n").encode())
            except Exception as _ex:
                _os.write(2, ("[QA " + getattr(self, "_attn_name", "?") + "-" + stage + "] err: " + str(_ex) + "\\n").encode())
        if _enable:
            self._attn_dbg_cnt = getattr(self, "_attn_dbg_cnt", 0) + 1
            _dbg2("00_in", hidden_states)
        qkv, _ = self.qkv_proj(hidden_states)
        _dbg2("P_qkv", qkv)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        _dbg2("P_q", q)
        _dbg2("P_k", k)
        _dbg2("P_v", v)
        # Add qk-norm
        q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim)
        q_by_head = self.q_norm(q_by_head)
        q = q_by_head.view(q.shape)
        _dbg2("P_qn", q)
        k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.view(k.shape)
        _dbg2("P_kn", k)
        q, k = self.rotary_emb(positions, q, k)
        _dbg2("P_q_rope", q)
        _dbg2("P_k_rope", k)
        attn_output = self.attn(q, k, v)
        _dbg2("P_attn", attn_output)
        output, _ = self.o_proj(attn_output)
        _dbg2("P_o", output)
        return output"""

if ATTN_FWD_NEEDLE not in src:
    print("ERROR attn fwd needle not found", file=sys.stderr); sys.exit(1)
src = src.replace(ATTN_FWD_NEEDLE, ATTN_FWD_REPL, 1)

PATH.write_text(src)
print("ok qwen3-attn instrument")
