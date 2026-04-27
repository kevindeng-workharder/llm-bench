"""Layer-by-layer debug for Qwen3DecoderLayer:
  - Patch __init__ to capture prefix as self._layer_name
  - Patch forward to print per-row max after each sub-op
  - Limit to first call per layer (so we see one full pass through all 36 layers)
"""
import sys, re
from pathlib import Path

PATH = Path("/home/ubuntu/ai-2.10/lib/python3.13/site-packages/vllm/model_executor/models/qwen3.py")
src = PATH.read_text()

# 1) Patch __init__: append `self._layer_name = prefix` after super().__init__()
INIT_NEEDLE = """    def __init__(
        self,
        config: Qwen3Config,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size"""

INIT_REPL = """    def __init__(
        self,
        config: Qwen3Config,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self._layer_name = prefix
        self.hidden_size = config.hidden_size"""

if INIT_NEEDLE not in src:
    print("ERROR init needle not found", file=sys.stderr); sys.exit(1)
src = src.replace(INIT_NEEDLE, INIT_REPL, 1)

# 2) Patch forward
FWD_NEEDLE = """    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual"""

FWD_REPL = """    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        import os as _os
        _enable = (hidden_states is not None and hidden_states.dim() >= 1
                   and 2 <= hidden_states.size(0) <= 16
                   and getattr(self, "_bdbg_cnt", 0) < 1)
        def _dbg(stage, t):
            if not _enable:
                return
            try:
                _max = t.max(dim=-1).values.float().cpu().tolist()
                _min = t.min(dim=-1).values.float().cpu().tolist()
                _has_nan = any(x != x for x in _max) or any(x != x for x in _min)
                _flag = " HAS_NAN" if _has_nan else ""
                _ln = self._layer_name
                _os.write(2, ("[QL " + _ln + "-" + stage + "] max=" + str(["NaN" if x != x else round(x,2) for x in _max]) + " min=" + str(["NaN" if x != x else round(x,2) for x in _min]) + _flag + "\\n").encode())
            except Exception as _ex:
                _os.write(2, ("[QL " + getattr(self, "_layer_name", "?") + "-" + stage + "] err: " + str(_ex) + "\\n").encode())
        if _enable:
            self._bdbg_cnt = getattr(self, "_bdbg_cnt", 0) + 1
            _dbg("00_in", hidden_states)
            if residual is not None:
                _dbg("00_res", residual)
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        _dbg("A_ln1", hidden_states)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )
        _dbg("B_attn", hidden_states)

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        _dbg("C_ln2", hidden_states)
        hidden_states = self.mlp(hidden_states)
        _dbg("D_mlp", hidden_states)
        return hidden_states, residual"""

if FWD_NEEDLE not in src:
    print("ERROR fwd needle not found", file=sys.stderr); sys.exit(1)
src = src.replace(FWD_NEEDLE, FWD_REPL, 1)

PATH.write_text(src)
print("ok qwen3 layers v2")
