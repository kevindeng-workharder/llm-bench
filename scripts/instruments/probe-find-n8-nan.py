"""Find where NaN/inf first appears in the model forward at N=8.

Patches Qwen3DecoderLayer.forward (and __init__ to capture layer index)
to inject per-stage NaN/inf checks. Prints only when an inf or NaN is
first detected, with a small budget so logs stay readable.

Active when batch dim in [2, 16] (catches N=8 decode steps) AND we're
NOT inside a CUDA graph capture (probe uses .item() which can't run
during capture).
"""
import sys, re
from pathlib import Path

PATH = Path("/home/ubuntu/ai-2.10/lib/python3.13/site-packages/vllm/model_executor/models/qwen3.py")
src = PATH.read_text()

# Strip prior probe blocks
src = re.sub(r"        # ---- N8PROBE[^\n]*\n(?:.*\n)*?        # ---- end ----\n", "", src)
src = re.sub(r"        # ---- N8PROBE-INIT[^\n]*\n(?:.*\n)*?        # ---- end ----\n", "", src)

# 1. Patch __init__ to capture layer index from prefix
INIT_NEEDLE = """    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        set_default_rope_theta(config, default_theta=1000000)"""

INIT_REPL = """    ) -> None:
        super().__init__()
        # ---- N8PROBE-INIT ----
        try:
            import re as _r
            _m = _r.search(r"layers\\.(\\d+)", str(prefix))
            self._layer_idx = int(_m.group(1)) if _m else -1
        except Exception:
            self._layer_idx = -1
        # ---- end ----
        self.hidden_size = config.hidden_size
        set_default_rope_theta(config, default_theta=1000000)"""

if INIT_NEEDLE not in src:
    print("ERROR init needle not found", file=sys.stderr); sys.exit(1)
src = src.replace(INIT_NEEDLE, INIT_REPL, 1)

# 2. Replace the forward method body with one that includes per-stage probes
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
        # ---- N8PROBE ----
        import os as _os
        import torch as _torch
        _bs = hidden_states.size(0) if hidden_states is not None else 0
        try:
            _capturing = _torch.cuda.is_current_stream_capturing()
        except Exception:
            _capturing = False
        _active = (2 <= _bs <= 16
                   and not _capturing
                   and getattr(Qwen3DecoderLayer, "_n8_print_budget", 64) > 0)
        _layer_idx = getattr(self, "_layer_idx", -1)
        def _check(stage, t):
            if not _active or t is None:
                return
            try:
                tf = t.float()
                _has_nan = bool(tf.isnan().any().item())
                _has_inf = bool(tf.isinf().any().item())
                if _has_nan or _has_inf:
                    rmax = tf.abs().max().item()
                    Qwen3DecoderLayer._n8_print_budget = getattr(Qwen3DecoderLayer, "_n8_print_budget", 64) - 1
                    # Find which row(s) have NaN/inf
                    bad_rows = []
                    if t.dim() >= 2:
                        per_row = t.float().view(t.size(0), -1)
                        nan_mask = per_row.isnan().any(dim=-1) | per_row.isinf().any(dim=-1)
                        bad_rows = nan_mask.nonzero(as_tuple=True)[0].cpu().tolist()
                    _os.write(2, ("[N8 L" + str(_layer_idx) + " " + stage + "] bs=" + str(_bs) + " HAS_NAN=" + str(_has_nan) + " HAS_INF=" + str(_has_inf) + " absmax=" + str(round(rmax,2)) + " bad_rows=" + str(bad_rows) + "\\n").encode())
            except Exception as _ex:
                _os.write(2, ("[N8 L" + str(_layer_idx) + " " + stage + "] err: " + str(_ex)[:60] + "\\n").encode())
        _check("00_in", hidden_states)
        # ---- end ----
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        # ---- N8PROBE ----
        _check("01_after_iln", hidden_states)
        # ---- end ----
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )
        # ---- N8PROBE ----
        _check("02_after_attn", hidden_states)
        # ---- end ----

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        # ---- N8PROBE ----
        _check("03_after_pln_h", hidden_states)
        _check("03_after_pln_r", residual)
        # ---- end ----
        hidden_states = self.mlp(hidden_states)
        # ---- N8PROBE ----
        _check("04_after_mlp", hidden_states)
        # ---- end ----
        return hidden_states, residual"""

if FWD_NEEDLE not in src:
    print("ERROR forward needle not found", file=sys.stderr); sys.exit(1)
src = src.replace(FWD_NEEDLE, FWD_REPL, 1)

PATH.write_text(src)
print("ok N8PROBE installed (with capture-skip)")
