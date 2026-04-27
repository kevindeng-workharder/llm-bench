"""v4: probe ALL layers but print only when row has NaN/inf, with row
indices captured. Filters real data (positions non-zero). Big budget.
"""
import sys, re
from pathlib import Path

PATH = Path("/home/ubuntu/ai-2.10/lib/python3.13/site-packages/vllm/model_executor/models/qwen3.py")
src = PATH.read_text()

src = re.sub(r"        # ---- N8PROBE[^\n]*\n(?:.*\n)*?        # ---- end ----\n", "", src)
src = re.sub(r"        # ---- N8PROBE-INIT[^\n]*\n(?:.*\n)*?        # ---- end ----\n", "", src)
src = re.sub(r"        # ---- N8STATS[^\n]*\n(?:.*\n)*?        # ---- end ----\n", "", src)

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
        # ---- N8STATS ----
        import os as _os
        import torch as _torch
        _bs = hidden_states.size(0) if hidden_states is not None else 0
        try:
            _capturing = _torch.cuda.is_current_stream_capturing()
        except Exception:
            _capturing = False
        _layer_idx = getattr(self, "_layer_idx", -1)
        # Trigger only on real-data batches (positions has non-zero values)
        # AND only when bs is in mixed-prefill+decode range (>= 8)
        _stats_active = False
        if (_bs >= 8 and not _capturing
                and getattr(Qwen3DecoderLayer, "_n8_stats_budget", 1500) > 0):
            try:
                _pos_max = int(positions.max().item()) if positions is not None else 0
                if _pos_max > 0:
                    _stats_active = True
            except Exception:
                pass
        def _check(stage, t):
            if not _stats_active or t is None or t.dim() < 2:
                return
            try:
                tf = t.float().view(t.size(0), -1)
                row_has_bad = tf.isnan().any(dim=-1) | tf.isinf().any(dim=-1)
                bad_rows = row_has_bad.nonzero(as_tuple=True)[0].cpu().tolist()
                if bad_rows:
                    Qwen3DecoderLayer._n8_stats_budget = getattr(Qwen3DecoderLayer, "_n8_stats_budget", 1500) - 1
                    _absmax = tf.abs().nan_to_num(0.0, posinf=0.0, neginf=0.0).amax().item()
                    _os.write(2, ("[N8 L" + str(_layer_idx) + " " + stage + "] bs=" + str(_bs) + " bad_rows=" + str(bad_rows) + " absmax(finite)=" + f"{_absmax:.1f}" + "\\n").encode())
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
        # ---- N8STATS ----
        _check("01_after_iln", hidden_states)
        # ---- end ----
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )
        # ---- N8STATS ----
        _check("02_after_attn", hidden_states)
        # ---- end ----

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        # ---- N8STATS ----
        _check("03_after_pln_h", hidden_states)
        # ---- end ----
        hidden_states = self.mlp(hidden_states)
        # ---- N8STATS ----
        _check("04_after_mlp", hidden_states)
        # ---- end ----
        return hidden_states, residual"""

if FWD_NEEDLE not in src:
    print("ERROR forward needle not found", file=sys.stderr); sys.exit(1)
src = src.replace(FWD_NEEDLE, FWD_REPL, 1)

PATH.write_text(src)
print("ok N8STATS v4 (all layers, NaN/inf-only, real-data filter)")
