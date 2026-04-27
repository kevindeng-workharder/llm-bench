"""v3: broaden bs range to catch real N=8 traffic. Print compact summary
showing whether rows are identical, with positions[:bs] to identify
synthetic-vs-real data.

Filter goal: capture real N=8 attention traffic, not vLLM dummy_run /
graph capture / profiler synthetic batches.
"""
import sys, re
from pathlib import Path

PATH = Path("/home/ubuntu/ai-2.10/lib/python3.13/site-packages/vllm/model_executor/models/qwen3.py")
src = PATH.read_text()

# Strip prior probe blocks
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
        # Trigger when bs in [2, 64] AND positions are NOT all zeros (filters
        # synthetic dummy_run batches which use positions=0 for all rows)
        _stats_active = False
        if (2 <= _bs <= 64 and not _capturing
                and getattr(Qwen3DecoderLayer, "_n8_stats_budget", 200) > 0
                and _layer_idx in (0, 35)):
            try:
                _pos_max = int(positions.max().item()) if positions is not None else 0
                if _pos_max > 0:
                    _stats_active = True
            except Exception:
                pass
        def _stats(stage, t):
            if not _stats_active or t is None or t.dim() < 2:
                return
            try:
                tf = t.float().view(t.size(0), -1)
                # Compare rows pairwise — count how many distinct rows we have
                # (use fingerprint of first 8 elements)
                fingerprints = tf[:, :8].cpu().tolist()
                fp_set = set(tuple(round(x, 4) for x in row) for row in fingerprints)
                num_distinct = len(fp_set)
                rmax = tf.amax(dim=-1).cpu().tolist()
                rmin = tf.amin(dim=-1).cpu().tolist()
                Qwen3DecoderLayer._n8_stats_budget = getattr(Qwen3DecoderLayer, "_n8_stats_budget", 200) - 1
                fmt = lambda L: "[" + ",".join(("nan" if x!=x else f"{x:.2f}") for x in L) + "]"
                _os.write(2, ("[ST L" + str(_layer_idx) + " " + stage + "] bs=" + str(_bs) + " distinct=" + str(num_distinct) + " max=" + fmt(rmax) + " min=" + fmt(rmin) + "\\n").encode())
            except Exception as _ex:
                _os.write(2, ("[ST L" + str(_layer_idx) + " " + stage + "] err: " + str(_ex)[:60] + "\\n").encode())
        if _stats_active and _layer_idx == 0:
            try:
                pos = positions.flatten()[:min(16, _bs)].cpu().tolist() if positions is not None else None
                _os.write(2, ("[ST ENTRY] bs=" + str(_bs) + " positions=" + str(pos) + "\\n").encode())
            except Exception as _ex:
                _os.write(2, ("[ST ENTRY] err: " + str(_ex)[:60] + "\\n").encode())
        _stats("00_in", hidden_states)
        # ---- end ----
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
        # ---- N8STATS ----
        _stats("02_after_attn", hidden_states)
        # ---- end ----

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        # ---- N8STATS ----
        _stats("04_after_mlp", hidden_states)
        # ---- end ----
        return hidden_states, residual"""

if FWD_NEEDLE not in src:
    print("ERROR forward needle not found", file=sys.stderr); sys.exit(1)
src = src.replace(FWD_NEEDLE, FWD_REPL, 1)

PATH.write_text(src)
print("ok N8STATS v3 (real-data filter, distinct-row count)")
