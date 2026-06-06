# Qwen3.6-27B multimodal (image + video) — riscv64 + ROCm gfx1100, dual 7900 XTX, TP=2

Turns the `qwen3_5` VL checkpoint's **vision path on** (it is run text-only in
[`qwen3_6-27b-quant-bench.md`](qwen3_6-27b-quant-bench.md)). Same Quark W8A8-INT8
weights, same TP=2 graph stack + gemv INT8 patch.

- **Launcher:** [`servers/vllm/qwen3_6-27b-quark-int8-graph-tp2-multimodal.sh`](../servers/vllm/qwen3_6-27b-quark-int8-graph-tp2-multimodal.sh)
- **Restart wrapper:** [`servers/vllm/restart-multimodal.sh`](../servers/vllm/restart-multimodal.sh)
- **Guest deps (run once):** [`scripts/multimodal-deps-apt.sh`](../scripts/multimodal-deps-apt.sh)
- **Results:** [`results/2026-06-06.md`](../results/2026-06-06.md)

## TL;DR — four flags + one deps script

Starting from the text launcher, multimodal is:

```
--limit-mm-per-prompt '{"image":1,"video":1}'      # enable the vision path
--mm-processor-kwargs '{"max_pixels":200704}'      # 448px cap (bounds ViT)
--media-io-kwargs '{"video":{"backend":"pyav"}}'   # decode video via pyav, not opencv
--mm-encoder-attn-backend TRITON_ATTN              # ViT attention O(N) — the real fix
```

plus `scripts/multimodal-deps-apt.sh` on the guest for the pyav decode stack.
`max-model-len` drops 2048 → 40960 and `max-num-seqs` 8 → 2 (KV budget, see §4).

## 1. ViT attention: `TRITON_ATTN` is the load-bearing fix (O(N) vs O(N²))

The vision encoder defaults to **`TORCH_SDPA`**, which materialises the full N×N
attention-score matrix → `O(N²)` memory, where **N = total pre-merge ViT patches**.
That blows up fast:

```
SDPA peak = num_heads(16) × N² × 2 bytes = 32·N²
  16.7 Mpx image → N = 65536 → 32·N² = exactly 128 GiB → OOM at profile_run
```

The dummy *video* profile uses `N = max_model_len × 8`, so under SDPA you are forced
to keep `max_model_len ≤ ~1900` **and** a tiny `max_pixels` — i.e. SDPA, not the
model, was capping context and resolution.

`MMEncoderAttention` on rocm supports `{FLASH_ATTN, ROCM_AITER_FA, TRITON_ATTN,
TORCH_SDPA}`, and `get_vit_attn_backend` **honours an explicit override**
(`if backend is not None: assert in supported; return backend`). It auto-fell to
SDPA only because the gfx1x FLASH_ATTN-triton auto-path needs the `flash_attn`
package + `flash_attn.flash_attn_triton_amd` + `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`
(none present). Bypass it:

```
--mm-encoder-attn-backend TRITON_ATTN     # vLLM's own Triton attention, O(N), no extra deps
```

**Verified:** `max-model-len 16384` + `max_pixels 200704` (448px) → no OOM (SDPA
would need 32·N² ≈ 512 GiB) and output correct (full-HD Big Buck Bunny → "mossy
mound with a tree, dark burrow, dappled forest"), and *faster* than SDPA. With
TRITON_ATTN, video scales like text (long ctx + HD); the earlier 1792/20480 caps
were artifacts of SDPA's O(N²).

## 2. Image: cap `max_pixels`

Without a cap the processor uses its default `longest_edge` (≈16.7 Mpx) → the §1
128 GiB OOM at `profile_run`. `--mm-processor-kwargs '{"max_pixels":200704}'`
(=448², or 401408=448×896 for larger) bounds N. Verified correct: a red-circle +
blue-square test image is described accurately.

## 3. Video: pyav, not opencv

- vLLM registers loaders `opencv` / `opencv_dynamic` / `molmo2` / …; **`pyav` is
  NOT a registered loader.** Selecting video pyav is done on the *codec* axis via
  `--media-io-kwargs '{"video":{"backend":"pyav"}}'` — the default opencv loader's
  `load_bytes` has a `backend="pyav"` branch that uses `av.open`. (`VLLM_VIDEO_LOADER_BACKEND`
  selects the *loader*, a different axis — leave it default.)
- apt reality on this riscv mirror: `python3-opencv` pulls **libmysqlclient24**
  (unfetchable); `python3-av` pulls **libcaca0** whose pinned version 404s. The
  working set is `--no-install-recommends python3-av` + a manual current-libcaca0
  `dpkg -i`, plus a symlink of the system `av` into the venv (no riscv `av` wheel).
  All scripted in `scripts/multimodal-deps-apt.sh`.
- The Qwen-VL processor yields a **~constant total video-token count** (~3697 for a
  10 s clip) regardless of duration or `max_pixels`: `sample_frames_from_video` takes
  `np.linspace(0, total-1, num_frames)` ≈ 37 evenly-spaced frames and trades
  frame-count against per-frame resolution. For denser temporal sampling, raise
  `fps`/`num_frames` in `mm_processor_kwargs` (needs relaunch).

**Verified correct** across: synthetic moving-red-square (tracks left→center→right),
HD Big Buck Bunny, Jellyfish (sea-nettle), Sintel (snowy mountain figure), and a
~70 s 3-scene montage (returns per-scene timestamps + detects the loop). It is a
thinking model — answers arrive after a `<think>` block (~900 tokens), so allow
`max_tokens ≥ 1536`.

## 4. Dual-concurrency @ 40K — validated

`max-model-len 40960`, `max-num-seqs 2`: with the vision tower resident the KV pool
is **96,548 tokens** ("Maximum concurrency 2.36x"), so 2×40960 = 81920 fits. Note
4×40K is **not** KV-feasible at this context (would need a smaller `max-model-len`).
Only 16 of 64 layers grow KV (full-attn); the 48 GDN layers keep a fixed recurrent
state.

Clean per-client decode (streaming, TTFT-cancelled, `ignore_eos`, warmed — see
[`results/2026-06-06.md`](../results/2026-06-06.md)):

| N | per-client decode | aggregate | mean TTFT |
|---|---|---|---|
| 1 | 14.3–14.9 | 14.3–14.9 | ~5 s |
| 2 | 11.85 | **23.7** | ~7.8 s |

Concurrency **scales** (N=1→2 = 1.65×, per-stream only −17%), matching the
[`p2p-shm`](../p2p-repro/p2p-shm/RESULTS.md) Quark-INT8 sweep (15.41 / 12.31, 1.60×)
within ~5% → the 40K context does **not** hurt concurrency. Two concurrent
*different* videos (BBB + Jellyfish) are both correct and isolated, no cross-talk,
no OOM, `Running: 2 reqs`, server-reported generation throughput peaks ~22.6 tok/s.

⚠️ Measure decode **TTFT-cancelled** and warm the batch shape: a naive wall-clock
`tokens/elapsed` folds the multi-second TTFT (and any early `stop`) into the rate
and will *look* like concurrency regresses — it does not.

## Environment

This launcher runs on **`/home/ubuntu/vllm-venv`** (the writable rootfs venv), not
`/data/ai-2.11` like the other launchers, because the multimodal bring-up baked
changes into it:

- the **gemv INT8 patch** is baked into `.../compressed_tensors/triton_scaled_mm.py`
  (M==1 `_gemv_i8_kernel` / M≤8 `_gemv_dot_kernel`) — no `sitecustomize` override
  needed (the `/data` path applies it via `/home/ubuntu/gemv-patch/`).
- the system **`av`** is symlinked into `site-packages` (pyav, §3).
- vLLM 0.21 (`quark` W8A8-INT8 per-channel needs ≥0.21 — see quant-bench §3).

It still `source`s `vllm-serve-env.sh` (ROCm 7.2.3 / PyTorch 2.11 runtime) and adds
`VLLM_NCCL_SO_PATH`/`LD_PRELOAD=/home/ubuntu/librccl-rebuilt.so.1.0` + `CC=clang`.
