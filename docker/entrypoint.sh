#!/bin/bash
#
# inference container dispatcher.
#
# Two ways to invoke:
#   1. Positional:   docker run image ENGINE [extra-args]
#                    ENGINE = llamacpp | vllm | bash | --help
#                    extra-args are appended verbatim to the underlying server.
#   2. Env-driven:   docker run -e ENGINE=... -e MODEL=... image
#                    All knobs available as env vars (see --help).
#
# Both can be combined: positional ENGINE wins, env vars provide defaults.
set -e

# ─────────── help ───────────
print_help() {
    cat <<'EOF'
Usage:
  docker run [docker-flags] inference:riscv64 ENGINE [extra-args]
  docker run [docker-flags] -e ENGINE=... -e MODEL=... inference:riscv64

ENGINE = llamacpp | vllm | bash | --help

Common ENV (defaults shown):
  ENGINE                              required: llamacpp | vllm
  MODEL                               required: path under /data
  PORT             8000               listen port
  GPU              0                  which GPU (0=card0, 1=card1)
  EXTRA_ARGS       ""                 extra args appended to launcher

llamacpp-specific:
  ALIAS            <basename(MODEL)>  --alias served name
  PARALLEL         2                  --parallel
  CTX              16384              -c context size
  THREADS          8                  --threads
  TEMP             0.7
  TOP_P            0.95
  TOP_K            40
  REPEAT_PENALTY   1.05
  CHAT_TEMPLATE    ""                 if empty → use --jinja --reasoning-format none
                                      otherwise pass as --chat-template <value>
  FLASH_ATTN       on                 --flash-attn

vllm-specific:
  SERVED_NAME      <basename(MODEL)>  --served-model-name
  MAX_NUM_SEQS     2                  --max-num-seqs (concurrency)
  MAX_MODEL_LEN    4096
  DTYPE            float16
  GPU_MEM_UTIL     0.85
  TP_SIZE          1                  --tensor-parallel-size
  CUDAGRAPH_MODE   FULL_DECODE_ONLY   set to "" → --enforce-eager
  CUDAGRAPH_CAPTURE_SIZES  [1,2,4,8] JSON array of batch sizes to capture
                           larger sizes can crash on this riscv build
  MAX_CUDAGRAPH_CAPTURE_SIZE  8       cap for capture; keep ≤ MAX_NUM_SEQS

Examples:
  # Qwen3.6-35B via llama.cpp on card0:8001
  docker run -d --device=/dev/kfd --device=/dev/dri \
    -v /data/models:/data -p 8001:8001 \
    -e ENGINE=llamacpp -e MODEL=/data/Qwen3.6-35B-A3B-MXFP4_MOE.gguf \
    -e ALIAS=qwen36-a3b -e GPU=0 -e PORT=8001 \
    inference:riscv64

  # Gemma-4-E2B via vLLM on card1:8002
  docker run -d --device=/dev/kfd --device=/dev/dri \
    -v /data/models:/data -p 8002:8002 \
    -e ENGINE=vllm -e MODEL=/data/gemma-4-E2B-it -e SERVED_NAME=gemma4-e2b \
    -e GPU=1 -e PORT=8002 \
    inference:riscv64

  # Drop into a shell for debugging
  docker run --rm -it --device=/dev/kfd --device=/dev/dri \
    -v /data/models:/data \
    inference:riscv64 bash
EOF
}

# ─────────── arg/env parsing ───────────
# First positional arg overrides ENGINE env var.
if [ $# -gt 0 ]; then
    case "$1" in
        llamacpp|vllm|bash|--help|-h)
            ENGINE="$1"; shift ;;
    esac
fi

ENGINE="${ENGINE:---help}"

case "$ENGINE" in
  --help|-h)
    print_help; exit 0 ;;

  bash)
    exec /bin/bash "$@" ;;

  llamacpp)
    : "${MODEL:?MODEL env var required (path to .gguf file under /data)}"
    : "${PORT:=8000}"
    : "${GPU:=0}"
    : "${PARALLEL:=2}"
    : "${CTX:=16384}"
    : "${THREADS:=8}"
    : "${TEMP:=0.7}"
    : "${TOP_P:=0.95}"
    : "${TOP_K:=40}"
    : "${REPEAT_PENALTY:=1.05}"
    : "${FLASH_ATTN:=on}"
    : "${EXTRA_ARGS:=}"
    : "${CHAT_TEMPLATE:=}"
    # default ALIAS = basename of model with .gguf stripped
    if [ -z "${ALIAS:-}" ]; then
        ALIAS="$(basename "$MODEL" .gguf)"
    fi

    export HIP_VISIBLE_DEVICES="$GPU"
    export ROCR_VISIBLE_DEVICES="$GPU"

    # Chat template: prefer --jinja (model's native template, supports tool
    # calls); fall back to user-specified --chat-template if CHAT_TEMPLATE set.
    if [ -n "$CHAT_TEMPLATE" ]; then
        TPL_ARGS=(--chat-template "$CHAT_TEMPLATE")
    else
        # --reasoning-format none keeps <think> blocks in `content` field
        # (some clients don't read `reasoning_content`). See repo README.
        TPL_ARGS=(--jinja --reasoning-format none)
    fi

    echo "[inference-entrypoint] ENGINE=llamacpp"
    echo "  MODEL=$MODEL"
    echo "  ALIAS=$ALIAS  PORT=$PORT  GPU=$GPU  PARALLEL=$PARALLEL  CTX=$CTX"
    echo "  THREADS=$THREADS  FLASH_ATTN=$FLASH_ATTN"
    echo "  EXTRA_ARGS=$EXTRA_ARGS"

    set -x
    exec /opt/llama/bin/llama-server \
        -m "$MODEL" \
        --alias "$ALIAS" \
        -ngl 99 \
        -mg 0 --split-mode none \
        -c "$CTX" \
        --parallel "$PARALLEL" \
        --cont-batching \
        --flash-attn "$FLASH_ATTN" \
        --host 0.0.0.0 --port "$PORT" \
        --threads "$THREADS" \
        --no-mmap \
        --cache-ram 0 \
        "${TPL_ARGS[@]}" \
        --temp "$TEMP" --top-p "$TOP_P" --top-k "$TOP_K" --min-p 0.0 \
        --repeat-penalty "$REPEAT_PENALTY" \
        $EXTRA_ARGS \
        "$@"
    ;;

  vllm)
    # Source the canonical riscv64+rocm runtime env (sets the dozen
    # VLLM_*/NCCL_*/TORCH_USE_RTLD_GLOBAL/USE_LIBUV vars that the launcher
    # scripts on the VM rely on; without VLLM_ENABLE_V1_MULTIPROCESSING=0
    # the engine subprocess swallows its own stderr and you get
    # "Engine core initialization failed" with no clue as to why).
    if [ -f /opt/launchers/server-env.sh ]; then
        # shellcheck disable=SC1091
        source /opt/launchers/server-env.sh
    fi

    : "${MODEL:?MODEL env var required (path to model dir/file under /data)}"
    : "${PORT:=8000}"
    : "${GPU:=0}"
    : "${MAX_NUM_SEQS:=2}"
    : "${MAX_MODEL_LEN:=4096}"
    : "${DTYPE:=float16}"
    : "${GPU_MEM_UTIL:=0.85}"
    : "${TP_SIZE:=1}"
    : "${CUDAGRAPH_MODE:=FULL_DECODE_ONLY}"
    : "${EXTRA_ARGS:=}"
    if [ -z "${SERVED_NAME:-}" ]; then
        SERVED_NAME="$(basename "$MODEL")"
    fi

    # IMPORTANT: HIP_VISIBLE_DEVICES alone makes torch see zero GPUs on this
    # riscv build (see llm-bench memory). ROCR_VISIBLE_DEVICES is the right
    # one to bind a specific card.
    unset HIP_VISIBLE_DEVICES
    export ROCR_VISIBLE_DEVICES="$GPU"
    # already set in Dockerfile, but redundancy doesn't hurt
    export TOKENIZERS_PARALLELISM=false

    # IMPORTANT: explicit cudagraph_capture_sizes is required on this riscv64
    # build. Without it, vLLM picks defaults (e.g. 256/512) and the cudagraph
    # capture step crashes inside attention forward (the triton-fallback
    # path can't replay under graph capture for those sizes). The
    # [1,2,4,8] subset has been verified to capture cleanly. Override via
    # CUDAGRAPH_CAPTURE_SIZES (JSON array) for experimentation.
    : "${CUDAGRAPH_CAPTURE_SIZES:=[1,2,4,8]}"
    : "${MAX_CUDAGRAPH_CAPTURE_SIZE:=8}"
    if [ -n "$CUDAGRAPH_MODE" ]; then
        COMP_ARGS=(--compilation-config \
          "{\"mode\":0,\"cudagraph_mode\":\"$CUDAGRAPH_MODE\",\"cudagraph_capture_sizes\":$CUDAGRAPH_CAPTURE_SIZES,\"max_cudagraph_capture_size\":$MAX_CUDAGRAPH_CAPTURE_SIZE}")
    else
        COMP_ARGS=(--enforce-eager)
    fi

    echo "[inference-entrypoint] ENGINE=vllm"
    echo "  MODEL=$MODEL"
    echo "  SERVED_NAME=$SERVED_NAME  PORT=$PORT  GPU=$GPU"
    echo "  MAX_NUM_SEQS=$MAX_NUM_SEQS  MAX_MODEL_LEN=$MAX_MODEL_LEN  DTYPE=$DTYPE"
    echo "  GPU_MEM_UTIL=$GPU_MEM_UTIL  TP_SIZE=$TP_SIZE"
    echo "  CUDAGRAPH_MODE=${CUDAGRAPH_MODE:-(eager)}"
    echo "  EXTRA_ARGS=$EXTRA_ARGS"

    set -x
    exec /home/ubuntu/ai-2.10/bin/python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" \
        --served-model-name "$SERVED_NAME" \
        --dtype "$DTYPE" \
        --max-model-len "$MAX_MODEL_LEN" \
        --max-num-seqs "$MAX_NUM_SEQS" \
        --gpu-memory-utilization "$GPU_MEM_UTIL" \
        --tensor-parallel-size "$TP_SIZE" \
        --host 0.0.0.0 --port "$PORT" \
        "${COMP_ARGS[@]}" \
        $EXTRA_ARGS \
        "$@"
    ;;

  *)
    echo "ERROR: unknown ENGINE='$ENGINE' (must be llamacpp / vllm / bash)" >&2
    print_help
    exit 1 ;;
esac
