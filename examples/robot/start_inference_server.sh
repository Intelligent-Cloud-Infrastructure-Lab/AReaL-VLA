#!/usr/bin/env bash
# Start the VLA inference server in the SimpleVLA conda environment.
# Run this BEFORE launching libero_rl.py in the AReaL environment.
#
# Usage:
#   bash examples/robot/start_inference_server.sh \
#       --model_path Haozhan72/Openvla-oft-SFT-libero-spatial-traj1 \
#       --benchmark libero_spatial \
#       --device cuda:0 \
#       --address tcp://*:5555
#
# The server blocks until it receives a SHUTDOWN signal or Ctrl+C.
# Launch it in a separate terminal or with `tmux` / `screen`.

set -e

# ── defaults (override via CLI args) ─────────────────────────────────────────
MODEL_PATH=""
BENCHMARK="libero_spatial"
DEVICE="cuda:0"
ADDRESS="tcp://*:5555"
CONDA_ENV="simplevla"

# ── parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_path)  MODEL_PATH="$2";  shift 2 ;;
        --benchmark)   BENCHMARK="$2";   shift 2 ;;
        --device)      DEVICE="$2";      shift 2 ;;
        --address)     ADDRESS="$2";     shift 2 ;;
        --conda_env)   CONDA_ENV="$2";   shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$MODEL_PATH" ]]; then
    echo "Error: --model_path is required"
    echo "Usage: $0 --model_path <path_or_hf_id> [--benchmark <name>] [--device cuda:0] [--address tcp://*:5555]"
    exit 1
fi

echo "=================================================="
echo "  VLA Inference Server"
echo "  model    : $MODEL_PATH"
echo "  benchmark: $BENCHMARK"
echo "  device   : $DEVICE"
echo "  address  : $ADDRESS"
echo "  conda env: $CONDA_ENV"
echo "=================================================="

# Activate the SimpleVLA conda environment and launch the server
conda run -n "$CONDA_ENV" --no-capture-output \
    python areal/engine/vla_inference_server.py \
        --model_path  "$MODEL_PATH" \
        --benchmark   "$BENCHMARK" \
        --device      "$DEVICE" \
        --address     "$ADDRESS"
