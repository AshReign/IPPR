#!/usr/bin/env bash
set -euo pipefail
P2_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONDONTWRITEBYTECODE=1
export YOLO_CONFIG_DIR="$P2_ROOT/runtime/ultralytics"
export MPLCONFIGDIR="$P2_ROOT/runtime/matplotlib"
export TORCH_HOME="$P2_ROOT/runtime/torch"
export HF_HOME="$P2_ROOT/runtime/huggingface"
export YOLO_AUTOINSTALL=false
export WANDB_DISABLED=true
export WANDB_MODE=disabled
export COMET_MODE=DISABLED
export HF_HUB_DISABLE_TELEMETRY=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p "$YOLO_CONFIG_DIR" "$MPLCONFIGDIR" "$TORCH_HOME" "$HF_HOME"
cd "$P2_ROOT"
export PYTHONPATH="$P2_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "${P2_PYTHON:-python}" "$@"
