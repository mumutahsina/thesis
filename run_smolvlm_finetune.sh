#!/usr/bin/env bash
# Fine-tune SmolVLM instruction models on processed_dataset using LoRA.
# Supports the following HF models:
#  1) HuggingFaceTB/SmolVLM2-2.2B-Instruct
#  2) HuggingFaceTB/SmolVLM-Instruct
#  3) HuggingFaceTB/SmolVLM-500M-Instruct
#
# Usage examples:
#   bash run_smolvlm_finetune.sh 2.2b   # uses HuggingFaceTB/SmolVLM2-2.2B-Instruct
#   bash run_smolvlm_finetune.sh 1b     # uses HuggingFaceTB/SmolVLM-Instruct
#   bash run_smolvlm_finetune.sh 500m   # uses HuggingFaceTB/SmolVLM-500M-Instruct
#
# Optional flags:
#   --no-venv           Do not create/activate .venv
#   --epochs N          Number of epochs (default: 1)
#   --batch-size N      Per-device batch size (default: 1)
#   --grad-accum N      Gradient accumulation steps (default: 4)
#   --lr LR             Learning rate (default: 2e-4)
#
# Notes:
# - This script uses scripts/finetune/train.py which tries AutoModelForVision2Seq first,
#   then falls back to seq2seq if needed. Adjust if SmolVLM requires a specific class.
# - Ensure processed_dataset exists. Run: bash run_preprocessing.sh

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
USE_VENV=1
EPOCHS=1
BATCH=1
ACC=4
LR=2e-4

# Default: none selected yet
MODEL_CHOICE=""
MODEL_ID=""

pick_model() {
  case "$1" in
    2.2b|2b|2_2b|smolvlm2|large)
      MODEL_ID="HuggingFaceTB/SmolVLM2-2.2B-Instruct"
      ;;
    1b|base|smolvlm)
      MODEL_ID="HuggingFaceTB/SmolVLM-Instruct"
      ;;
    500m|0.5b|small)
      MODEL_ID="HuggingFaceTB/SmolVLM-500M-Instruct"
      ;;
    *)
      echo "Unknown model choice: $1" >&2
      echo "Valid choices: 2.2b | 1b | 500m" >&2
      exit 1
      ;;
  esac
}

# Parse args
if [[ $# -eq 0 ]]; then
  echo "Usage: bash run_smolvlm_finetune.sh <2.2b|1b|500m> [--no-venv] [--epochs N] [--batch-size N] [--grad-accum N] [--lr LR]"
  exit 1
fi

# First positional argument selects the model
MODEL_CHOICE="$1"; shift || true
pick_model "$MODEL_CHOICE"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-venv) USE_VENV=0; shift ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --batch-size) BATCH="$2"; shift 2 ;;
    --grad-accum) ACC="$2"; shift 2 ;;
    --lr) LR="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: bash run_smolvlm_finetune.sh <2.2b|1b|500m> [--no-venv] [--epochs N] [--batch-size N] [--grad-accum N] [--lr LR]"
      exit 0 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

# Python
if command -v python3 >/dev/null 2>&1; then PY=python3; else PY=python; fi

# venv
if [ "$USE_VENV" -eq 1 ]; then
  if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at $VENV_DIR"
    "$PY" -m venv "$VENV_DIR"
  fi
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  PIP=pip
else
  PIP=pip
fi

# Deps
$PY -m pip install --upgrade pip
$PIP install -r "$PROJECT_DIR/requirements.txt"

# Kick off training
OUT_DIR="$PROJECT_DIR/runs/finetune/$(echo "$MODEL_ID" | tr '/' '_')"
$PY "$PROJECT_DIR/scripts/finetune/train.py" \
  --model "$MODEL_ID" \
  --data-root "$PROJECT_DIR/processed_dataset" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH" \
  --grad-accum "$ACC" \
  --lr "$LR" \
  --output-dir "$OUT_DIR"

echo "Training complete. Check outputs in: $OUT_DIR"