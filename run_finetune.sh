#!/usr/bin/env bash
# Helper to run LoRA fine-tuning on processed_dataset
# Usage:
#   bash run_finetune.sh [--no-venv] [--model <hf_model_id>] [--epochs N] [--batch-size N]

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
USE_VENV=1
MODEL="Salesforce/blip2-flan-t5-base"
EPOCHS=1
BATCH=1
ACC=4

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-venv) USE_VENV=0; shift ;;
    --model) MODEL="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --batch-size) BATCH="$2"; shift 2 ;;
    --grad-accum) ACC="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: bash run_finetune.sh [--no-venv] [--model <id>] [--epochs N] [--batch-size N] [--grad-accum N]"
      exit 0 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if command -v python3 >/dev/null 2>&1; then PY=python3; else PY=python; fi

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

$PY -m pip install --upgrade pip
$PIP install -r "$PROJECT_DIR/requirements.txt"

$PY "$PROJECT_DIR/scripts/finetune/train.py" \
  --model "$MODEL" \
  --data-root "$PROJECT_DIR/processed_dataset" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH" \
  --grad-accum "$ACC" \
  --output-dir "$PROJECT_DIR/runs/finetune/$(echo "$MODEL" | tr '/' '_')"
