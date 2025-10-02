#!/usr/bin/env bash
# A helper script to install dependencies and run the preprocessing.
# Usage:
#   bash run_preprocessing.sh [--no-venv]
# By default, it will create a .venv virtual environment in the project root.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
USE_VENV=1

for arg in "$@"; do
  case "$arg" in
    --no-venv)
      USE_VENV=0
      shift
      ;;
    -h|--help)
      echo "Usage: bash run_preprocessing.sh [--no-venv]"
      exit 0
      ;;
    *)
      echo "Unknown argument: $arg"
      echo "Usage: bash run_preprocessing.sh [--no-venv]"
      exit 1
      ;;
  esac

done

# Determine python executable
if command -v python3 >/dev/null 2>&1; then
  PYTHON=python3
elif command -v python >/dev/null 2>&1; then
  PYTHON=python
else
  echo "Python is not installed or not in PATH. Please install Python 3.9+" >&2
  exit 1
fi

if [ "$USE_VENV" -eq 1 ]; then
  # Create venv if it doesn't exist
  if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at $VENV_DIR"
    "$PYTHON" -m venv "$VENV_DIR"
  fi
  # Activate venv
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  PIP=pip
else
  PIP=pip
fi

# Upgrade pip and install requirements
$PYTHON -m pip install --upgrade pip
$PIP install -r "$PROJECT_DIR/requirements.txt"

# Run the preprocessing script
$PYTHON "$PROJECT_DIR/scripts/preprocess_invoices.py"

# Example: run a quick zero-shot baseline on the val split (optional)
# $PYTHON "$PROJECT_DIR/scripts/experiments/zero_shot_eval.py" --model smolvlm/smolvlm-256m --split val --max-samples 50 --out "$PROJECT_DIR/runs/zero_shot/metrics.json"
# Example: run Hugging Face zero-shot with a public VLM (optional, heavy):
# $PYTHON "$PROJECT_DIR/scripts/experiments/hf_zero_shot_infer.py" --model Salesforce/blip2-flan-t5-xl --split val --max-samples 10 --out "$PROJECT_DIR/runs/hf_zero_shot/metrics.json"
