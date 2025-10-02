#!/usr/bin/env bash
# Upload a fine-tuned checkpoint directory to the Hugging Face Hub.
# Usage examples:
#   bash run_push_to_hub.sh runs/finetune/HuggingFaceTB_SmolVLM2-2.2B-Instruct Mumu02/SmolVLM2-2.2B-Instruct
#   bash run_push_to_hub.sh runs/finetune/HuggingFaceTB_SmolVLM-Instruct Mumu02/SmolVLM-Instruct
#   bash run_push_to_hub.sh runs/finetune/HuggingFaceTB_SmolVLM-500M-Instruct Mumu02/SmolVLM-500M-Instruct
# Optional flags:
#   --no-venv   Use system Python instead of creating/activating .venv
#   --private   Create the Hub repo as private if it doesn't exist
#   --token X   Explicit HF token (else uses cached login)

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
USE_VENV=1
PRIVATE_FLAG=""
TOKEN_ARG=""

if [[ $# -lt 2 ]]; then
  echo "Usage: bash run_push_to_hub.sh <checkpoint_dir> <repo_id> [--no-venv] [--private] [--token <hf_token>]"
  exit 1
fi

CKPT_DIR="$1"; shift
REPO_ID="$1"; shift

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-venv) USE_VENV=0; shift ;;
    --private) PRIVATE_FLAG="--private"; shift ;;
    --token) TOKEN_ARG="--token $2"; shift 2 ;;
    -h|--help)
      echo "Usage: bash run_push_to_hub.sh <checkpoint_dir> <repo_id> [--no-venv] [--private] [--token <hf_token>]"; exit 0 ;;
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

$PY "$PROJECT_DIR/scripts/finetune/push_to_hub.py" "$CKPT_DIR" "$REPO_ID" $PRIVATE_FLAG $TOKEN_ARG

echo "Uploaded $CKPT_DIR to $REPO_ID on Hugging Face Hub."