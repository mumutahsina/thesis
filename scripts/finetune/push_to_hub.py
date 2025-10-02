import argparse
from pathlib import Path

from huggingface_hub import HfApi, HfFolder, create_repo, upload_folder


def ensure_token(token: str | None) -> str:
    if token:
        return token
    # Try to read from cache (~/.huggingface/token)
    tok = HfFolder.get_token()
    if not tok:
        raise RuntimeError("No Hugging Face token found. Pass --token or login with `huggingface-cli login`.")
    return tok


def main():
    ap = argparse.ArgumentParser(description="Upload a fine-tuned checkpoint folder to Hugging Face Hub.")
    ap.add_argument("checkpoint_dir", type=str, help="Local path to the saved model folder (e.g., runs/finetune/HuggingFaceTB_SmolVLM2-2.2B-Instruct)")
    ap.add_argument("repo_id", type=str, help="Target Hub repo id, e.g., Mumu02/SmolVLM2-2.2B-Instruct")
    ap.add_argument("--private", action="store_true", help="Create the repo as private if it does not exist")
    ap.add_argument("--token", type=str, default=None, help="Hugging Face access token (else use cached login)")
    args = ap.parse_args()

    ckpt = Path(args.checkpoint_dir)
    if not ckpt.exists() or not ckpt.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt}")

    token = ensure_token(args.token)
    api = HfApi(token=token)

    # Create repo if it doesn't exist
    try:
        create_repo(repo_id=args.repo_id, token=token, private=args.private, exist_ok=True)
    except Exception as e:
        print(f"Warning: create_repo failed or already exists: {e}")

    # Upload entire folder (including adapter weights, tokenizer, processor, etc.)
    print(f"Uploading {ckpt} to {args.repo_id} ...")
    upload_folder(
        folder_path=str(ckpt),
        repo_id=args.repo_id,
        token=token,
        repo_type="model",
        commit_message="Upload fine-tuned checkpoint",
    )
    print("Upload completed.")


if __name__ == "__main__":
    main()
