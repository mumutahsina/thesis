import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .utils import FIELDS, load_split, evaluate_field_metrics, dump_metrics

# This evaluates a (fine-tuned) model. For now, it's a placeholder that echoes pseudo predictions.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True, help="HF model id or local path to fine-tuned checkpoint")
    ap.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    ap.add_argument("--data-root", type=str, default=str(Path(__file__).resolve().parents[2] / "processed_dataset"))
    ap.add_argument("--out", type=str, default=str(Path("runs/eval") / "metrics.json"))
    args = ap.parse_args()

    split_dir = Path(args.data_root) / args.split
    df = load_split(split_dir)

    # TODO: Replace with actual model inference
    def predict(row) -> Dict[str, str]:
        return {f: "" for f in FIELDS}

    records: List[Dict] = []
    for _, row in df.iterrows():
        pred = predict(row)
        gold = {f: str(row.get(f, "")) for f in FIELDS}
        records.append({"pred": pred, "gold": gold})

    metrics = evaluate_field_metrics(records)
    metrics["model"] = args.model
    metrics["split"] = args.split

    dump_metrics(Path(args.out), metrics)
    print("Eval metrics:", metrics)


if __name__ == "__main__":
    main()
