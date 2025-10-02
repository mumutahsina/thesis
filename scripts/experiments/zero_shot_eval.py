import argparse
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .utils import FIELDS, load_split, evaluate_field_metrics, dump_metrics

# Placeholder zero-shot using OCRed text as pseudo-predictions baseline.
# Replace with actual SmolVLM inference when integrating models.

def build_prompt(ocred_text: str) -> str:
    return (
        "Extract the following fields from the invoice: "
        "invoice_number, invoice_date, client_name, seller_name.\n\n"
        f"OCR: {ocred_text[:2000]}"
    )

def pseudo_extract(ocred_text: str) -> Dict[str, str]:
    # Very naive heuristics as a baseline; to be replaced by model predictions
    out = {f: "" for f in FIELDS}
    text = ocred_text or ""
    # crude patterns
    for token in text.split():
        if token.replace("#", "").replace("-", "").isdigit() and len(token) >= 6:
            out["invoice_number"] = out["invoice_number"] or token.strip(",.;:")
            break
    out["client_name"] = text[:60].split(" ")[:3]
    out["client_name"] = " ".join(out["client_name"]).strip()
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="smolvlm/smolvlm-256m")
    ap.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    ap.add_argument("--max-samples", type=int, default=200)
    ap.add_argument("--data-root", type=str, default=str(Path(__file__).resolve().parents[2] / "processed_dataset"))
    ap.add_argument("--out", type=str, default=str(Path("runs/zero_shot") / "metrics.json"))
    args = ap.parse_args()

    split_dir = Path(args.data_root) / args.split
    df = load_split(split_dir)
    if args.max_samples:
        df = df.head(args.max_samples)

    records: List[Dict] = []
    latencies = []
    for _, row in df.iterrows():
        t0 = time.perf_counter()
        # Replace pseudo_extract with model inference
        pred = pseudo_extract(row.get("ocred_text", ""))
        dt = time.perf_counter() - t0
        latencies.append(dt)
        gold = {f: str(row.get(f, "")) for f in FIELDS}
        records.append({"pred": pred, "gold": gold})

    metrics = evaluate_field_metrics(records)
    metrics["avg_latency_sec"] = sum(latencies) / max(1, len(latencies))
    metrics["model"] = args.model
    metrics["split"] = args.split

    dump_metrics(Path(args.out), metrics)
    print("Zero-shot metrics:", metrics)

if __name__ == "__main__":
    main()
