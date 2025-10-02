import json
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd

FIELDS = ["invoice_number", "invoice_date", "client_name", "seller_name"]

def load_split(split_dir: Path) -> pd.DataFrame:
    labels = pd.read_csv(split_dir / "labels.csv")
    labels["image_path"] = labels["filename"].apply(lambda x: str(split_dir / "images" / x))
    return labels

def exact_match(pred: str, gold: str) -> float:
    return float((pred or "").strip() == (gold or "").strip())

def token_f1(pred: str, gold: str) -> float:
    p = set((pred or "").split())
    g = set((gold or "").split())
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    tp = len(p & g)
    prec = tp / len(p) if p else 0.0
    rec = tp / len(g) if g else 0.0
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)

def evaluate_field_metrics(records: List[Dict]) -> Dict:
    metrics = {f: {"exact": [], "f1": []} for f in FIELDS}
    for r in records:
        for f in FIELDS:
            metrics[f]["exact"].append(exact_match(r["pred"].get(f, ""), r["gold"].get(f, "")))
            metrics[f]["f1"].append(token_f1(r["pred"].get(f, ""), r["gold"].get(f, "")))
    summary = {}
    for f in FIELDS:
        summary[f"{f}_exact"] = sum(metrics[f]["exact"]) / max(1, len(metrics[f]["exact"]))
        summary[f"{f}_f1"] = sum(metrics[f]["f1"]) / max(1, len(metrics[f]["f1"]))
    return summary

def dump_metrics(path: Path, payload: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
