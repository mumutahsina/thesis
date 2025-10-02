import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd
from PIL import Image
from transformers import pipeline

from .utils import FIELDS, load_split, evaluate_field_metrics, dump_metrics


PROMPT_TEMPLATE = (
    "You are an information extraction assistant. Given an invoice image and its OCR text, "
    "extract the following fields as a JSON object with keys: invoice_number, invoice_date, client_name, seller_name.\n\n"
    "OCR Text:\n{ocr}\n\n"
    "Return only a compact JSON object with these four keys and string values."
)


def infer_with_hf(model_id: str, image_path: str, ocr_text: str, task: str = "image-text-to-text") -> Dict[str, str]:
    # Try generic VLM pipeline; fall back to text-only if image models are unavailable
    try:
        vlm = pipeline(task, model=model_id)
        prompt = PROMPT_TEMPLATE.format(ocr=ocr_text[:2000])
        out = vlm(image_path, prompt, max_new_tokens=128)
        text = out[0]["generated_text"] if isinstance(out, list) else str(out)
    except Exception:
        # Fallback: text-only LLM summarization using text-generation
        llm = pipeline("text-generation", model=model_id)
        prompt = PROMPT_TEMPLATE.format(ocr=ocr_text[:2000])
        out = llm(prompt, max_new_tokens=128)
        text = out[0]["generated_text"] if isinstance(out, list) else str(out)

    # Try to parse JSON from the output
    pred = {f: "" for f in FIELDS}
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            j = json.loads(text[start:end+1])
            for f in FIELDS:
                if f in j and isinstance(j[f], str):
                    pred[f] = j[f]
    except Exception:
        pass
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="Salesforce/blip2-flan-t5-xl", help="HF model id")
    ap.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    ap.add_argument("--max-samples", type=int, default=25)
    ap.add_argument("--data-root", type=str, default=str(Path(__file__).resolve().parents[2] / "processed_dataset"))
    ap.add_argument("--out", type=str, default=str(Path("runs/hf_zero_shot") / "metrics.json"))
    ap.add_argument("--task", type=str, default="image-text-to-text", help="transformers pipeline task")
    args = ap.parse_args()

    split_dir = Path(args.data_root) / args.split
    df = load_split(split_dir)
    if args.max_samples:
        df = df.head(args.max_samples)

    records: List[Dict] = []
    latencies = []
    for _, row in df.iterrows():
        img_path = row["image_path"]
        ocr = str(row.get("ocred_text", ""))
        t0 = time.perf_counter()
        pred = infer_with_hf(args.model, img_path, ocr, task=args.task)
        dt = time.perf_counter() - t0
        latencies.append(dt)
        gold = {f: str(row.get(f, "")) for f in FIELDS}
        records.append({"pred": pred, "gold": gold})

    metrics = evaluate_field_metrics(records)
    metrics["avg_latency_sec"] = sum(latencies) / max(1, len(latencies))
    metrics["model"] = args.model
    metrics["split"] = args.split

    dump_metrics(Path(args.out), metrics)
    print("HF zero-shot metrics:", metrics)


if __name__ == "__main__":
    main()
