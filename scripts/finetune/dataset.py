from pathlib import Path
from typing import Dict, List

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

# Columns present in labels.csv that are inputs/metadata rather than output labels
NON_LABEL_COLUMNS = {"filename", "image_path", "ocred_text", "json_raw"}


def build_prompt(ocr_text: str, fields: List[str]) -> str:
    fields_csv = ", ".join(fields)
    return (
        "Extract the following fields from the invoice image as JSON with keys: "
        f"{fields_csv}.\n\n"
        f"OCR Text:\n{(ocr_text or '')[:2000]}\n\n"
        "Return only a compact JSON object."
    )


class InvoiceExtractionDataset(Dataset):
    def __init__(self, split_dir: Path):
        self.split_dir = Path(split_dir)
        df = pd.read_csv(self.split_dir / "labels.csv")
        # Determine label fields dynamically from CSV columns
        self.fields: List[str] = [
            c for c in df.columns
            if c not in NON_LABEL_COLUMNS
        ]
        # Maintain CSV order; if none found, fall back to previous default fields
        if not self.fields:
            self.fields = ["invoice_number", "invoice_date", "client_name", "seller_name"]
        # compute image paths
        df["image_path"] = df["filename"].apply(lambda x: str(self.split_dir / "images" / x))
        self.records: List[Dict] = df.to_dict(orient="records")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict:
        r = self.records[idx]
        image = Image.open(r["image_path"]).convert("RGB")
        prompt = build_prompt(str(r.get("ocred_text", "")), self.fields)
        target = {f: str(r.get(f, "")) for f in self.fields}
        return {
            "image": image,
            "prompt": prompt,
            "target": target,
            "fields": self.fields,
        }
