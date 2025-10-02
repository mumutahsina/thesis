#%% md
# Invoice Dataset Preprocessing

# This script loads invoice images (JPG) and their corresponding CSV annotations,
# then builds input-label pairs and saves a processed dataset with train/val/test splits.
#
# Assumptions:
# - Each CSV in datasets/ (e.g., batch1_1.csv) contains columns: File Name, Json Data, OCRed Text.
# - The corresponding images live under folders like datasets/batch1_1/ with filenames matching File Name.
# - We create a supervised dataset: input = image file; labels saved in labels.csv with both OCR text and selected JSON fields.
#
# Outputs:
# - A directory processed_dataset/ with sub-folders: train/, val/, test/.
# - For each split: images/ folder with copied images and labels.csv mapping filename -> label fields.
#
# You can adapt the label schema as needed for your task (e.g., extract invoice_number from JSON).

#%%
import os
import json
import random
import shutil
from pathlib import Path

import pandas as pd

# Detect project root: this file lives in /scripts/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = PROJECT_ROOT / 'datasets'
PROCESSED_DIR = PROJECT_ROOT / 'processed_dataset'
PROCESSED_DIR.mkdir(exist_ok=True)

print('Project root:', PROJECT_ROOT)
print('Datasets dir:', DATASETS_DIR)
print('Processed dir:', PROCESSED_DIR)

#%%
# Discover CSV files that describe batches
csv_paths = sorted(DATASETS_DIR.glob('*.csv'))
if not csv_paths:
    print('No CSV files found in', DATASETS_DIR)
else:
    print('Found CSV files:')
    for p in csv_paths:
        print(' -', p.name)

#%%
from typing import List, Dict

def load_batch(csv_path: Path) -> List[Dict]:
    df = pd.read_csv(csv_path)
    # Normalize expected column names
    col_map = {c.lower().strip(): c for c in df.columns}

    def get_col(name: str) -> str:
        target = name.replace(' ', '').lower()
        for key, orig in col_map.items():
            if key.replace(' ', '') == target:
                return orig
        raise KeyError(f'Missing expected column: {name} in {csv_path}')

    fname_col = get_col('File Name')
    json_col = get_col('Json Data')
    ocr_col = get_col('OCRed Text')

    # Identify image folder by csv stem (e.g., batch1_1.csv -> datasets/batch1_1/)
    images_dir = csv_path.with_suffix('')
    if not images_dir.exists():
        alt = csv_path.parent / csv_path.stem
        if alt.exists():
            images_dir = alt
    if not images_dir.exists():
        print(f'Warning: images dir not found for {csv_path}, tried {images_dir}')

    records: List[Dict] = []
    for _, row in df.iterrows():
        fname = str(row[fname_col]).strip()
        img_path = images_dir / fname
        json_raw = row[json_col]
        ocred_text = row[ocr_col] if pd.notna(row[ocr_col]) else ''

        # Keep JSON as raw string; also try to parse for sanity
        json_data = None
        if isinstance(json_raw, str):
            try:
                json_data = json.loads(json_raw)
            except Exception:
                json_data = None

        rec: Dict = {
            'filename': fname,
            'image_path': str(img_path),
            'ocred_text': str(ocred_text) if not pd.isna(ocred_text) else '',
            'json_raw': json_raw if isinstance(json_raw, str) else json.dumps(json_raw)
        }
        if json_data and isinstance(json_data, dict):
            inv = json_data.get('invoice', {})
            rec['invoice_number'] = inv.get('invoice_number', '')
            rec['invoice_date'] = inv.get('invoice_date', '')
            rec['client_name'] = inv.get('client_name', '')
            rec['seller_name'] = inv.get('seller_name', '')
        else:
            rec['invoice_number'] = ''
            rec['invoice_date'] = ''
            rec['client_name'] = ''
            rec['seller_name'] = ''
        records.append(rec)
    return records

all_records: List[Dict] = []
for csv_path in csv_paths:
    print('Loading', csv_path.name)
    batch_recs = load_batch(csv_path)
    all_records.extend(batch_recs)

print('Total loaded records:', len(all_records))

#%%
# Filter only those that have existing images
valid_records = [r for r in all_records if Path(r['image_path']).exists()]
missing = len(all_records) - len(valid_records)
print(f'Total records: {len(all_records)}, valid with images: {len(valid_records)}, missing images: {missing}')

#%%
# Shuffle and split
random.seed(42)
sz = len(valid_records)
if sz == 0:
    print('No valid records with existing images found. Exiting early.')
else:
    random.shuffle(valid_records)
    n_train = int(0.8 * sz)
    n_val = int(0.1 * sz)
    n_test = sz - n_train - n_val
    splits = {
        'train': valid_records[:n_train],
        'val': valid_records[n_train:n_train+n_val],
        'test': valid_records[n_train+n_val:]
    }
    print({k: len(v) for k, v in splits.items()})

    #%%
    def write_split(name: str, recs):
        split_dir = PROCESSED_DIR / name
        img_out = split_dir / 'images'
        split_dir.mkdir(parents=True, exist_ok=True)
        img_out.mkdir(parents=True, exist_ok=True)

        rows = []
        for r in recs:
            src = Path(r['image_path'])
            dst = img_out / src.name
            try:
                if not dst.exists():
                    shutil.copy2(src, dst)
            except Exception as e:
                print(f'Copy failed for {src} -> {dst}: {e}')
                continue
            rows.append({
                'filename': src.name,
                'ocred_text': r.get('ocred_text', ''),
                'json_raw': r.get('json_raw', ''),
                'invoice_number': r.get('invoice_number',''),
                'invoice_date': r.get('invoice_date',''),
                'client_name': r.get('client_name',''),
                'seller_name': r.get('seller_name','')
            })
        df = pd.DataFrame(rows)
        df.to_csv(split_dir / 'labels.csv', index=False)
        print(f'Wrote {len(rows)} records to {split_dir}')

    for name, recs in splits.items():
        write_split(name, recs)

    print('Done.')
