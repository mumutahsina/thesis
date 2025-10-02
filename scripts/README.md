Preprocessing scripts

Files:
- preprocess_invoices.ipynb: Cell-based notebook script to prepare the dataset. It scans datasets/*.csv and associated image folders, then writes processed_dataset/ with train/val/test splits. Use any IDE that supports #%% cells (PyCharm, VSCode) to run cells sequentially.
- preprocess_invoices.py: Same logic as a Python script with #%% cells; you can run as a standard Python module.

How to run
Option A) Using the helper bash script (recommended):
   bash run_preprocessing.sh
   # or without virtualenv
   bash run_preprocessing.sh --no-venv

Option B) Manual steps:
1) Create a Python environment and install dependencies:
   pip install -r requirements.txt

2) Run from project root either of the following:
   - Using Python script:
       python scripts/preprocess_invoices.py
   - Using notebook-like file:
       Open scripts/preprocess_invoices.ipynb in PyCharm/VSCode and run cells in order.

Outputs
- processed_dataset/
  - train/
    - images/
    - labels.csv
  - val/
    - images/
    - labels.csv
  - test/
    - images/
    - labels.csv

Notes
- The labels.csv includes: filename, ocred_text, json_raw, invoice_number, invoice_date, client_name, seller_name.
- The script only includes records whose image files exist.
- Splits are 80/10/10 with a fixed random seed (42).

Experiments (thesis)
- See scripts/experiments/ for zero-shot baseline, LoRA fine-tuning skeleton, and evaluation utilities aligned to the research questions.
