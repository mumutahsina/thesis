Fine-tuning scripts

This folder contains a minimal, practical setup to fine-tune a Hugging Face vision-language model with LoRA on the processed_dataset produced by scripts/preprocess_invoices.py.

Files
- dataset.py: PyTorch Dataset for processed_dataset splits, building prompts from OCR and loading images.
- train.py: Training entry-point using transformers + peft. It attempts to load a VLM (AutoModelForVision2Seq + AutoProcessor) and falls back to a text seq2seq model if needed. Targets are compact JSON strings with fields: invoice_number, invoice_date, client_name, seller_name.

Quickstart
1) Ensure you have processed_dataset with train/val/test:
   python scripts/preprocess_invoices.py

2) Install requirements:
   pip install -r requirements.txt

3) Run fine-tuning (example with BLIP-2):
   python scripts/finetune/train.py \
     --model Salesforce/blip2-flan-t5-base \
     --data-root processed_dataset \
     --epochs 1 \
     --batch-size 1 \
     --grad-accum 4 \
     --output-dir runs/finetune/blip2-lora

Notes
- Adjust --model to other HF VLMs if desired. For SmolVLM models, replace with the correct HF IDs.
- LoRA target_modules are generic placeholders ("q","k","v","o"); specific models may require different module names.
- If CUDA is available, fp16 is enabled automatically.
- The dataset uses labels.csv from each split and pairs images in images/ with OCR and target fields.

Convenience runner
- To fine-tune the three SmolVLM Instruct variants via bash helper:
  - Large (2.2B): bash run_smolvlm_finetune.sh 2.2b
  - Base (1B):    bash run_smolvlm_finetune.sh 1b
  - Small (500M): bash run_smolvlm_finetune.sh 500m
  Add flags like --epochs, --batch-size, --grad-accum, --lr as needed.

Upload to Hugging Face Hub
- After training completes, upload the checkpoint using the helper:
  bash run_push_to_hub.sh runs/finetune/HuggingFaceTB_SmolVLM2-2.2B-Instruct Mumu02/SmolVLM2-2.2B-Instruct
  bash run_push_to_hub.sh runs/finetune/HuggingFaceTB_SmolVLM-Instruct    Mumu02/SmolVLM-Instruct
  bash run_push_to_hub.sh runs/finetune/HuggingFaceTB_SmolVLM-500M-Instruct Mumu02/SmolVLM-500M-Instruct
- Ensure you are logged in: huggingface-cli login (or pass --token <HF_TOKEN> to the script).
