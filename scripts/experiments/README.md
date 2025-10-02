SmolVLM Thesis Experiments

This directory contains scripts and configuration to run experiments for the thesis research questions:

Research Questions
1. Can compact multimodal LLMs (SmolVLM) effectively process structured financial documents?
2. How does fine-tuning improve performance on invoice extraction compared to zero-shot?
3. What trade-offs exist between accuracy and efficiency when using smaller vs. larger SmolVLM variants?

Methodology Overview
- Dataset: processed_dataset/ created by scripts/preprocess_invoices.py
- Models: SmolVLM-256M, SmolVLM-1B, SmolVLM-2B (via Hugging Face names)
- Training: LoRA fine-tuning for extraction tasks
- Evaluation: exact match and token-level F1 on fields; latency and memory

Quickstart
1) Ensure processed_dataset exists with train/val/test.
2) Install requirements: pip install -r requirements.txt
3) Run baseline zero-shot evaluation (pseudo):
   python scripts/experiments/zero_shot_eval.py --model smolvlm/smolvlm-256m --max-samples 100
4) Run HF zero-shot using a public VLM (uses transformers pipeline):
   python scripts/experiments/hf_zero_shot_infer.py --model Salesforce/blip2-flan-t5-xl --split val --max-samples 25
   # Alternative models (adjust task flag if needed):
   #   microsoft/Florence-2-base (task may be different)
5) Run LoRA fine-tuning (skeleton):
   python scripts/experiments/train_lora.py --model smolvlm/smolvlm-256m --epochs 1 --output-dir runs/smolvlm-256m-lora
6) Fine-tune with HF VLM + LoRA (new):
   python scripts/finetune/train.py --model Salesforce/blip2-flan-t5-base --data-root processed_dataset --epochs 1 --output-dir runs/finetune/blip2-lora
7) Evaluate fine-tuned model:
   python scripts/experiments/eval_extraction.py --model runs/smolvlm-256m-lora --split test

Outputs
- runs/<exp-name>/ with checkpoints and metrics.json
- reports/ summarizing metrics and efficiency trade-offs

Note: These scripts are light-weight scaffolding to be adapted if the specific SmolVLM packages differ. Replace model names with correct HF repos if needed.
