import argparse
import json
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, AutoTokenizer, AutoModelForVision2Seq, AutoModelForSeq2SeqLM, TrainingArguments, Trainer

from .dataset import InvoiceExtractionDataset


class SimpleVLMCollator:
    def __init__(self, processor=None, tokenizer=None, max_target_len: int = 128):
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_target_len = max_target_len

    def __call__(self, batch):
        # Determine fields from the first batch element (dataset provides 'fields')
        fields = batch[0].get("fields", [])
        # Build text targets as compact JSON strings
        targets = []
        for b in batch:
            t = {f: b["target"].get(f, "") for f in fields}
            targets.append(json.dumps(t, ensure_ascii=False))
        prompts = [b["prompt"] for b in batch]
        images = [b["image"] for b in batch]

        inputs = {}
        if self.processor is not None:
            proc = self.processor(text=prompts, images=images, return_tensors="pt", padding=True)
            inputs.update(proc)
        elif self.tokenizer is not None:
            inputs.update(self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True))
        else:
            raise ValueError("Either processor or tokenizer must be provided")

        # Choose tokenizer for labels
        tok = self.tokenizer
        if tok is None and self.processor is not None and hasattr(self.processor, "tokenizer"):
            tok = self.processor.tokenizer
        if tok is None:
            raise ValueError("No tokenizer available for encoding labels")

        labels = tok(
            targets,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_target_len,
        )["input_ids"]

        inputs["labels"] = labels
        return inputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="Salesforce/blip2-flan-t5-base")
    ap.add_argument("--data-root", type=str, default=str(Path(__file__).resolve().parents[2] / "processed_dataset"))
    ap.add_argument("--output-dir", type=str, default="runs/finetune/blip2-lora")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=4)
    args = ap.parse_args()

    data_root = Path(args.data_root)
    train_ds = InvoiceExtractionDataset(data_root / "train")
    val_ds = InvoiceExtractionDataset(data_root / "val")

    # Try to load a vision-language model processor, else fall back to text model
    processor = None
    tokenizer = None
    model = None
    try:
        model = AutoModelForVision2Seq.from_pretrained(args.model)
        processor = AutoProcessor.from_pretrained(args.model)
        tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else AutoTokenizer.from_pretrained(args.model)
    except Exception:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
        tokenizer = AutoTokenizer.from_pretrained(args.model)

    lora_cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, target_modules=["q", "v", "k", "o"])  # may need adjustments per model
    model = get_peft_model(model, lora_cfg)

    collator = SimpleVLMCollator(processor=processor, tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        fp16=torch.cuda.is_available(),
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    if tokenizer is not None:
        tokenizer.save_pretrained(args.output_dir)
    if processor is not None:
        processor.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
