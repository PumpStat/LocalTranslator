#!/usr/bin/env python
"""
QLoRA fine-tuning baseline for translation (ko|en|es).

Minimal dependencies: transformers, accelerate, peft, bitsandbytes, torch.

Data format (JSONL): one object per line with keys:
  - source: "ko"|"en"|"es"
  - target: "ko"|"en"|"es"
  - input:  source text
  - output: target text (ground truth)

Example line:
  {"source":"ko","target":"en","input":"폭타가 꾸준히 나오는 체력곡.",
   "output":"A stamina chart with lots of runs."}

This script constructs a simple chat-format prompt:
  system:  You are a professional translator specialized in arcade rhythm games.
  user:    Translate from <source> to <target>:\n<input>
  assistant: <output>

Only assistant tokens are trained (prompt tokens are masked with -100).

Usage:
  python finetune/qlora_baseline.py \
    --model Qwen/Qwen2.5-14B-Instruct \
    --train_file data/train.jsonl \
    --eval_file data/val.jsonl \
    --output_dir adapters/qwen14b-qlora \
    --per_device_train_batch_size 1 --gradient_accumulation_steps 8 \
    --learning_rate 2e-4 --num_train_epochs 1

To use 4-bit QLoRA on a single GPU (recommended), ensure bitsandbytes is available.
"""

import argparse
import json
import os
from dataclasses import dataclass
from typing import List, Dict, Optional

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


# ---------------------
# Data loading
# ---------------------
class JsonlDataset(Dataset):
    def __init__(self, path: str):
        self.items: List[Dict] = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.items.append(json.loads(line))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


SYSTEM_PROMPT_BASE = (
    "You are a professional translator specialized in arcade rhythm games "
    "(Pump It Up, maimai, DDR). Follow the glossary and rules strictly."
)


def build_messages(example: Dict) -> List[Dict[str, str]]:
    src = example.get("source", "en")
    tgt = example.get("target", "ko")
    inp = example["input"]
    out = example["output"]
    return [
        {"role": "system", "content": SYSTEM_PROMPT_BASE},
        {"role": "user", "content": f"Translate from {src} to {tgt}:\n{inp}"},
        {"role": "assistant", "content": out},
    ]


@dataclass
class DataCollator:
    tokenizer: AutoTokenizer
    def __call__(self, features: List[Dict]):
        # Build chat-formatted examples and mask prompt tokens
        texts = []
        for ex in features:
            messages = build_messages(ex)
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
        batch = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        # Mask everything until the last assistant turn
        # Find assistant start token positions by building again with generation prompt
        labels = batch["input_ids"].clone()
        # Simple heuristic: mask all tokens up to the last occurrence of eos_token_id before assistant content
        # More robust approach is to re-tokenize per-example and slice at assistant boundary.
        # Implement per-example masking using message rebuild.
        for i, ex in enumerate(features):
            messages = build_messages(ex)
            pre = messages[:-1]
            pre_txt = self.tokenizer.apply_chat_template(pre, tokenize=False, add_generation_prompt=False)
            pre_ids = self.tokenizer(pre_txt, return_tensors="pt")["input_ids"][0]
            pre_len = pre_ids.shape[-1]
            labels[i, :pre_len] = -100
        batch["labels"] = labels
        return batch


def create_model_and_tokenizer(model_name: str, use_4bit: bool = True):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    quant_cfg = None
    torch_dtype = None
    if use_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=(torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16),
        )
    else:
        torch_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_cfg,
        torch_dtype=torch_dtype,
        device_map="auto",
    )

    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    # LoRA config (QLoRA defaults)
    lora_cfg = LoraConfig(
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=(
            [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]
        ),
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model, tokenizer


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Base model name or path (e.g., Qwen/Qwen2.5-14B-Instruct)")
    ap.add_argument("--train_file", required=True, help="Path to train jsonl")
    ap.add_argument("--eval_file", help="Path to eval jsonl")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--use_4bit", action="store_true", default=True)
    ap.add_argument("--no_4bit", action="store_true", help="Disable 4bit (use fp16/bf16)")
    # Trainer args (common subset)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=1)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
    ap.add_argument("--learning_rate", type=float, default=2e-4)
    ap.add_argument("--num_train_epochs", type=float, default=1.0)
    ap.add_argument("--max_steps", type=int, default=-1)
    ap.add_argument("--save_steps", type=int, default=200)
    ap.add_argument("--eval_steps", type=int, default=200)
    ap.add_argument("--logging_steps", type=int, default=50)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--lr_scheduler_type", type=str, default="cosine")
    ap.add_argument("--max_seq_len", type=int, default=2048)
    ap.add_argument("--gradient_checkpointing", action="store_true", default=True)
    args = ap.parse_args()
    if args.no_4bit:
        args.use_4bit = False
    return args


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    model, tokenizer = create_model_and_tokenizer(args.model, use_4bit=bool(args.use_4bit))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = JsonlDataset(args.train_file)
    eval_ds = JsonlDataset(args.eval_file) if args.eval_file else None

    collator = DataCollator(tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        evaluation_strategy=("steps" if eval_ds is not None else "no"),
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        bf16=(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        fp16=(torch.cuda.is_available() and not torch.cuda.is_bf16_supported()),
        gradient_checkpointing=args.gradient_checkpointing,
        ddp_find_unused_parameters=False,
        report_to=["none"],
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    # Save adapter + tokenizer
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved adapter and tokenizer to {args.output_dir}")


if __name__ == "__main__":
    main()

