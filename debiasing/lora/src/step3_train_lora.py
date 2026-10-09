"""Continued LoRA-SFT of our replicated Olmo-3-7B-SFT on the Deep + Shallow diversity data.

Starts from `tuongvy2603/Olmo-3-7B-Instruct-SFT-replicate` and trains one LoRA adapter on
data/train_messages.jsonl (built by step2_convert_to_messages.py).

Dataset format (Dolci-Instruct-SFT style):
    {"id", "messages": [{"role","content","function_calls","functions"}, ...],
     "source_dataset", "domain"}

The final assistant turn becomes `completion`; everything before it is `prompt`.
`completion_only_loss=True` masks the prompt tokens from the loss.

Usage:
    python debiasing/lora/src/step3_train_lora.py
    python debiasing/lora/src/step3_train_lora.py --wandb_mode disabled

    # Multi-GPU
    accelerate launch --num_processes 4 debiasing/lora/src/step3_train_lora.py
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer

DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "train_messages.jsonl"


def load_messages_jsonl(path: str) -> Dataset:
    rows = []
    with open(path) as f:
        for line in f:
            ex = json.loads(line)
            messages = ex["messages"]
            if not messages or messages[-1]["role"] != "assistant":
                raise ValueError(f"row {ex.get('id')} does not end with assistant turn")
            rows.append({
                "prompt": messages[:-1],
                "completion": [messages[-1]],
            })
    return Dataset.from_list(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", default="tuongvy2603/Olmo-3-7B-Instruct-SFT-replicate")
    p.add_argument("--data_path", default=str(DEFAULT_DATA_PATH))
    p.add_argument("--output_dir", default="work/lora/olmo3_7b_sft_lora_adapter")
    p.add_argument("--num_train_epochs", type=float, default=5.0)
    p.add_argument("--per_device_train_batch_size", type=int, default=8)
    p.add_argument("--gradient_accumulation_steps", type=int, default=2)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--lr_scheduler_type", default="cosine",
                   choices=["cosine", "linear", "constant",
                            "constant_with_warmup", "cosine_with_min_lr"])
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--logging_steps", type=int, default=5)
    p.add_argument("--save_steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    # LoRA
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_target_modules", nargs="+",
                   default=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"])
    # W&B
    p.add_argument("--wandb_project", default="bitd-continue-sft")
    p.add_argument("--wandb_mode", default="online",
                   choices=["online", "offline", "disabled"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"lora-{args.model_id.split('/')[-1]}-debias"

    print(f"[config] model={args.model_id}  lr={args.learning_rate}")
    print(f"[config] epochs={args.num_train_epochs}  bs={args.per_device_train_batch_size}"
          f"  grad_accum={args.gradient_accumulation_steps}  max_len={args.max_length}")
    print(f"[config] data={args.data_path}")
    print(f"[config] output_dir={output_dir}")

    os.environ["WANDB_PROJECT"] = args.wandb_project
    os.environ["WANDB_MODE"] = args.wandb_mode
    with open(output_dir / "run_config.json", "w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    train_dataset = load_messages_jsonl(args.data_path)
    print(f"[data ] {len(train_dataset)} examples loaded")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    sft_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_length=args.max_length,
        bf16=True,
        completion_only_loss=True,
        packing=False,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        report_to="none" if args.wandb_mode == "disabled" else "wandb",
        run_name=run_name,
        seed=args.seed,
        model_init_kwargs={"dtype": torch.bfloat16},
        dataset_num_proc=1,
    )

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    trainer = SFTTrainer(
        model=args.model_id,
        args=sft_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"[done ] adapter saved to {output_dir}")


if __name__ == "__main__":
    main()
