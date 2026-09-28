"""Merge the continued LoRA-SFT adapter into Olmo-3-7B-SFT so vLLM can serve it.

The adapter repo holds only the LoRA weights (no config.json), so it is
merged into its base model once and saved as a normal checkpoint. This is
the checkpoint evaluated as olmo3_7b_sft_lora in the paper.

Usage:
    python debiasing/lora/merge_adapter.py --adapter <adapter-repo> --out work/models/olmo3_7b_sft_lora
    bash evaluation/run_model.sh olmo3_7b_sft_lora

Merging runs on CPU and needs about 30 GB of RAM and 15 GB of disk.
"""
from __future__ import annotations

import argparse
from pathlib import Path

BASE_MODEL = "tuongvy2603/BITD_baseline"
ADAPTER = None  # the adapter will be released soon, see debiasing/lora/README.md


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=BASE_MODEL)
    ap.add_argument("--adapter", default=ADAPTER, help="Hugging Face repo or local path of the LoRA adapter")
    ap.add_argument("--out", type=Path, default=Path("work/models/olmo3_7b_sft_lora"))
    args = ap.parse_args()
    if not args.adapter:
        ap.error("--adapter is required (the adapter is not released yet, see debiasing/lora/README.md)")

    if args.out.exists() and any(args.out.iterdir()):
        print(f"{args.out} already exists, nothing to do.")
        return
    args.out.mkdir(parents=True, exist_ok=True)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"base    {args.base}\nadapter {args.adapter}\nout     {args.out}", flush=True)
    base = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16, device_map="cpu")
    merged = PeftModel.from_pretrained(base, args.adapter).merge_and_unload()
    merged.save_pretrained(args.out, safe_serialization=True)

    # The adapter repo carries the chat template used in training, so take the
    # tokenizer from there and fall back to the base model's.
    try:
        tok = AutoTokenizer.from_pretrained(args.adapter)
    except Exception:
        tok = AutoTokenizer.from_pretrained(args.base)
    tok.save_pretrained(args.out)
    print(f"done. Evaluate with: bash evaluation/run_model.sh olmo3_7b_sft_lora", flush=True)


if __name__ == "__main__":
    main()
