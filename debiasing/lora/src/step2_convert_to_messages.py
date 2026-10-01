"""Merge the Deep and Shallow response files into one training set in the
Dolci-Instruct-SFT messages format.

Input rows (from step1_gen_responses_{deep,shallow}.py):
    {"row_idx", "entity_id", "prompt", "assistant_response"}

Output row schema:
    {
        "id":             str,
        "messages":       [{"role","content","function_calls","functions"}, ...],
        "source_dataset": str,
        "domain":         str,
    }

Usage:
    python debiasing/lora/src/step2_convert_to_messages.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_INPUTS = [
    DATA_DIR / "data_deep_responses.jsonl",
    DATA_DIR / "data_shallow_responses.jsonl",
]
DEFAULT_OUTPUT = DATA_DIR / "train_messages.jsonl"
SOURCE = "bitd_debias_lora"


def domain_from_entity(entity_id: str) -> str:
    # entity_id looks like "<row_idx>_<cohort>" (e.g. "2040523_deep");
    # the cohort/domain is the last "_"-separated token.
    return entity_id.rsplit("_", 1)[-1]


def turn(role: str, content: str) -> dict:
    return {"content": content, "function_calls": None, "functions": None, "role": role}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path, nargs="+", default=DEFAULT_INPUTS)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--source-dataset", default=SOURCE)
    args = ap.parse_args()

    # entity_id repeats (30 copies per topic, e.g. "2040523_deep" x30);
    # append a 1-based occurrence counter so ids stay unique.
    seen_counts: dict[str, int] = {}
    n = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fout:
        for src in args.inputs:
            with src.open(encoding="utf-8") as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    ex = json.loads(line)
                    seen_counts[ex["entity_id"]] = seen_counts.get(ex["entity_id"], 0) + 1
                    row = {
                        "id": f"{args.source_dataset}_{ex['entity_id']}_{seen_counts[ex['entity_id']]}",
                        "messages": [
                            turn("user", ex["prompt"]),
                            turn("assistant", ex["assistant_response"]),
                        ],
                        "source_dataset": args.source_dataset,
                        "domain": domain_from_entity(ex["entity_id"]),
                    }
                    fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                    n += 1
            print(f"  read {src.name}")
    print(f"wrote {n} rows -> {args.output}")


if __name__ == "__main__":
    main()
