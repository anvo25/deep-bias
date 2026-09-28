"""Load the released data, from a local copy or straight from Hugging Face.

Every analysis script accepts --data. Point it at a local clone of the
dataset repo to work offline, or leave it unset to download each file on
first use from https://huggingface.co/datasets/anvo25/deep-bias (cached by
huggingface_hub afterwards).
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

HF_REPO = "anvo25/deep-bias"

MODELS = [
    "olmo3_7b_pretrained", "olmo3_7b_sft", "olmo3_7b_dpo", "olmo3_7b_rlvr",
    "tulu3_8b_sft", "claude_sonnet_5", "gpt_5_6_sol",
    "olmo3_7b_sft_gepa", "olmo3_7b_sft_lora",
]


def _path(rel: str, data_dir: str | None) -> Path:
    if data_dir:
        return Path(data_dir) / rel
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(HF_REPO, rel, repo_type="dataset"))


def _jsonl(p: Path):
    op = gzip.open if p.suffix == ".gz" else open
    with op(p, "rt") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_prompts(data_dir=None) -> list[dict]:
    return list(_jsonl(_path("prompts/train.jsonl", data_dir)))


def load_framings(data_dir=None) -> list[dict]:
    return list(_jsonl(_path("framings/train.jsonl", data_dir)))


def load_outputs(model: str, data_dir=None) -> dict[str, dict]:
    """One model's analysis-ready outputs, keyed by prompt-family id.

    With a local data_dir, any name under outputs/ works, including models
    you evaluated yourself with evaluation/finalize.py."""
    if data_dir is None and model not in MODELS:
        raise ValueError(f"unknown model {model!r}, expected one of {MODELS}")
    return {r["id"]: r for r in _jsonl(_path(f"outputs/{model}.jsonl", data_dir))}


def load_raw(model: str, condition: str, data_dir=None) -> list[dict]:
    """Raw sampled responses. condition is 'direct' or 'framed'."""
    return list(_jsonl(_path(f"raw/{model}/{condition}.jsonl.gz", data_dir)))
