"""Maintainer script: export the research outputs into the public dataset layout.

Not needed to use or reproduce anything. It documents exactly how the files on
https://huggingface.co/datasets/anvo25/deep-bias were produced from the
internal pipeline outputs, so the provenance of every released number is
inspectable.

What it does
  * copies the 4,442 prompt families, their 133,260 framings, and every
    evaluated model's clustered outputs into a flat, documented schema
  * converts the cluster judge's surface_forms dict into a list, because a
    dict keyed by arbitrary answer strings cannot be stored as an Arrow column
  * redacts credential-shaped strings. The prompt families keep the verbatim
    Dolci-Instruct-SFT rows they were built from (the `evidence` field), and a
    few of those upstream rows contain API tokens that users pasted into their
    prompts. They are replaced with [REDACTED_TOKEN]. No result depends on them.

Usage
  python scripts/export_release_data.py --src /path/to/research/repo --out ./hf_release
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepbias.schema import raw_direct, raw_framed, release_record  # noqa: E402

SECRET = re.compile(
    r"hf_[A-Za-z0-9]{30,}|sk-ant-[A-Za-z0-9_-]{20,}|sk-[A-Za-z0-9_-]{20,}"
    r"|AIza[0-9A-Za-z_-]{30,}|ghp_[A-Za-z0-9]{30,}|AKIA[0-9A-Z]{16}"
)

# release config name -> (internal output dir, checkpoint or API model, note)
MODELS = {
    "olmo3_7b_pretrained": ("olmo3_pretrained", "allenai/Olmo-3-1025-7B", "queried with a Q:/A: completion prompt"),
    "olmo3_7b_sft":        ("olmo3_sft", "tuongvy2603/Olmo-3-7B-Instruct-SFT-replicate", "our re-SFT of Olmo-3-1025-7B on Dolci-Instruct-SFT only"),
    "olmo3_7b_dpo":        ("olmo3_dpo", "allenai/Olmo-3-7B-Instruct-DPO", "official checkpoint"),
    "olmo3_7b_rlvr":       ("olmo3_rlvr", "allenai/Olmo-3-7B-Instruct", "official checkpoint"),
    "tulu3_8b_sft":        ("tulu3_sft", "allenai/Llama-3.1-Tulu-3-8B-SFT", "official checkpoint"),
    "claude_sonnet_5":     ("claude_sonnet_5", "anthropic/claude-sonnet-5", "via OpenRouter, reasoning disabled"),
    "gpt_5_6_sol":         ("gpt56sol", "gpt-5.6-sol", "via the OpenAI API, reasoning disabled"),
    "olmo3_7b_sft_gepa":   ("bitd_gepa", "tuongvy2603/Olmo-3-7B-Instruct-SFT-replicate", "with the GEPA-optimized system prompt"),
    "olmo3_7b_sft_lora":   ("olmo3_sft_lora_data_new", "tuongvy2603/Olmo-3-7B-Instruct-SFT-replicate + diversity LoRA adapter", "continued LoRA-SFT adapter merged"),
}


def redact(o, hits):
    if isinstance(o, str):
        new, n = SECRET.subn("[REDACTED_TOKEN]", o)
        hits[0] += n
        return new
    if isinstance(o, list):
        return [redact(v, hits) for v in o]
    if isinstance(o, dict):
        return {k: redact(v, hits) for k, v in o.items()}
    return o


def jsonl(path):
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write(path: Path, rows, gz=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    op = gzip.open if gz else open
    n = 0
    with op(path, "wt") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    src, out = a.src, a.out
    hits = [0]

    pool = {}
    for r in jsonl(src / "src/dolci_bias_probes/checkpoints/full/step4/step4_deduped_v2.jsonl"):
        pool[str(r["row_idx"])] = r["rewrite"]
        yield_row = {"id": str(r["row_idx"]), "prompt": r["rewrite"], **{k: v for k, v in r.items() if k not in ("row_idx", "rewrite")}}
        pool.setdefault("_rows", []).append(redact(yield_row, hits))
    n = write(out / "prompts/train.jsonl", pool.pop("_rows"))
    print(f"prompts   {n:>7,} rows  (redacted {hits[0]} credential-shaped strings)")

    fr = ({"id": r["entry_id"], "framing_idx": r["framing_idx"], "prompt": r["base_question"],
           "framing_topic": r["framing_topic"], "framing": r["framing"]}
          for r in jsonl(src / "outputs/deep_bias/models/olmo3_sft/eval/framings_deduped.jsonl"))
    print(f"framings  {write(out / 'framings/train.jsonl', fr):>7,} rows")

    for cfg, (d, ckpt, note) in MODELS.items():
        ev = src / f"outputs/deep_bias/models/{d}/eval"
        rows = [release_record(r, pool.get(r["entry_id"]), ckpt)
                for r in jsonl(ev / "responses_clustered_deduped.jsonl")]
        n = write(out / f"outputs/{cfg}.jsonl", rows)
        rd = write(out / f"raw/{cfg}/direct.jsonl.gz",
                   (raw_direct(x) for x in jsonl(ev / "responses_raw.jsonl") if x["entry_id"] in pool), gz=True)
        rf = write(out / f"raw/{cfg}/framed.jsonl.gz",
                   (raw_framed(x) for x in jsonl(ev / "framing_responses_raw.jsonl") if x["entry_id"] in pool), gz=True)
        print(f"{cfg:20s} {n:>6,} families | raw {rd:>7,} direct + {rf:>7,} framed")


if __name__ == "__main__":
    main()
