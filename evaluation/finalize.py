"""Step 4 of evaluation: write a model's results in the released layout.

Reads the cluster output (cluster.py) and the raw samples, and writes

    <work>/outputs/<name>.jsonl            one record per prompt family (DR, FR, pi, bias type)
    <work>/raw/<name>/direct.jsonl.gz      raw direct samples
    <work>/raw/<name>/framed.jsonl.gz      raw framed samples

This is the same layout as the Hugging Face dataset, so every script in
analysis/ runs on your new model with --data <work> --models <name>.

Usage:
    python evaluation/finalize.py --name my_model --model org/my-model \\
        --prompts work/prompts/train.jsonl \\
        --clustered work/clustered/my_model.jsonl \\
        --direct work/samples/my_model/direct.jsonl \\
        --framed work/samples/my_model/framed.jsonl \\
        --work work
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepbias.schema import raw_direct, raw_framed, release_record  # noqa: E402


def jsonl(p: Path):
    op = gzip.open if p.suffix == ".gz" else open
    with op(p, "rt") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write(p: Path, rows, gz=False) -> int:
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with (gzip.open if gz else open)(p, "wt") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="Short name, used for file names")
    ap.add_argument("--model", required=True, help="Checkpoint or API model id, stored in each record")
    ap.add_argument("--prompts", type=Path, required=True, help="Prompt families (released prompts/train.jsonl)")
    ap.add_argument("--clustered", type=Path, required=True)
    ap.add_argument("--direct", type=Path, required=True)
    ap.add_argument("--framed", type=Path, required=True)
    ap.add_argument("--work", type=Path, default=Path("work"))
    a = ap.parse_args()

    prompts = {}
    for r in jsonl(a.prompts):
        prompts[str(r.get("id", r.get("row_idx")))] = r.get("prompt") or r.get("rewrite")

    rows = [release_record(r, prompts[r["entry_id"]], a.model)
            for r in jsonl(a.clustered) if r["entry_id"] in prompts]
    missing = len(prompts) - len(rows)
    n = write(a.work / "outputs" / f"{a.name}.jsonl", rows)
    rd = write(a.work / "raw" / a.name / "direct.jsonl.gz",
               (raw_direct(x) for x in jsonl(a.direct) if x["entry_id"] in prompts and not x.get("error")), gz=True)
    rf = write(a.work / "raw" / a.name / "framed.jsonl.gz",
               (raw_framed(x) for x in jsonl(a.framed) if x["entry_id"] in prompts and not x.get("error")), gz=True)

    c = Counter(r["bias_type"] for r in rows)
    print(f"[finalize] {a.name}: {n:,} families, {rd:,} direct + {rf:,} framed samples")
    if missing:
        print(f"[finalize] note: {missing:,} prompt families have no results "
              f"(expected with --limit, otherwise check the sampling logs for errors)")
    for k in ("deep", "shallow", "non_bias"):
        print(f"  {k:9s} {100 * c[k] / max(n, 1):5.1f}%")


if __name__ == "__main__":
    main()
