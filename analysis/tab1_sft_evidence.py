"""Table 1: search Dolci-Instruct-SFT for training rows that explain a bias.

For every Deep or Shallow prompt family of Olmo-3-7B-SFT (1,590 families),
look for a Dolci-Instruct-SFT row where
  * the user turn contains "random", is at most 300 characters, and contains
    every key word of the probe prompt (e.g. "butterfly"), and
  * the assistant turn is 1 to 80 characters and contains the SFT model's
    top direct answer.
The paper reports 5 matches, all verified by hand. They are 4 distinct
Dolci rows, since two big-cat families match the same row.

Needs the Dolci rows written by dataset/step1_extract.py (keep responses,
which is the default):

    python dataset/step1_extract.py
    python analysis/tab1_sft_evidence.py --rows work/dataset/step1/step1_rows.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepbias.data import load_outputs  # noqa: E402

STOP = {"choose", "a", "random", "of", "the", "type", "kind"}
WORD = re.compile(r"[a-z0-9]+")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", type=Path, default=Path("work/dataset/step1/step1_rows.jsonl"))
    ap.add_argument("--data", default=None, help="local dataset copy (default: download from HF)")
    ap.add_argument("--model", default="olmo3_7b_sft")
    a = ap.parse_args()

    # 1) short Dolci rows that ask for something random
    pool = []
    with a.rows.open() as f:
        for line in f:
            if "andom" not in line:          # cheap prefilter before parsing
                continue
            r = json.loads(line)
            u, resp = r["user_prompt"], r.get("assistant_response")
            if resp is None:
                sys.exit("these rows have no assistant_response; rerun step1 without --drop-responses")
            ul = u.lower()
            if "random" in ul and len(u) <= 300 and 1 <= len(resp) <= 80:
                pool.append((set(WORD.findall(ul)), resp.lower(), r))
    print(f"Dolci rows with 'random', user <= 300 chars, response 1-80 chars: {len(pool):,}")

    # 2) biased families and their probe key words
    fams = [r for r in load_outputs(a.model, a.data).values() if r["bias_type"] in ("deep", "shallow")]
    print(f"Deep + Shallow families for {a.model}: {len(fams):,}\n")

    hits = []
    for fam in fams:
        ans = (fam["direct"]["top_answer"] or "").lower().strip(".\"' ")
        if len(ans) < 2:
            continue
        words = set(WORD.findall(fam["prompt"].lower())) - STOP
        for wordset, resp, row in pool:
            if words <= wordset and ans in resp:
                hits.append((fam, row))
                break

    for fam, row in hits:
        print(f"[{fam['bias_type']:7s} pi={fam['pi']:.2f}] {fam['prompt']}  ->  SFT answer: {fam['direct']['top_answer']}")
        print(f"   Dolci row {row['row_idx']} ({row.get('source_dataset', '')})")
        print(f"   user:      {row['user_prompt']!r}")
        print(f"   assistant: {row['assistant_response']!r}\n")
    print(f"{len(hits)} of {len(fams):,} biased families have a matching Dolci row "
          f"({len({row['row_idx'] for _, row in hits})} distinct rows)")


if __name__ == "__main__":
    main()
