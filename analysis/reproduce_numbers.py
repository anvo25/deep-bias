"""Recompute every headline number in the paper from the released outputs.

No GPU, no API keys, about a minute on a laptop. Each number is printed next
to the value reported in the paper, and the script exits non-zero if any of
them fails to reproduce.

    python analysis/reproduce_numbers.py                 # data from Hugging Face
    python analysis/reproduce_numbers.py --data ./hf     # a local copy

Families whose model produced no valid answer at all under direct prompting
have no top answer, so they are excluded from every percentage, exactly as in
the paper. This only affects Claude Sonnet 5: 2 families have no record and 2
have an empty top answer, so 4,438 of 4,442 are used.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepbias.data import load_outputs  # noqa: E402
from deepbias.match import surface_match  # noqa: E402
from deepbias.metrics import DR_THRESHOLD  # noqa: E402

FAILS: list[str] = []


def valid(rows: dict) -> list[dict]:
    return [r for r in rows.values() if (r["direct"]["top_answer"] or "").strip()]


def pct(rows, kind) -> float:
    return 100 * sum(r["bias_type"] == kind for r in rows) / len(rows)


def check(label: str, got: float, paper: float, tol: float = 0.05) -> None:
    ok = abs(got - paper) <= tol
    if not ok:
        FAILS.append(label)
    print(f"  {'PASS' if ok else 'FAIL'}  {label:46s} computed {got:7.2f}   paper {paper:7.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="local dataset copy (default: download from HF)")
    data = ap.parse_args().data
    out = {m: valid(load_outputs(m, data)) for m in (
        "olmo3_7b_pretrained", "olmo3_7b_sft", "olmo3_7b_dpo", "olmo3_7b_rlvr",
        "tulu3_8b_sft", "claude_sonnet_5", "gpt_5_6_sol",
        "olmo3_7b_sft_gepa", "olmo3_7b_sft_lora")}

    print("\nPrevalence of Deep and Shallow bias")
    paper = {"olmo3_7b_sft": (10.8, 25.0), "tulu3_8b_sft": (14.7, 28.6),
             "claude_sonnet_5": (9.9, 67.8), "gpt_5_6_sol": (17.4, 37.1)}
    deeps, shals = [], []
    for m, (pd, ps) in paper.items():
        d, s = pct(out[m], "deep"), pct(out[m], "shallow")
        deeps.append(d); shals.append(s)
        check(f"{m} Deep %", d, pd)
        check(f"{m} Shallow %", s, ps)
    check("mean Deep % over the four models", float(np.mean(deeps)), 13.2)
    check("mean Shallow % over the four models", float(np.mean(shals)), 39.6)

    print("\nBias across the Olmo-3-7B training stages")
    for m, (pd, ps) in {"olmo3_7b_pretrained": (9.8, 8.0), "olmo3_7b_sft": (10.8, 25.0),
                        "olmo3_7b_dpo": (10.0, 36.2), "olmo3_7b_rlvr": (12.6, 38.9)}.items():
        check(f"{m} Deep %", pct(out[m], "deep"), pd)
        check(f"{m} Shallow %", pct(out[m], "shallow"), ps)

    print("\nDeeper SFT biases more often match the pretrained answer")
    pre = {r["id"]: r["direct"]["top_answer"] for r in out["olmo3_7b_pretrained"]}
    pis, shared = [], []
    for r in out["olmo3_7b_sft"]:
        if r["id"] not in pre or r["direct"]["dr"] <= DR_THRESHOLD:
            continue
        pis.append(r["pi"])
        shared.append(surface_match(r["direct"]["top_answer"], pre[r["id"]]))
    pis, shared = np.array(pis), np.array(shared, dtype=float)
    # Figure 4's bins are half-open, [0, 0.1) ... [0.8, 1.0], matching the
    # original np.digitize(..., right=False) binning. pi is a product of two
    # multiples of 1/30, so values of exactly 0.1 and 0.8 do occur. The two
    # end bins checked here are stable under the 6-decimal rounding of the
    # released pi. The inner bins are not, so fig4_pretrained_vs_sft.py
    # rebuilds exact rates from the sample counts.
    lo, hi = pis < 0.1, pis >= 0.8
    check("same-top-answer %, lowest bin  [0, 0.1)", 100 * shared[lo].mean(), 20.7)
    check("same-top-answer %, highest bin [0.8, 1]", 100 * shared[hi].mean(), 62.3)
    check("Pearson r(pi, same top answer)", float(np.corrcoef(pis, shared)[0, 1]), 0.27, tol=0.005)

    print("\nDebiasing (independent classification)")
    for m, trip in {"olmo3_7b_sft": (10.8, 25.0, 64.2), "olmo3_7b_sft_gepa": (9.2, 19.2, 71.6),
                    "olmo3_7b_sft_lora": (6.3, 14.7, 79.0)}.items():
        for kind, p in zip(("deep", "shallow", "non_bias"), trip):
            check(f"{m} {kind} %", pct(out[m], kind), p)

    print(f"\n{'All numbers reproduced.' if not FAILS else f'{len(FAILS)} number(s) did NOT reproduce: {FAILS}'}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
