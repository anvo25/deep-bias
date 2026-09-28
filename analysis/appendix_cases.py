"""Appendix case-study figures.

  profile_<id>          one prompt family of Olmo-3-7B-SFT, Direct | Framed
                        (four families)
  <id>__<a>_vs_<b>      two models on one prompt family, one row per model:
                        pretrained vs SFT, SFT vs LoRA-SFT, SFT vs GEPA

    python analysis/appendix_cases.py                 # data from Hugging Face
    python analysis/appendix_cases.py --data ./hf     # a local copy

Writes .pdf and .png of each to --out.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.answer_panels import (  # noqa: E402
    ROW_LABELS, bar_colors, collapse_numwords, compare_figure, render_panel, save,
    shared_axis, top_answers)
from deepbias.data import load_outputs  # noqa: E402

PROFILES = ["1847229", "742711", "305159", "652044"]
COMPARES = [("972328", "olmo3_7b_pretrained", "olmo3_7b_sft"),
            ("1693670", "olmo3_7b_sft", "olmo3_7b_sft_lora"),
            ("777925", "olmo3_7b_sft", "olmo3_7b_sft_gepa")]


def _wspace(*rowsets):
    """Column gap scaled to the longest y-tick label, so a long label does not
    run into the bar-end number of the panel to its left."""
    longest = max((len(l) for rows in rowsets for l, _ in rows), default=10)
    return min(3.2, max(1.2, 0.115 * longest))


def profile(eid: str, r: dict, out: Path) -> None:
    d, f = shared_axis([collapse_numwords(r["direct"]["distribution"]),
                        collapse_numwords(r["framed"]["distribution"])], eid)
    dm, fm = top_answers(r)
    n = len(d)
    fig, axes = plt.subplots(1, 2, figsize=(10.0, max(3.6, 0.48 * n + 1.6)),
                             sharex=True, gridspec_kw={"wspace": _wspace(d, f)})
    render_panel(axes[0], d, bar_colors(d, dm, fm), "Direct", min_slots=n)
    render_panel(axes[1], f, bar_colors(f, dm, fm), "Framed", min_slots=n)
    fig.subplots_adjust(left=0.16, right=0.97, top=0.88, bottom=0.14)
    print(f"profile_{eid}: DR={r['direct']['dr']:.2f} {dm!r}  FR={r['framed']['fr']:.2f} {fm!r}")
    save(fig, out / f"profile_{eid}", dpi=170)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="local dataset copy (default: download from HF)")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "figures")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    models = {m for _, x, y in COMPARES for m in (x, y)}
    data = {m: load_outputs(m, a.data) for m in models}
    for eid in PROFILES:
        profile(eid, data["olmo3_7b_sft"][eid], a.out)
    for eid, ma, mb in COMPARES:
        ra, rb = data[ma][eid], data[mb][eid]
        print(f"{eid}: {ma} {ra['direct']['top_answer']!r}@{ra['direct']['dr']:.2f} -> "
              f"{mb} {rb['direct']['top_answer']!r}@{rb['direct']['dr']:.2f}")
        fig = compare_figure(eid, ra, rb, ROW_LABELS[ma], ROW_LABELS[mb], wspace=_wspace)
        save(fig, a.out / f"{eid}__{ma}_vs_{mb}", dpi=170, pad_inches=0.2)


if __name__ == "__main__":
    main()
