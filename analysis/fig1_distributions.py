"""Figure 1: the four answer-distribution panels of Olmo-3-7B-SFT.

Figure 1 pairs a Deep example ("Choose a random popular butterfly", family
1847229) with a Shallow one ("Choose a random number", family 305159). This
script draws the Direct and Framed panel of each as a separate image, which
the figure then places by hand.

    python analysis/fig1_distributions.py                 # data from Hugging Face
    python analysis/fig1_distributions.py --data ./hf     # a local copy

Writes butterfly_1847229_{direct,framed} and number_305159_{direct,framed}
(.pdf and .png) to --out.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.answer_panels import (  # noqa: E402
    bar_colors, collapse_numwords, render_panel, save, shared_axis, top_answers)
from deepbias.data import load_outputs  # noqa: E402

CASES = [("1847229", "butterfly"), ("305159", "number")]

# Answers that echo a word from the scenario instead of naming a number
# ("give the host one number...") are pooled into "(other)".
FORCE_OTHER = {"305159": {"host", "gift option 2"}}


def render_single(stem: Path, items, colors) -> None:
    fig_h = max(2.4, 0.48 * len(items) + 0.7)
    fig, ax = plt.subplots(1, 1, figsize=(5.6, fig_h))
    render_panel(ax, items, colors, panel_title="")
    fig.subplots_adjust(left=0.30, right=0.95, top=0.97, bottom=0.06)
    save(fig, stem, dpi=200)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="local dataset copy (default: download from HF)")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "figures")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    sft = load_outputs("olmo3_7b_sft", a.data)
    for eid, name in CASES:
        r = sft[eid]
        dm, fm = top_answers(r)
        d, f = shared_axis([collapse_numwords(r["direct"]["distribution"]),
                            collapse_numwords(r["framed"]["distribution"])],
                           eid, force_other=FORCE_OTHER.get(eid, set()))
        print(f"{name} ({eid}): DR={r['direct']['dr']:.2f} top={dm!r}  "
              f"FR={r['framed']['fr']:.2f} framed top={fm!r}  pi={r['pi']:.2f}")
        render_single(a.out / f"{name}_{eid}_direct", d, bar_colors(d, dm, fm))
        render_single(a.out / f"{name}_{eid}_framed", f, bar_colors(f, dm, fm))


if __name__ == "__main__":
    main()
