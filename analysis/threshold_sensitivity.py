"""The Deep/Shallow split is stable across the choice of threshold.

The paper's cutoffs DR > 0.65 and pi > 0.40 are one decision, "a clear
majority" t, applied twice: Deep means DR > t and FR > t, so pi > t^2, and
0.65^2 = 0.4225 rounds to 0.40. This sweeps t from 0.50 to 0.86 and plots
the mean Deep and Shallow share over the four post-SFT models of Figure 2.
Shallow exceeds Deep at every t. Families with no valid direct answer count
as Non-bias here.

    python analysis/threshold_sensitivity.py                 # data from Hugging Face
    python analysis/threshold_sensitivity.py --data ./hf     # a local copy

Writes threshold_sensitivity.pdf and .png to --out.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepbias.data import load_outputs  # noqa: E402

T_PAPER = 0.65
MODELS = ["olmo3_7b_sft", "tulu3_8b_sft", "claude_sonnet_5", "gpt_5_6_sol"]
AXIS_FS, TICK_FS, LEG_FS = 14, 12, 11
PANEL_BG, SPINE_C, GRID_C = "#fafafa", "#b0b0b0", "#dcdcdc"
DEEP_C, SHAL_C = "#1b5e20", "#c08a17"


def _rate(x: float, n: int) -> float:
    """The exact rate k/n behind a released rate rounded to 6 decimals, so
    comparisons against t and t^2 at the boundary match the float computation
    used for the paper."""
    return round(x * n) / n if n else x


def load(rows: dict) -> tuple[list[tuple[float, float]], int]:
    """(DR, pi) per family with a valid direct answer, and the family count."""
    out = []
    for r in rows.values():
        d, f = r["direct"], r["framed"]
        if not (d["top_answer"] or "").strip():
            continue
        dr = _rate(d["dr"], d["n_samples"])
        out.append((dr, dr * _rate(f["fr"], f["n_samples"])))
    return out, len(rows)


def split(rows, n_all, t):
    """Deep := DR>t and FR>t (i.e. pi>t^2). Shallow := DR>t but not Deep."""
    pi_star = t * t
    deep = sum(1 for dr, pi in rows if pi > pi_star)
    shal = sum(1 for dr, pi in rows if dr > t and pi <= pi_star)
    return 100 * deep / n_all, 100 * shal / n_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="local dataset copy (default: download from HF)")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "figures")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({"font.family": "sans-serif"})
    data = {m: load(load_outputs(m, a.data)) for m in MODELS}
    ts = np.round(np.arange(0.50, 0.861, 0.01), 2)
    deep = [float(np.mean([split(*data[m], t)[0] for m in MODELS])) for t in ts]
    shal = [float(np.mean([split(*data[m], t)[1] for m in MODELS])) for t in ts]

    fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.0))
    ax.set_facecolor(PANEL_BG)
    for s in ax.spines.values():
        s.set_color(SPINE_C)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(True, color=GRID_C, lw=0.8); ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=TICK_FS)

    ax.fill_between(ts, deep, shal, color=SHAL_C, alpha=0.13, zorder=1)
    ax.plot(ts, shal, color=SHAL_C, lw=2.6, label="Shallow", zorder=3)
    ax.plot(ts, deep, color=DEEP_C, lw=2.6, label="Deep", zorder=3)

    i = int(np.argmin(np.abs(ts - T_PAPER)))
    ax.axvline(T_PAPER, color="#222", ls="--", lw=1.7, zorder=4)
    for y, c in ((shal[i], SHAL_C), (deep[i], DEEP_C)):
        ax.plot([T_PAPER], [y], "o", ms=7, color=c, mec="white", mew=1.4, zorder=5)
        ax.annotate(f"{y:.1f}%", (T_PAPER, y), textcoords="offset points",
                    xytext=(9, 6), fontsize=LEG_FS, fontweight="bold", color=c, zorder=6)
    ax.text(T_PAPER - 0.006, 47.5, "paper: $t\\!=\\!0.65$\n$(\\pi^\\ast\\!=\\!0.65^2\\!\\approx\\!0.40)$",
            fontsize=LEG_FS, fontweight="bold", color="#222", ha="right", va="top", zorder=6)
    ax.text(0.755, 42.0, "Shallow exceeds Deep\nat every threshold",
            fontsize=LEG_FS, fontweight="bold", color=SHAL_C, ha="center", zorder=6)

    ax.set_xlabel("Majority threshold $t$   (Deep: $DR>t$ and $FR>t$)", fontsize=AXIS_FS)
    ax.set_ylabel("Mean share of prompt families (%)", fontsize=AXIS_FS)
    ax.set_xlim(0.50, 0.86); ax.set_ylim(0, 52)
    ax.legend(fontsize=LEG_FS, frameon=True, edgecolor=SPINE_C, loc="lower left")
    fig.tight_layout(pad=0.9)
    stem = a.out / "threshold_sensitivity"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=190, bbox_inches="tight")
    plt.close(fig)

    print("   t   pi*=t^2  mean Deep %  mean Shallow %")
    for t in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80):
        j = int(np.argmin(np.abs(ts - t)))
        print(f"  {t:.2f}   {t*t:.3f}       {deep[j]:5.1f}          {shal[j]:5.1f}"
              f"{'   (paper)' if abs(t - T_PAPER) < 1e-9 else ''}")
    print("Shallow > Deep at every t:", all(s > d for s, d in zip(shal, deep)))
    print(f"wrote {stem}.pdf / .png")


if __name__ == "__main__":
    main()
