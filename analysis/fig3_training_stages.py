"""Figure 3: Deep and Shallow bias across the Olmo-3-7B training stages.

One stacked bar per checkpoint (Pretrained, SFT, DPO, RLVR), Deep at the
bottom and Shallow on top, with the total bias proportion marked above.
Families with no valid direct answer are excluded, as in
analysis/reproduce_numbers.py.

    python analysis/fig3_training_stages.py                 # data from Hugging Face
    python analysis/fig3_training_stages.py --data ./hf     # a local copy

Writes bias_vbars_olmo3_pipeline_deepbias.pdf and .png to --out.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepbias.data import load_outputs  # noqa: E402

DEEP_C    = "#1b5e20"
SHALLOW_C = "#c08a17"
TOTAL_C   = "#000000"   # line, markers, bracket and total labels
SPINE_C   = "#b0b0b0"
GRID_C    = "#dcdcdc"

STAGES = [
    ("Pretrained", "olmo3_7b_pretrained"),
    ("SFT",        "olmo3_7b_sft"),
    ("DPO",        "olmo3_7b_dpo"),
    ("RLVR",       "olmo3_7b_rlvr"),
]


def shares(rows: dict) -> tuple[float, float, int]:
    """Deep %, Shallow % and n over families with a valid direct answer."""
    valid = [r for r in rows.values() if (r["direct"]["top_answer"] or "").strip()]
    n = len(valid)
    n_d = sum(r["bias_type"] == "deep" for r in valid)
    n_s = sum(r["bias_type"] == "shallow" for r in valid)
    return 100 * n_d / n, 100 * n_s / n, n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="local dataset copy (default: download from HF)")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "figures")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "axes.edgecolor": SPINE_C,
        "axes.linewidth": 1.2,
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })

    stages, deep, shal = [], [], []
    for label, model in STAGES:
        d, s, n = shares(load_outputs(model, a.data))
        stages.append(label)
        deep.append(d); shal.append(s)
        print(f"{label:12s}  n={n:5,d}  Deep={d:5.1f}%  Shallow={s:5.1f}%  Total={d+s:5.1f}%")

    x = list(range(len(stages)))
    fig, ax = plt.subplots(figsize=(9.0, 6.5))

    bar_w = 0.62
    ax.bar(x, deep, width=bar_w, color=DEEP_C,
           edgecolor="white", linewidth=1.2, label="Deep bias",
           zorder=3)
    ax.bar(x, shal, width=bar_w, bottom=deep, color=SHALLOW_C,
           edgecolor="white", linewidth=1.2, label="Shallow bias",
           zorder=3)

    for xi, d, s in zip(x, deep, shal):
        if d > 4:
            ax.text(xi, d / 2, f"{d:.1f}%", ha="center", va="center",
                    fontsize=18, fontweight="bold", color="white", zorder=4)
        if s > 4:
            ax.text(xi, d + s / 2, f"{s:.1f}%", ha="center", va="center",
                    fontsize=18, fontweight="bold", color="white", zorder=4)

    totals = [d + s for d, s in zip(deep, shal)]
    ax.plot(x, totals, color=TOTAL_C, lw=2.8,
            marker="o", ms=11, markerfacecolor=TOTAL_C,
            markeredgecolor="white", markeredgewidth=1.6,
            label="Total bias proportion = (Deep + Shallow)",
            zorder=6)

    for xi, t in zip(x, totals):
        bracket_x = xi - bar_w / 2 - 0.06
        tick_len  = 0.07
        ax.plot([bracket_x, bracket_x], [0, t],
                color=TOTAL_C, lw=2.4, zorder=5)
        ax.plot([bracket_x, bracket_x + tick_len], [t, t],
                color=TOTAL_C, lw=2.4, zorder=5)
        ax.plot([bracket_x, bracket_x + tick_len], [0, 0],
                color=TOTAL_C, lw=2.4, zorder=5)
        ax.text(xi, t + 2.2, f"{t:.1f}%",
                ha="center", va="bottom",
                fontsize=20, fontweight="bold", color=TOTAL_C, zorder=7)

    ax.set_xticks(x)
    ax.set_xticklabels(stages, fontsize=23, fontweight="bold")
    ax.set_ylim(0, 70)
    ax.set_ylabel("Bias proportion", fontsize=21)
    ax.tick_params(axis="y", labelsize=18)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v)}%"))
    ax.grid(axis="y", color=GRID_C, lw=1.0, zorder=1)
    ax.set_axisbelow(True)

    leg = ax.legend(loc="upper right", fontsize=18, frameon=True,
                    framealpha=0.95, edgecolor=SPINE_C, ncol=1)
    leg.get_frame().set_linewidth(0.8)

    fig.tight_layout()
    stem = a.out / "bias_vbars_olmo3_pipeline_deepbias"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {stem}.pdf / .png")


if __name__ == "__main__":
    main()
