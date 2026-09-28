"""Figure 2: share of Deep, Shallow and Non-bias prompt families per model.

One horizontal stacked bar each for Olmo-3-7B-SFT, Tulu-3-8B-SFT, Claude
Sonnet 5 and GPT-5.6 Sol. Families with no valid direct answer are excluded,
as in analysis/reproduce_numbers.py.

    python analysis/fig2_bias_types.py                 # data from Hugging Face
    python analysis/fig2_bias_types.py --data ./hf     # a local copy

Writes bias_type_bars_sft_deepbias.pdf and .png to --out.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepbias.data import load_outputs  # noqa: E402

PANEL_BG = "#fafafa"
SPINE_C  = "#b0b0b0"
DEEP_C    = "#1b5e20"
SHALLOW_C = "#c08a17"
NONBIAS_C = "#cfcfcf"

TICK_FS    = 48
PCT_FS     = 47

MODELS = [
    ("Olmo-3-7B SFT", "olmo3_7b_sft"),
    ("Tulu-3-8B SFT", "tulu3_8b_sft"),
    ("Sonnet 5", "claude_sonnet_5"),
    ("GPT-5.6 Sol", "gpt_5_6_sol"),
]


def shares(rows: dict) -> tuple[int, float, float]:
    """n, Deep % and Shallow % over families with a valid direct answer."""
    valid = [r for r in rows.values() if (r["direct"]["top_answer"] or "").strip()]
    n = len(valid)
    deep = 100 * sum(r["bias_type"] == "deep" for r in valid) / n
    shal = 100 * sum(r["bias_type"] == "shallow" for r in valid) / n
    return n, deep, shal


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

    series: list[dict] = []
    for label, model in MODELS:
        n, deep_pct, shal_pct = shares(load_outputs(model, a.data))
        bias_pct = deep_pct + shal_pct
        series.append({"label": label, "n": n, "deep": deep_pct,
                       "shallow": shal_pct, "bias": bias_pct})
        print(f"{label:26s}  n={n:5,d}  Deep={deep_pct:5.1f}%  "
              f"Shallow={shal_pct:5.1f}%  Bias={bias_pct:5.1f}%")

    WITHIN  = 1.80
    raw_y: list[float] = [i * WITHIN for i in range(len(series))]
    y_top = max(raw_y)
    y = np.array([y_top - p for p in raw_y])

    # Height scales with the number of bars so each bar keeps the same
    # thickness and spacing.
    fig_h = (WITHIN * (len(series) - 1) + 1.3) * (7.5 / 3.1)
    fig, ax = plt.subplots(figsize=(20.0, fig_h))
    ax.set_facecolor(PANEL_BG)
    bar_h = 0.92

    BRACKET_FS = 49
    LEGEND_FS  = 37

    deep_handle = shal_handle = non_handle = None
    pct_texts: list[tuple] = []  # (Text, segment_width_in_data_units)
    for yi, s in zip(y, series):
        non = max(0.0, 100.0 - s["bias"])
        h_d = ax.barh(yi, s["deep"], height=bar_h, color=DEEP_C,
                      edgecolor="white", linewidth=1.2, zorder=3)
        h_s = ax.barh(yi, s["shallow"], left=s["deep"], height=bar_h,
                      color=SHALLOW_C, edgecolor="white", linewidth=1.2,
                      zorder=3)
        h_n = ax.barh(yi, non, left=s["bias"], height=bar_h, color=NONBIAS_C,
                      edgecolor="white", linewidth=1.2, zorder=3)
        deep_handle = deep_handle or h_d
        shal_handle = shal_handle or h_s
        non_handle  = non_handle  or h_n
        t_d = ax.text(s["deep"] / 2, yi, f"{s['deep']:.1f}%",
                ha="center", va="center", fontsize=PCT_FS,
                fontweight="bold", color="white", zorder=4)
        t_s = ax.text(s["deep"] + s["shallow"] / 2, yi, f"{s['shallow']:.1f}%",
                ha="center", va="center", fontsize=PCT_FS,
                fontweight="bold", color="white", zorder=4)
        t_n = ax.text(s["bias"] + non / 2, yi, f"{non:.1f}%",
                ha="center", va="center", fontsize=PCT_FS,
                fontweight="bold", color="#444", zorder=4)
        pct_texts += [(t_d, s["deep"]), (t_s, s["shallow"]), (t_n, non)]

        bracket_top  = yi + bar_h / 2 + 0.04
        bracket_tick = bracket_top - 0.025
        ax.plot([0, s["bias"]], [bracket_top, bracket_top],
                color="#222", lw=2.2, zorder=5)
        ax.plot([0, 0], [bracket_top, bracket_tick],
                color="#222", lw=2.2, zorder=5)
        ax.plot([s["bias"], s["bias"]], [bracket_top, bracket_tick],
                color="#222", lw=2.2, zorder=5)
        ax.text(0, bracket_top + 0.03,
                f"total bias proportion = {s['bias']:.1f}%",
                ha="left", va="bottom",
                fontsize=BRACKET_FS, fontweight="bold", color="#222",
                zorder=6)

    ax.set_yticks(y)
    ax.set_yticklabels([s["label"] for s in series], fontsize=TICK_FS, fontweight="bold")
    ax.tick_params(axis="y", pad=12)
    ax.set_ylim(min(y) - 0.65, max(y) + 0.65)
    ax.set_xlim(0, 100.5)
    ax.set_xticks([])
    ax.set_xlabel("")
    ax.spines["bottom"].set_visible(False)
    ax.set_axisbelow(True)

    mean_deep    = sum(s["deep"]    for s in series) / len(series)
    mean_shallow = sum(s["shallow"] for s in series) / len(series)
    mean_non     = sum(100.0 - s["bias"] for s in series) / len(series)
    leg = ax.legend(handles=[deep_handle, shal_handle, non_handle],
                    labels=[f"Deep bias\n(mean: {mean_deep:.1f}%)",
                            f"Shallow bias\n(mean: {mean_shallow:.1f}%)",
                            f"Non-bias\n(mean: {mean_non:.1f}%)"],
                    loc="lower right", bbox_to_anchor=(1.0, 1.08),
                    ncol=3, fontsize=LEGEND_FS, frameon=True, framealpha=0.95,
                    edgecolor=SPINE_C, handlelength=2.2, handleheight=1.6,
                    borderpad=0.8, columnspacing=2.0)
    leg.get_frame().set_linewidth(0.8)
    fig.tight_layout(pad=1.0)

    # A percentage label wider than its segment (e.g. a Deep share near 10%)
    # would overflow into the neighbouring segment or the tick label. After
    # layout, measure each label against its segment and shrink only the
    # ones that do not fit.
    MIN_PCT_FS = 22
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for txt, seg_width in pct_texts:
        if seg_width <= 0:
            continue
        text_w_px = txt.get_window_extent(renderer=renderer).width
        x0_px = ax.transData.transform((0, 0))[0]
        x1_px = ax.transData.transform((seg_width, 0))[0]
        seg_w_px = x1_px - x0_px
        if text_w_px > seg_w_px * 0.90:
            new_fs = max(MIN_PCT_FS, txt.get_fontsize() * (seg_w_px * 0.90) / text_w_px)
            txt.set_fontsize(new_fs)

    stem = a.out / "bias_type_bars_sft_deepbias"
    fig.savefig(stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"),          bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {stem}.pdf / .png")


if __name__ == "__main__":
    main()
