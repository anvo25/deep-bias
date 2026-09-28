"""Figure 4: deeper SFT biases more often keep the pretrained model's answer.

For every biased Olmo-3-7B-SFT prompt family (DR > 0.65), check whether the
SFT model's direct top answer matches the pretrained model's direct top
answer (surface_match). Families are binned by the SFT bias depth score
pi = DR x FR, and the figure plots the share of matches per bin. Bins are
[0, 0.1), [0.1, 0.2), ..., and a bin with fewer than 30 families is merged
into the bin to its left, so the last bin is [0.8, 1.0].

    python analysis/fig4_pretrained_vs_sft.py                 # data from Hugging Face
    python analysis/fig4_pretrained_vs_sft.py --data ./hf     # a local copy

Writes pi_pretrained_vs_sft_same_top_deepbias_olmo.pdf and .png to --out.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepbias.data import load_outputs  # noqa: E402
from deepbias.match import surface_match  # noqa: E402
from deepbias.metrics import DR_THRESHOLD, PI_THRESHOLD  # noqa: E402


def _rate(x: float, n: int) -> float:
    """The exact rate k/n behind a released rate rounded to 6 decimals.
    pi values such as 2/3 x 0.6 sit on a bin edge, and only the unrounded
    product falls on the same side of the edge as in the paper."""
    return round(x * n) / n if n else x


def collect(pre_rows: dict, sft_rows: dict) -> tuple[np.ndarray, np.ndarray]:
    """(pi, shared) arrays over biased SFT families (DR > 0.65) whose
    pretrained model gave a valid direct answer."""
    pre_top = {}
    for r in pre_rows.values():
        top = (r["direct"]["top_answer"] or "").strip()
        if top:
            pre_top[r["id"]] = top

    pi_list, sh_list = [], []
    for r in sft_rows.values():
        d, f = r["direct"], r["framed"]
        dr = _rate(d["dr"], d["n_samples"])
        if r["id"] not in pre_top or dr <= DR_THRESHOLD:
            continue
        pi_list.append(dr * _rate(f["fr"], f["n_samples"]))
        sft_top = (d["top_answer"] or "").strip()
        sh_list.append(bool(sft_top and surface_match(sft_top, pre_top[r["id"]])))
    print(f"    SFT families: {len(sft_rows):,}   biased: {len(pi_list):,}   "
          f"({len(pi_list) / max(len(sft_rows), 1) * 100:.1f}%)")
    return np.array(pi_list), np.array(sh_list, dtype=bool)


def bin_and_merge(pi, sh, n_bins=10, small_n=30):
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    qs[-1] = qs[-1] + 1e-9
    idx = np.digitize(pi, qs[1:-1], right=False)
    rows = []
    for b in range(n_bins):
        mask = idx == b
        nb = int(mask.sum())
        if nb == 0:
            rows.append({"lo": qs[b], "hi": qs[b+1], "n": 0, "sh": 0,
                         "rate": float("nan"), "pi_mean": float("nan")})
            continue
        rows.append({"lo": qs[b], "hi": qs[b+1], "n": nb,
                     "sh": int(sh[mask].sum()),
                     "rate": float(sh[mask].mean()),
                     "pi_mean": float(pi[mask].mean())})
    i = len(rows) - 1
    while i > 0:
        cur, prev = rows[i], rows[i-1]
        if cur["n"] < small_n:
            n_new  = prev["n"] + cur["n"]
            sh_new = prev["sh"] + cur["sh"]
            if n_new > 0:
                mu_p = prev["pi_mean"] if prev["n"] > 0 else 0.0
                mu_c = cur["pi_mean"]  if cur["n"]  > 0 else 0.0
                pi_n = (mu_p * prev["n"] + mu_c * cur["n"]) / n_new
                rate = sh_new / n_new
            else:
                pi_n, rate = float("nan"), float("nan")
            rows[i-1] = {"lo": prev["lo"], "hi": cur["hi"], "n": n_new,
                         "sh": sh_new, "rate": rate, "pi_mean": pi_n}
            del rows[i]
        i -= 1
    return [r for r in rows if r["n"] > 0]


# Sized for one column of a two-column paper, slightly wider than the column
# so that \includegraphics[width=\columnwidth] scales it down.
AXIS_FS  = 15
TICK_FS  = 12
NTOP_FS  = 9
LEG_FS   = 10
PCT_FS   = 8

PANEL_BG = "#fafafa"
SPINE_C  = "#b0b0b0"
GRID_C   = "#dcdcdc"

LINE_COLOR = "#0072B2"
REGION_LABEL_Y = 74   # just under the ylim=80 ceiling


def draw_panel(ax, rows, t_star, mean_shallow, mean_deep):
    color = LINE_COLOR
    x = np.array([r["pi_mean"] for r in rows])
    y_pct = np.array([r["rate"] for r in rows]) * 100
    n = np.array([r["n"]       for r in rows])
    sizes = 40 + 8 * np.sqrt(n.astype(float))

    ax.set_facecolor(PANEL_BG)
    for sp in ax.spines.values():
        sp.set_color(SPINE_C)

    ax.axvspan(-0.02, t_star, color="#c08a17", alpha=0.10, zorder=0)
    ax.axvspan(t_star, 1.02,   color="#1b5e20", alpha=0.10, zorder=0)
    ax.axvline(t_star, color="black", ls="--", lw=1.2, alpha=0.7, zorder=1,
               label=f"Deep/Shallow boundary ($\\pi$ = {t_star:.2f})")
    ax.text(t_star / 2, REGION_LABEL_Y, "Shallow", ha="center", va="center",
            fontsize=AXIS_FS - 4, fontweight="bold", color="#c08a17", zorder=2)
    ax.text((t_star + 1.0) / 2, REGION_LABEL_Y, "Deep", ha="center", va="center",
            fontsize=AXIS_FS - 4, fontweight="bold", color="#1b5e20", zorder=2)

    if not np.isnan(mean_shallow):
        ax.plot([-0.02, t_star], [mean_shallow * 100] * 2,
                color="#c08a17", ls=":", lw=1.6, alpha=0.85, zorder=2,
                label=f"Shallow mean = {mean_shallow*100:.1f}%")
    if not np.isnan(mean_deep):
        ax.plot([t_star, 1.02], [mean_deep * 100] * 2,
                color="#1b5e20", ls=":", lw=1.6, alpha=0.85, zorder=2,
                label=f"Deep mean = {mean_deep*100:.1f}%")

    ax.plot(x, y_pct, "--", color=color, lw=1.8, alpha=0.9, zorder=2)
    ax.scatter(x, y_pct, s=sizes, color=color, edgecolor="white",
               linewidth=1.2, zorder=3, marker="o",
               label="circle size = number of prompts")

    # Labels alternate above and below the markers so that neighbouring bins
    # with similar values do not collide. The white box keeps the blue text
    # readable on the green Deep region.
    for i, (xi, yi) in enumerate(zip(x, y_pct)):
        va = "bottom" if i % 2 == 0 else "top"
        dy = 10 if i % 2 == 0 else -10
        ax.annotate(f"{yi:.1f}%", xy=(xi, yi),
                    xytext=(0, dy), textcoords="offset points",
                    ha="center", va=va,
                    fontsize=PCT_FS, fontweight="bold", color=color, zorder=4,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                              edgecolor="none", alpha=0.85))

    twin = ax.twiny()
    twin.set_xlim(ax.get_xlim())
    twin.set_xticks(x)
    twin.set_xticklabels([f"n={int(v):,}" for v in n],
                         rotation=45, fontsize=NTOP_FS, color="black")
    twin.tick_params(axis="x", which="both", length=0, pad=4)
    for spine in twin.spines.values():
        spine.set_visible(False)

    ax.set_xlabel("Bias depth score of SFT  $\\pi_{\\mathrm{SFT}} = $ DR $\\cdot$ FR",
                  fontsize=AXIS_FS)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0.0, 80)
    ax.set_yticks([0, 20, 40, 60, 80])
    ax.tick_params(axis="both", which="major", labelsize=TICK_FS)
    ax.grid(True, axis="y", color=GRID_C, lw=0.9, alpha=0.9, zorder=0)
    ax.set_axisbelow(True)
    leg = ax.legend(loc="lower right", fontsize=LEG_FS, frameon=True,
                    framealpha=0.95, edgecolor=SPINE_C)
    leg.get_frame().set_linewidth(0.8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="local dataset copy (default: download from HF)")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "figures")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    print("--- Olmo-3-7B ---")
    pi, sh = collect(load_outputs("olmo3_7b_pretrained", a.data),
                     load_outputs("olmo3_7b_sft", a.data))
    rows = bin_and_merge(pi, sh, n_bins=10, small_n=30)
    t_star = PI_THRESHOLD
    sm, dm = pi <= t_star, pi > t_star
    mean_shallow = float(sh[sm].mean()) if sm.any() else float("nan")
    mean_deep    = float(sh[dm].mean()) if dm.any() else float("nan")
    corr = float(np.corrcoef(pi, sh.astype(float))[0, 1])
    print(f"    n={sh.size:,}  overall={sh.mean()*100:.2f}%  r(pi, shared)={corr:+.3f}  "
          f"Shallow mean={mean_shallow*100:.1f}% (n={int(sm.sum())})  "
          f"Deep mean={mean_deep*100:.1f}% (n={int(dm.sum())})")
    print("    pi range           mean pi      n  shared  shared %")
    for r in rows:
        print(f"    [{r['lo']:.3f}, {r['hi']:.3f}]   {r['pi_mean']:.3f}  {r['n']:5,}  "
              f"{r['sh']:6,}  {r['rate']*100:7.1f}%")

    plt.rcParams.update({
        "font.family": "sans-serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, ax = plt.subplots(1, 1, figsize=(5.0, 4.2))
    draw_panel(ax, rows, t_star, mean_shallow, mean_deep)
    ax.set_ylabel("Rate of top$_{\\mathrm{SFT}}$ = top$_{\\mathrm{pretrained}}$  (%)",
                  fontsize=AXIS_FS - 2)
    fig.tight_layout(pad=1.2)
    stem = a.out / "pi_pretrained_vs_sft_same_top_deepbias_olmo"
    fig.savefig(stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"),          bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {stem}.pdf / .png")


if __name__ == "__main__":
    main()
