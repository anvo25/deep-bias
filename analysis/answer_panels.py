"""Answer-distribution bar panels shared by the case-study figures.

Used by fig1_distributions.py, fig5_case_sft_vs_lora.py and
appendix_cases.py. Each panel is a horizontal bar chart of answer counts out
of 30 samples. Green marks the model's direct top answer, amber its framed
top answer when that differs, grey everything else, and answers outside the
top rows are pooled into "(other)".
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

PANEL_BG = "#ffffff"
SPINE_C = "#b0b0b0"
DIRECT_C = "#1b5e20"
FRAMED_C = "#c08a17"
OTHER_C = "#9aa5b1"

TOP_N = 6  # at most this many answers per panel, the rest go to "(other)"

ROW_LABELS = {
    "olmo3_7b_pretrained": "Olmo-3-7B-Pretrained",
    "olmo3_7b_sft": "Olmo-3-7B-SFT",
    "olmo3_7b_sft_lora": "Olmo-3-7B-SFT + LoRA-SFT",
    "olmo3_7b_sft_gepa": "Olmo-3-7B-SFT + GEPA",
}

# Number words are shown as digits ("One" and "1" become one bar).
_NUMWORD_TO_DIGIT = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10",
}

# Per-family display aliases. "42" and "Forty-two" are one answer under
# surface_match but separate clusters, so they are merged into one bar.
ALIASES = {"305159": {"forty-two": "42", "option 3": "3"}}


def _alias(eid: str, lbl: str) -> str:
    return ALIASES.get(eid, {}).get(lbl.strip().lower(), lbl)


def _norm_label(lbl: str) -> str:
    return _NUMWORD_TO_DIGIT.get(lbl.strip().lower(), lbl)


def collapse_numwords(dist: list[dict]) -> list[dict]:
    """Merge number-word answers into their digit form. The digit form is
    displayed when present, otherwise the most common surface form."""
    digit_keys = set(_NUMWORD_TO_DIGIT.values())
    by_key: dict[str, dict] = {}
    for c in dist:
        key = _norm_label(c["answer"])
        if key not in by_key:
            by_key[key] = {"count": 0, "_top_src": c["answer"], "_top_n": -1}
        by_key[key]["count"] += int(c["count"])
        if key in digit_keys:
            by_key[key]["_top_src"] = key
            by_key[key]["_top_n"] = 10**9
        elif c["answer"].lower() == key.lower():
            by_key[key]["_top_src"] = c["answer"]
            by_key[key]["_top_n"] = 10**9
        elif int(c["count"]) > by_key[key]["_top_n"]:
            by_key[key]["_top_src"] = c["answer"]
            by_key[key]["_top_n"] = int(c["count"])
    return [{"answer": v["_top_src"], "count": v["count"]} for v in by_key.values()]


def _is_displayable_label(lbl: str, max_len: int = 22) -> bool:
    """Long or formatting-heavy answers (the model ignored the "one short
    answer" instruction) are pooled into "(other)" instead of getting a bar."""
    if len(lbl) > max_len:
        return False
    if any(ch in lbl for ch in ("\n", "*", "#", ":")):
        return False
    return True


def shared_axis(dists: list[list[dict]], eid: str, top_n: int = TOP_N,
                force_other: set[str] = frozenset(), alpha_ties: bool = False
                ) -> list[list[tuple[str, int]]]:
    """One shared set of rows across several distributions.

    Rows are ordered by combined count. Ties keep first-seen order, or
    alphabetical order with alpha_ties. An answer only gets its own row if it
    recurs (combined count >= 2), and answers in force_other (lowercase) are
    always pooled into "(other)".
    """
    per = []
    for d in dists:
        by, disp = {}, {}
        for c in d:
            cnt = int(c["count"])
            lbl = _alias(eid, c["answer"])
            if cnt <= 0 or not _is_displayable_label(lbl):
                continue
            k = lbl.lower()
            by[k] = by.get(k, 0) + cnt
            disp.setdefault(k, lbl)
        per.append((by, disp))
    union = list(dict.fromkeys(k for by, _ in per for k in by))
    if alpha_ties:
        union.sort()
    union.sort(key=lambda l: -sum(by.get(l, 0) for by, _ in per))
    show = {k: next((d[k] for _, d in per if k in d), k) for k in union}
    recur = [l for l in union if sum(by.get(l, 0) for by, _ in per) >= 2]
    top, beyond = recur[:top_n], recur[top_n:] + [l for l in union if l not in recur]
    kept = [l for l in top if l not in force_other]
    rest = beyond + [l for l in top if l in force_other]
    out = []
    for by, _ in per:
        row = [(show[l], by.get(l, 0)) for l in kept]
        if any(sum(b.get(l, 0) for l in rest) for b, _ in per):
            row.append(("(other)", sum(by.get(l, 0) for l in rest)))
        out.append(row)
    return out


def top_answers(r: dict) -> tuple[str, str]:
    return r["direct"]["top_answer"] or "", r["framed"]["top_answer"] or ""


def bar_colors(items, base_mode: str, fram_mode: str) -> list[str]:
    out = []
    for lbl, _ in items:
        ll = lbl.lower()
        if ll == (base_mode or "").lower():
            out.append(DIRECT_C)
        elif fram_mode and ll == fram_mode.lower():
            out.append(FRAMED_C)
        else:
            out.append(OTHER_C)
    return out


def render_panel(ax, items, colors, panel_title: str, *, min_slots: int = 4):
    ax.set_facecolor(PANEL_BG)
    for sp in ax.spines.values():
        sp.set_color(SPINE_C)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ys = np.arange(len(items))
    counts = [c for _, c in items]
    pct_labels = [f"{lbl} ({round(c / 30 * 100)}%)" for lbl, c in items]
    ax.barh(ys, counts, color=colors, edgecolor="white", linewidth=0.7, height=0.78)
    for y, c in zip(ys, counts):
        ax.text(c + 0.5, y, f"{c}", va="center", ha="left",
                fontsize=18, color="#333", fontweight="bold")
    ax.set_yticks(ys)
    ax.set_yticklabels(pct_labels, fontsize=20)
    ax.invert_yaxis()
    n_slots = max(len(items), min_slots)
    ax.set_ylim(n_slots - 0.5, -0.5)
    ax.set_xlim(0, 33)
    ax.set_xticks([])
    ax.tick_params(axis="x", bottom=False, labelbottom=False)
    ax.spines["bottom"].set_visible(False)
    ax.set_axisbelow(True)
    if panel_title:
        ax.set_title(panel_title, fontsize=22, fontweight="bold",
                     pad=8, loc="center", color="#222")


def compare_figure(eid: str, ra: dict, rb: dict, label_a: str, label_b: str,
                   wspace, alpha_ties: bool = False):
    """Two models, one row each, Direct | Framed columns on a shared axis.

    wspace is either a number or a function of the four row lists."""
    ad, af, bd, bf = shared_axis([collapse_numwords(ra["direct"]["distribution"]),
                                  collapse_numwords(ra["framed"]["distribution"]),
                                  collapse_numwords(rb["direct"]["distribution"]),
                                  collapse_numwords(rb["framed"]["distribution"])],
                                 eid, alpha_ties=alpha_ties)
    adm, afm = top_answers(ra)
    bdm, bfm = top_answers(rb)
    ws = wspace(ad, af, bd, bf) if callable(wspace) else wspace
    n = len(ad)
    row_h = 0.48 * n + 1.4
    fig = plt.figure(figsize=(10.0, row_h * 2 + 0.4))
    subs = fig.subfigures(2, 1, hspace=0.03)
    for sf, lab, items, cols, title, top in [
        (subs[0], label_a, (ad, af), (bar_colors(ad, adm, afm),
                                      bar_colors(af, adm, afm)), True, 0.74),
        (subs[1], label_b, (bd, bf), (bar_colors(bd, bdm, bfm),
                                      bar_colors(bf, bdm, bfm)), False, 0.86)]:
        sf.suptitle(lab, fontsize=22, fontweight="bold", y=0.97)
        ax = sf.subplots(1, 2, gridspec_kw={"wspace": ws})
        render_panel(ax[0], items[0], cols[0], "Direct" if title else "", min_slots=n)
        render_panel(ax[1], items[1], cols[1], "Framed" if title else "", min_slots=n)
        sf.subplots_adjust(left=0.16, right=0.97, top=top, bottom=0.05)
    return fig


def save(fig, stem, dpi: int, **kw) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", bbox_inches="tight",
                    **({"dpi": dpi} if ext == "png" else {}), **kw)
    plt.close(fig)
    print(f"  wrote {stem}.pdf / .png")
