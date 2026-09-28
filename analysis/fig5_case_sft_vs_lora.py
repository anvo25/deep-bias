"""Figure 5: "Choose a random Disney movie" before and after LoRA-SFT debiasing.

Olmo-3-7B-SFT answers "The Lion King" in all 30 direct samples. After
LoRA-SFT debiasing the same prompt family spreads over ten titles. Rows are
the two models, columns the Direct and Framed answer distributions.

    python analysis/fig5_case_sft_vs_lora.py                 # data from Hugging Face
    python analysis/fig5_case_sft_vs_lora.py --data ./hf     # a local copy

Writes 1693670__sft_vs_lora.pdf and .png to --out.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.answer_panels import ROW_LABELS, compare_figure, save  # noqa: E402
from deepbias.data import load_outputs  # noqa: E402

EID = "1693670"  # "Choose a random Disney movie"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="local dataset copy (default: download from HF)")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "figures")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    sft = load_outputs("olmo3_7b_sft", a.data)[EID]
    lora = load_outputs("olmo3_7b_sft_lora", a.data)[EID]
    # Ties in combined count are broken alphabetically (Cinderella above Toy
    # Story), as in the paper.
    fig = compare_figure(EID, sft, lora, ROW_LABELS["olmo3_7b_sft"],
                         ROW_LABELS["olmo3_7b_sft_lora"], wspace=1.7, alpha_ties=True)
    save(fig, a.out / f"{EID}__sft_vs_lora", dpi=170, pad_inches=0.2)


if __name__ == "__main__":
    main()
