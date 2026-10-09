"""Optimize a debiasing system prompt for Olmo-3-7B-SFT with GEPA.

GEPA (via DSPy) mutates a seed instruction to make the student's answers on
open-ended "Choose a random ..." prompts as spread out as possible. It
trains and validates on 90 anchor prompt families (30 Deep, 30 Shallow,
30 non-bias, in anchors/). A reflection LM reads per-prompt feedback and
proposes new instructions. Afterwards the discovered prompt and a
no-system-prompt baseline are rolled out on 100 held-out prompts and
scored.

Requirements:
  * the student served behind an OpenAI-compatible endpoint, e.g.
      vllm serve tuongvy2603/Olmo-3-7B-Instruct-SFT-replicate --port 8000
  * an API key for the reflection LM in the environment (OPENAI_API_KEY
    for the default openai/gpt-5.6-sol)

Usage:
    python debiasing/gepa/run_gepa.py --data /path/to/deep-bias --out-dir work/gepa

Outputs in --out-dir: optimized_prompt.txt, initial_prompt.txt,
gepa_log/, eval_rollouts/, test_anchor_ids.json, per_anchor_results.json,
run_config.json.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

from gepa_debias import GEPARunConfig, load_anchors, run_gepa  # noqa: E402

DEFAULTS = GEPARunConfig(out_dir=REPO / "work" / "gepa")

# litellm provider prefix -> environment variable holding its API key
_KEY_ENV = {"openai": "OPENAI_API_KEY", "openrouter": "OPENROUTER_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--student-model", default=DEFAULTS.student_model,
                    help="served model name of the student (default: %(default)s)")
    ap.add_argument("--student-base-url", default=DEFAULTS.student_base_url,
                    help="OpenAI-compatible endpoint serving the student (default: %(default)s)")
    ap.add_argument("--reflection-model", default=DEFAULTS.reflection_lm_model,
                    help="litellm model name of the reflection LM (default: %(default)s)")
    ap.add_argument("--out-dir", type=Path, default=DEFAULTS.out_dir,
                    help="where logs and the discovered prompt are written (default: %(default)s)")
    ap.add_argument("--anchors-dir", type=Path, default=HERE / "anchors",
                    help="directory with deep.jsonl, shallow.jsonl, nonbias.jsonl "
                         "(default: %(default)s)")
    ap.add_argument("--data", default=None,
                    help="local copy of the released dataset; omit to download from Hugging Face")
    args = ap.parse_args()

    provider = args.reflection_model.split("/", 1)[0]
    env_var = _KEY_ENV.get(provider)
    if env_var and not os.environ.get(env_var):
        raise SystemExit(f"{args.reflection_model} needs an API key: set ${env_var}.")

    deep_ids, shallow_ids, nonbias_ids = load_anchors(args.anchors_dir)
    cfg = GEPARunConfig(
        out_dir=args.out_dir,
        train_anchor_ids=deep_ids + shallow_ids + nonbias_ids,
        data_dir=args.data,
        student_model=args.student_model,
        student_base_url=args.student_base_url,
        reflection_lm_model=args.reflection_model,
    )
    summary = run_gepa(cfg)
    print("\nSUMMARY:", summary)


if __name__ == "__main__":
    main()
