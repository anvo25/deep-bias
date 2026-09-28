"""Prompt text used by the GEPA run, kept as plain files.

  seed_instruction.txt   the seed system instruction that GEPA mutates
  feedback_failure.txt   metric feedback when the top-1 share is above 65%
  feedback_partial.txt   metric feedback when the top-1 share is 50% to 65%
  feedback_ok.txt        metric feedback when the top-1 share is below 50%

The feedback files share the placeholders {n_residual}, {K}, {bias_mode},
{top1_answer}, {top1_share_pct}, {ok_threshold_pct}, {bias_threshold_pct}
and {dist}.
"""
from __future__ import annotations

from pathlib import Path

PROMPTS_DIR = Path(__file__).parent


def load(name: str) -> str:
    """Return the contents of `name`.txt without its trailing newline."""
    path = PROMPTS_DIR / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"prompt file not found: {path}")
    return path.read_text().rstrip("\n")
