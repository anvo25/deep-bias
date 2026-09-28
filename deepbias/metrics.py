"""Bias depth metrics.

For one prompt family and one model:
  DR  direct rate: share of the 30 direct samples that give the top answer
  FR  framed rate: share of the 30 scenario reframings that give that SAME
      direct top answer
  pi  bias depth score: DR x FR

A family is
  Deep     if pi > 0.40
  Shallow  if it is not Deep and DR > 0.65
  Non-bias otherwise

The two cutoffs are one decision applied twice: "a clear majority" on both
axes, DR > 0.65 and FR > 0.65, which gives pi > 0.65^2 = 0.4225, rounded to
0.40.
"""
from __future__ import annotations

DR_THRESHOLD = 0.65
PI_THRESHOLD = 0.40


def bias_type(dr: float, pi: float) -> str:
    if pi > PI_THRESHOLD:
        return "deep"
    if dr > DR_THRESHOLD:
        return "shallow"
    return "non_bias"


def breakdown(rows) -> dict[str, float]:
    """Percent of prompt families that are Deep, Shallow, and Non-bias."""
    rows = list(rows)
    n = len(rows)
    out = {"deep": 0, "shallow": 0, "non_bias": 0}
    for r in rows:
        out[r["bias_type"]] += 1
    return {k: 100 * v / n for k, v in out.items()} | {"n": n}
