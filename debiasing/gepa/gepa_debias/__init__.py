"""GEPA system-prompt optimization for debiasing an SFT model.

  config.py     GEPARunConfig, the run settings
  signature.py  the DSPy program and its seed instruction
  data.py       anchors, released reference outputs, train/val/test sets
  clients.py    student and reflection LMs
  metric.py     the distributional metric and its feedback text
  phases.py     held-out rollouts and scoring after GEPA
  runner.py     end-to-end orchestration

Named gepa_debias so it does not shadow the `gepa` package DSPy depends on.
"""
from .config import GEPARunConfig
from .data import load_anchors
from .runner import run_gepa

__all__ = ["GEPARunConfig", "load_anchors", "run_gepa"]
