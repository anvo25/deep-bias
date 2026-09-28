"""LM factories for the student and the reflection LM."""
from __future__ import annotations

import dspy

from deepbias.client import resolve_api_key

from .config import GEPARunConfig


def make_student_lm(run: GEPARunConfig, temperature: float = 0.6,
                    max_tokens: int = 25) -> dspy.LM:
    """The SFT student behind an OpenAI-compatible server, as a dspy.LM."""
    return dspy.LM(model=f"openai/{run.student_model}",  # litellm OpenAI-compatible route
                   api_base=run.student_base_url,
                   api_key=resolve_api_key(run.student_base_url),
                   temperature=temperature,
                   max_tokens=max_tokens,
                   cache=False)


def make_reflection_lm(run: GEPARunConfig) -> dspy.LM:
    """The reflection LM that proposes new instruction text. litellm reads
    its API key from the environment (e.g. OPENAI_API_KEY)."""
    return dspy.LM(model=run.reflection_lm_model,
                   temperature=run.reflection_temperature,
                   max_tokens=run.reflection_max_tokens,
                   cache=False,
                   reasoning_effort=run.reflection_reasoning_effort)
