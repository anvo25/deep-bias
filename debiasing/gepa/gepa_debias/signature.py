"""The DSPy program whose instruction GEPA optimizes."""
from __future__ import annotations

import dspy

from .prompts import load as _load_prompt

_SEED_INSTRUCTION = _load_prompt("seed_instruction")


class DebiasedAnswer(dspy.Signature):
    __doc__ = _SEED_INSTRUCTION  # the seed instruction, read from prompts/
    user_prompt: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="One short word or phrase, no other text.")


class Debiaser(dspy.Module):
    """Single-predictor module. GEPA mutates `answer.signature.instructions`."""

    def __init__(self):
        super().__init__()
        self.answer = dspy.Predict(DebiasedAnswer)

    def forward(self, user_prompt: str):
        return self.answer(user_prompt=user_prompt)
