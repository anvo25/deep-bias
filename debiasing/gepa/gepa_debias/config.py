"""Run configuration. The defaults are the settings of the paper run."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class GEPARunConfig:
    # Output location. GEPA logs, the discovered prompt, and eval rollouts go here.
    out_dir: Path

    # Anchor prompt families (ids in the released dataset). GEPA trains and
    # validates on all of them; see anchors/{deep,shallow,nonbias}.jsonl.
    train_anchor_ids: list[str] = field(default_factory=list)

    # Released dataset: a local copy, or None to download from Hugging Face.
    data_dir: Optional[str] = None

    # Student: the SFT model behind an OpenAI-compatible (vLLM) server.
    student_model: str = "tuongvy2603/Olmo3_baseline"
    student_base_url: str = "http://localhost:8000/v1"

    # Sampling inside the metric.
    k_inner: int = 30                  # samples per metric call
    inner_concurrency: int = 30        # parallel student calls per metric call

    # Held-out test prompts for the post-GEPA check.
    test_n: int = 100
    test_seed: int = 42

    # GEPA optimizer.
    gepa_auto: str = "medium"          # "light" | "medium" | "heavy"
    reflection_minibatch_size: int = 3
    num_threads: int = 8
    seed: int = 0
    track_stats: bool = True
    skip_perfect_score: bool = False

    # Reflection LM (proposes new instruction text), a litellm model name.
    reflection_lm_model: str = "openai/gpt-5.6-sol"
    reflection_max_tokens: int = 32000
    reflection_reasoning_effort: str = "medium"
    reflection_temperature: float = 1.0

    # Concurrency of the held-out rollouts.
    eval_concurrency: int = 128

    def eval_dir(self, tag: str) -> Path:
        """Directory for the held-out rollouts of one condition."""
        return Path(self.out_dir) / "eval_rollouts" / tag
