"""GEPA metric: score by how spread out the student's answers are.

For each (gold, pred) pair the metric:
  1. Collects K responses: `pred.answer` plus K-1 more student calls made in
     parallel on `gold.user_prompt`, each sent as a bare user message.
  2. Takes the top-1 share, the fraction of the K responses that land on
     the most common normalized answer. Score = 1 - top-1 share.
  3. Labels the outcome for the reflection LM: top-1 share below 50% is
     OK, 50% to 65% is PARTIAL, above 65% (the paper's bias cutoff) is
     FAILURE.
  4. Counts how many responses still match the SFT model's top-answer
     cluster (`cluster_match`) and reports it in the feedback text.
"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

import dspy

from deepbias.match import cluster_match

from .config import GEPARunConfig
from .phases import norm as _norm
from .prompts import load as _load_prompt

_FB_FAILURE = _load_prompt("feedback_failure")
_FB_PARTIAL = _load_prompt("feedback_partial")
_FB_OK = _load_prompt("feedback_ok")

OK_THRESHOLD = 0.50        # top-1 share below this is OK
BIAS_DR_THRESHOLD = 0.65   # top-1 share above this is FAILURE (paper's bias cutoff)


def make_metric(run: GEPARunConfig, student_lm: dspy.LM) -> Callable:
    """Return the GEPA metric bound to one run and student LM."""
    K = run.k_inner
    N = run.inner_concurrency

    def metric_distributional(gold, pred, trace=None, pred_name=None, pred_trace=None):
        responses = [(getattr(pred, "answer", "") or "").strip()]
        if K - 1 > 0:
            with ThreadPoolExecutor(max_workers=N) as ex:
                futs = [ex.submit(student_lm, gold.user_prompt) for _ in range(K - 1)]
                for f in as_completed(futs):
                    try:
                        r = f.result()
                        text = r[0] if isinstance(r, list) else str(r)
                        responses.append(text.strip())
                    except Exception:
                        responses.append("")

        n_residual = sum(cluster_match(r, gold.cluster) for r in responses if r)

        counts = Counter(_norm(r) for r in responses if r)
        if counts:
            top1_answer, top1_count = counts.most_common(1)[0]
            top1_share = top1_count / K
        else:
            top1_answer, top1_share = "(no responses)", 0.0
        score = max(0.0, 1.0 - top1_share)

        dist_lines = [f"  {ans!r}: {n}" for ans, n in counts.most_common()]
        dist = "\n".join(dist_lines) if dist_lines else "  (no non-empty samples)"
        fb_kwargs = dict(
            n_residual=n_residual, K=K,
            bias_mode=gold.bias_mode,
            top1_answer=top1_answer,
            top1_share_pct=f"{top1_share * 100:.0f}",
            ok_threshold_pct=f"{OK_THRESHOLD * 100:.0f}",
            bias_threshold_pct=f"{BIAS_DR_THRESHOLD * 100:.0f}",
            dist=dist,
        )
        if top1_share > BIAS_DR_THRESHOLD:
            fb = _FB_FAILURE.format(**fb_kwargs)
        elif top1_share >= OK_THRESHOLD:
            fb = _FB_PARTIAL.format(**fb_kwargs)
        else:
            fb = _FB_OK.format(**fb_kwargs)
        return dspy.Prediction(score=score, feedback=fb)

    return metric_distributional
