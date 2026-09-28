"""End-to-end GEPA run. Public entry point: `run_gepa(run)`."""
from __future__ import annotations

import json
import time
from pathlib import Path

import dspy

from . import data as gepa_data
from .clients import make_reflection_lm, make_student_lm
from .config import GEPARunConfig
from .metric import make_metric
from .phases import norm, rollouts, score
from .signature import Debiaser


def run_gepa(run: GEPARunConfig) -> dict:
    """Select held-out test prompts, optimize the instruction with GEPA on
    the anchors, roll out baseline and discovered prompt on the test
    prompts, and score them. Returns the scoring summary."""
    out_dir = Path(run.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ref = gepa_data.load_reference(run.data_dir)
    print(f"loaded reference outputs for {len(ref)} prompt families")

    student_lm = make_student_lm(run)
    reflection_lm = make_reflection_lm(run)
    dspy.settings.configure(lm=student_lm)

    print("\n" + "=" * 78)
    print(f"PHASE 0: select {run.test_n} random non-anchor test prompts")
    print("=" * 78)
    anchors = run.train_anchor_ids
    test_ids = gepa_data.select_test_ids(ref, run.test_n, run.test_seed, exclude=anchors)
    deep_test = [e for e in test_ids if gepa_data.is_deep_test(ref[e], norm)]
    shal_test = [e for e in test_ids if not gepa_data.is_deep_test(ref[e], norm)]
    (out_dir / "test_anchor_ids.json").write_text(json.dumps(test_ids, indent=2))
    print(f"  Deep test : {len(deep_test)}   Shallow test: {len(shal_test)}")

    print("\n" + "=" * 78)
    print(f"PHASE 1: GEPA on {len(anchors)} anchors x {gepa_data.TRAIN_REPS} identical-text reps  "
          f"(k_inner={run.k_inner}, inner_concurrency={run.inner_concurrency})")
    print("=" * 78)
    trainset, valset = gepa_data.build_trainset_valset(ref, anchors)
    print(f"  trainset={len(trainset)}, valset={len(valset)}")

    base = Debiaser()
    (out_dir / "initial_prompt.txt").write_text(base.answer.signature.instructions)

    optimizer = dspy.GEPA(metric=make_metric(run, student_lm),
                          auto=run.gepa_auto,
                          reflection_lm=reflection_lm,
                          reflection_minibatch_size=run.reflection_minibatch_size,
                          num_threads=run.num_threads,
                          track_stats=run.track_stats,
                          skip_perfect_score=run.skip_perfect_score,
                          log_dir=str(out_dir / "gepa_log"),
                          seed=run.seed)
    t0 = time.time()
    optimized = optimizer.compile(base, trainset=trainset, valset=valset)
    gepa_secs = time.time() - t0
    opt_prompt = optimized.answer.signature.instructions
    (out_dir / "optimized_prompt.txt").write_text(opt_prompt)
    print(f"\n  GEPA done in {gepa_secs:.0f}s ({gepa_secs / 60:.1f} min)")

    print("\n" + "=" * 78)
    print(f"PHASE 2 & 3: evaluate baseline + GEPA on {len(test_ids)} test prompts")
    print("=" * 78)
    direct_tasks = gepa_data.build_direct_tasks(ref, test_ids)
    framing_tasks = gepa_data.build_framing_tasks(test_ids, run.data_dir)
    print(f"  direct={len(direct_tasks)}, framing={len(framing_tasks)}")
    rollouts(run, opt_prompt, direct_tasks, framing_tasks)

    summary = score(run, ref, test_ids, deep_test, shal_test)

    (out_dir / "run_config.json").write_text(json.dumps({
        "student_model": run.student_model,
        "k_inner": run.k_inner, "inner_concurrency": run.inner_concurrency,
        "test_n": run.test_n, "test_seed": run.test_seed,
        "gepa_auto": run.gepa_auto, "seed": run.seed,
        "reflection_lm_model": run.reflection_lm_model,
        "reflection_max_tokens": run.reflection_max_tokens,
        "reflection_reasoning_effort": run.reflection_reasoning_effort,
        "summary": summary,
        "gepa_wall_clock_sec": gepa_secs,
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, indent=2))
    print(f"\nwrote {out_dir / 'run_config.json'}")
    return summary
