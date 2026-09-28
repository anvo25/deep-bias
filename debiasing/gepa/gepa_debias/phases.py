"""Held-out check after GEPA.

  Rollouts: sample the student on held-out test prompts, once without a
  system prompt and once with the discovered prompt.
  Scoring: per prompt, the share of responses that still fall in the SFT
  model's top-answer cluster (residual bias rate), split by Deep/Shallow.
"""
from __future__ import annotations

import json
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Optional

from openai import OpenAI

from deepbias.client import resolve_api_key
from deepbias.match import cluster_match

from .config import GEPARunConfig

_WS = re.compile(r"\s+")
_PUNC = re.compile(r"[\.\,\;\:\!\?\"\'\`]+")
_NUMWORD = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
            "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}


def norm(s: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace, map number words."""
    s = s.strip().lower()
    s = _PUNC.sub("", s)
    s = _WS.sub(" ", s).strip()
    return _NUMWORD.get(s, s)


def _call_student(client: OpenAI, model: str, sys_prompt: Optional[str],
                  user_prompt: str, max_tokens: int = 25) -> str:
    msgs = []
    if sys_prompt:
        msgs.append({"role": "system", "content": sys_prompt})
    msgs.append({"role": "user", "content": user_prompt})
    r = client.chat.completions.create(model=model, messages=msgs,
                                       temperature=0.6, top_p=0.95,
                                       max_tokens=max_tokens)
    return r.choices[0].message.content.strip()


def run_eval_phase(run: GEPARunConfig, tasks: list[dict],
                   key_fields: tuple[str, ...], out_path: Path,
                   sys_prompt: Optional[str], label: str) -> None:
    """Resumable rollout: rows whose `key_fields` already exist in
    `out_path` without an error are skipped."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        with out_path.open() as f:
            for ln in f:
                try:
                    r = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                if "response" in r and not r.get("error"):
                    done.add(tuple(r[k] for k in key_fields))
    todo = [t for t in tasks if tuple(t[k] for k in key_fields) not in done]
    print(f"  [{label}] todo={len(todo)} (already done={len(done)})", flush=True)
    if not todo:
        return
    client = OpenAI(base_url=run.student_base_url,
                    api_key=resolve_api_key(run.student_base_url), timeout=60.0)
    n_ok = n_err = 0
    with out_path.open("a") as fout, \
            ThreadPoolExecutor(max_workers=run.eval_concurrency) as ex:
        futs = {ex.submit(_call_student, client, run.student_model,
                          sys_prompt, t["prompt"]): t for t in todo}
        for fut in as_completed(futs):
            t = futs[fut]
            row = {**t, "ts": time.time(), "model": run.student_model,
                   "endpoint": run.student_base_url, "system_prompt": sys_prompt}
            try:
                row["response"] = fut.result()
                n_ok += 1
            except Exception as e:
                row["error"] = str(e)
                n_err += 1
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  [{label}] done ok={n_ok} err={n_err}", flush=True)


def rollouts(run: GEPARunConfig, opt_prompt: str,
             direct_tasks: list[dict], framing_tasks: list[dict]) -> None:
    """Roll out the no-system-prompt baseline and the discovered prompt."""
    for tag, sp in [("P0_baseline", None), ("P_gepa", opt_prompt)]:
        print(f"\n--- {tag} ---")
        eval_dir = run.eval_dir(tag)
        run_eval_phase(run, direct_tasks, ("entry_id", "replicate_idx"),
                       eval_dir / "responses_raw.jsonl", sp, f"{tag}/direct")
        run_eval_phase(run, framing_tasks, ("entry_id", "framing_idx"),
                       eval_dir / "framing_responses_raw.jsonl", sp, f"{tag}/framing")


def _load_direct(eval_dir: Path) -> dict[str, list[str]]:
    direct = defaultdict(list)
    with (eval_dir / "responses_raw.jsonl").open() as f:
        for ln in f:
            r = json.loads(ln)
            if r.get("response"):
                direct[r["entry_id"]].append(r["response"].strip())
    return direct


def score(run: GEPARunConfig, ref: dict, test_ids: Iterable[str],
          deep_ids: Iterable[str], shallow_ids: Iterable[str]) -> dict:
    """Residual bias rate per test prompt with a Deep/Shallow summary.
    Writes per_anchor_results.json to run.out_dir and returns the summary."""
    print("\n" + "=" * 78)
    print("PHASE 4: score residual bias rate (cluster_match: surface_forms "
          "+ lemma_keys + stems)")
    print("=" * 78)

    d0 = _load_direct(run.eval_dir("P0_baseline"))
    dX = _load_direct(run.eval_dir("P_gepa"))

    rows = []
    n_deep = n_shal = 0
    deep_flip = deep_drop = shal_flip = shal_drop = 0
    deep_set, shal_set = set(deep_ids), set(shallow_ids)
    for eid in test_ids:
        if not d0.get(eid) or not dX.get(eid):
            continue
        cluster = ref[eid]["cluster"]
        dr_before = sum(cluster_match(r, cluster) for r in d0[eid]) / len(d0[eid])
        residual = sum(cluster_match(r, cluster) for r in dX[eid]) / len(dX[eid])
        new_top = Counter(norm(r) for r in dX[eid]).most_common(1)[0][0]
        baseline_mode = ref[eid]["mode"]
        flipped = new_top != norm(baseline_mode)
        dropped = residual < 0.5
        if eid in deep_set:
            cohort = "Deep"
            n_deep += 1
            deep_flip += int(flipped)
            deep_drop += int(dropped)
        elif eid in shal_set:
            cohort = "Shallow"
            n_shal += 1
            shal_flip += int(flipped)
            shal_drop += int(dropped)
        else:
            cohort = "Other"
        rows.append({"entry_id": eid, "cohort": cohort,
                     "dr_before": dr_before, "residual_bias_rate": residual,
                     "baseline_top": baseline_mode, "new_top_norm": new_top,
                     "flipped": flipped, "residual_below_half": dropped})
    (Path(run.out_dir) / "per_anchor_results.json").write_text(json.dumps(rows, indent=2))

    summary = {
        "n_Deep": n_deep, "n_Shallow": n_shal,
        "Deep_flipped": deep_flip, "Shallow_flipped": shal_flip,
        "Deep_drop_below_half": deep_drop, "Shallow_drop_below_half": shal_drop,
    }
    nd, ns = max(n_deep, 1), max(n_shal, 1)
    print(f"\n--- Cohort summary (n_Deep={n_deep}, n_Shallow={n_shal}) ---")
    print("  Top-answer-flipped:")
    print(f"    Deep    : {deep_flip}/{nd} = {deep_flip / nd * 100:.0f}%")
    print(f"    Shallow : {shal_flip}/{ns} = {shal_flip / ns * 100:.0f}%")
    print("  Residual-bias-rate < 0.5:")
    print(f"    Deep    : {deep_drop}/{nd} = {deep_drop / nd * 100:.0f}%")
    print(f"    Shallow : {shal_drop}/{ns} = {shal_drop / ns * 100:.0f}%")
    try:
        from scipy.stats import fisher_exact
        p_flip = fisher_exact([[deep_flip, n_deep - deep_flip],
                               [shal_flip, n_shal - shal_flip]])[1]
        p_drop = fisher_exact([[deep_drop, n_deep - deep_drop],
                               [shal_drop, n_shal - shal_drop]])[1]
        summary["fisher_p_flip"] = float(p_flip)
        summary["fisher_p_drop"] = float(p_drop)
        print(f"\nFisher's exact:  p(flip)={p_flip:.3g}  p(drop<0.5)={p_drop:.3g}")
    except ImportError:
        pass
    return summary
