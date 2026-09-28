"""Anchors, released SFT outputs, and the GEPA train/val/test sets.

Every prompt is sent as the released prompt text plus ANSWER_SUFFIX, the
same string on every call. Replicate variety comes from the student's
sampling temperature alone, as in the main evaluation.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable

import dspy

from deepbias.client import ANSWER_SUFFIX
from deepbias.data import load_framings, load_outputs

REFERENCE_MODEL = "olmo3_7b_sft"  # the student's own released outputs
N_REPLICATES = 30                 # direct samples per test prompt, as in the main eval
TRAIN_REPS = 3                    # copies of each anchor in the trainset
VAL_REPS = 2                      # copies of each anchor in the valset


def load_anchors(anchors_dir: Path) -> tuple[list[str], list[str], list[str]]:
    """Return (deep_ids, shallow_ids, nonbias_ids) in file order."""
    def _read(path: Path) -> list[str]:
        with path.open() as f:
            return [json.loads(ln)["entity_id"] for ln in f if ln.strip()]

    anchors_dir = Path(anchors_dir)
    return (_read(anchors_dir / "deep.jsonl"),
            _read(anchors_dir / "shallow.jsonl"),
            _read(anchors_dir / "nonbias.jsonl"))


def _as_cluster(c: dict) -> dict:
    """Released cluster record to the dict shape `cluster_match` expects."""
    return {"canonical": c["answer"],
            "surface_forms": {s["text"]: s["count"] for s in c["surface_forms"]},
            "lemma_keys": c["lemma_keys"]}


def load_reference(data_dir=None) -> dict[str, dict]:
    """The SFT model's released outputs, keyed by prompt-family id.

    Each value has: prompt, mode (direct top answer), mode_rate (direct DR),
    cluster (the direct top cluster), framing_mode (framed top answer).
    """
    ref = {}
    for eid, r in load_outputs(REFERENCE_MODEL, data_dir).items():
        direct, framed = r["direct"], r.get("framed") or {}
        ref[eid] = {"prompt": r["prompt"],
                    "mode": direct["top_answer"],
                    "mode_rate": direct["dr"],
                    "cluster": _as_cluster(direct["distribution"][0]),
                    "framing_mode": framed.get("top_answer")}
    return ref


def build_trainset_valset(ref: dict, anchor_ids: list[str]
                          ) -> tuple[list[dspy.Example], list[dspy.Example]]:
    """Each anchor appears TRAIN_REPS times in train and VAL_REPS times in val.

    Repeats are separate examples on purpose: each one is a fresh stochastic
    rollout of the same prompt text.
    """
    train, val = [], []
    for eid in anchor_ids:
        r = ref[eid]
        ex = dict(user_prompt=f"{r['prompt']}\n\n{ANSWER_SUFFIX}",
                  bias_mode=r["mode"], entry_id=eid, cluster=r["cluster"])
        train += [dspy.Example(**ex).with_inputs("user_prompt") for _ in range(TRAIN_REPS)]
        val += [dspy.Example(**ex).with_inputs("user_prompt") for _ in range(VAL_REPS)]
    return train, val


def select_test_ids(ref: dict, test_n: int, test_seed: int,
                    exclude: Iterable[str], dr_floor: float = 0.3) -> list[str]:
    """Pick `test_n` random non-anchor prompts with direct DR >= dr_floor."""
    rng = random.Random(test_seed)
    excluded = set(exclude)
    candidates = [eid for eid, r in ref.items()
                  if eid not in excluded and r["mode_rate"] >= dr_floor]
    rng.shuffle(candidates)
    return candidates[:test_n]


def is_deep_test(r: dict, norm) -> bool:
    """Coarse Deep label for a test prompt: direct and framed top answers
    are equal after `norm`."""
    return norm(r["mode"] or "") == norm(r["framing_mode"] or "")


def build_direct_tasks(ref: dict, entry_ids: Iterable[str]) -> list[dict]:
    """N_REPLICATES direct rollouts per test prompt."""
    tasks = []
    for eid in entry_ids:
        prompt = f"{ref[eid]['prompt']}\n\n{ANSWER_SUFFIX}"
        for rep in range(N_REPLICATES):
            tasks.append({"entry_id": eid, "replicate_idx": rep, "prompt": prompt})
    return tasks


def build_framing_tasks(entry_ids: Iterable[str], data_dir=None) -> list[dict]:
    """One rollout per released framing of each test prompt."""
    ids = set(entry_ids)
    return [{"entry_id": r["id"],
             "framing_idx": r["framing_idx"],
             "base_question": r.get("prompt"),
             "framing_topic": r.get("framing_topic"),
             "prompt": f"{r['framing']}\n\n{ANSWER_SUFFIX}"}
            for r in load_framings(data_dir) if r["id"] in ids]
