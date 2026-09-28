"""Step 3 of evaluation: cluster raw answers and compute DR and FR.

For each prompt family this reads the direct samples (sample_direct.py) and
the framed samples (sample_framed.py), groups answers that mean the same
thing into clusters, and writes one row per family with the full answer
distribution for both conditions:

    {
      "entry_id": "...",
      "n_baseline": 30, "n_framing": 30,
      "baseline": {"distribution": [{"canonical": ..., "count": ..., "rate": ...,
                                      "surface_forms": {...}}, ...],
                   "mode": ..., "mode_rate": ...,          # top answer and DR
                   "n_distinct_clusters": ..., ...},
      "framing":  {"distribution": [...],
                   "framed_rate": ...,                     # FR
                   ...}
    }

finalize.py turns this file into the released schema (with pi and bias type).

Usage:
    python evaluation/cluster.py \
        --responses-raw work/raw/my_model/direct.jsonl \
        --framing-raw   work/raw/my_model/framed.jsonl \
        --out           work/clustered/my_model.jsonl

Algorithm (5-stage clustering):
    Stage 1: Lemma canonicalization.
        Lowercase, keep digits, strip stopwords, lemmatize (noun + verb;
        take shorter form).
    Stage 2: Bucket by exact canonical match.
    Stage 3 (derivational stem): merge buckets whose canonicals reduce to
        the same stem after stripping one suffix (modern ~ modernist ~
        modernism ~ innovation ~ innovative), AND embedding cosine ≥ 0.55.
    Stage 4 (containment): merge buckets where one canonical's word-set is a
        subset of another's, AND embedding cosine ≥ 0.55, AND smaller side has
        ≤ 3 words.
    Stage 5 (synonym): merge buckets with embedding cosine ≥ 0.85, no shared
        wording required.
    Stages 3 and 4 share the same 0.55 floor: both merge on wording
    evidence (same stem, or containment) rather than pure meaning, so
    neither is trusted below that floor, and neither is held to the
    stricter 0.85 bar Stage 5 needs when there's no wording evidence at all.

Concurrency:
    - Embedding is the bottleneck; we batch-encode all unique canonicals
      across all entries in one call (with internal progress bar).
    - The per-entry union-find is then CPU-bound and runs in a process pool.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

HERE = Path(__file__).parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))




STOPWORDS = {
    "a", "an", "the", "of", "to", "in", "on", "at", "and", "or",
    "is", "are", "its", "this", "that",
}

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_THRESH = 0.85           # Stage 5: pure embedding similarity, no wording evidence
# Stages 3 and 4 both merge on strong *wording* evidence (same stem, or
# one contained in the other), which is a weaker signal than true synonymy,
# so both require only this lower embed floor rather than EMBED_THRESH.
SAME_WORDING_EMBED_FLOOR = 0.55
MAX_PHRASE_LEN = 3              # Stage 4: smaller side must have ≤ this many words

# Globals populated in worker init
_EMB_LOOKUP: dict[str, np.ndarray] | None = None


# ---------------- lemma canonicalization ----------------

def canonical(text: str) -> str:
    """Lowercase, keep alphanumeric tokens, strip stopwords, lemmatize each
    token via the project's central `_word_lemma` (lemminflect-backed,
    handles teeth->tooth / feet->foot / men->man and other irregulars),
    join with spaces.

    Stopword removal is local to within-stage clustering: it makes
    'the monarch' bucket to the same key as 'monarch'.  The cross-stage
    `surface_match` does not need to strip stopwords because it works on
    already-lemmatized keys.
    """
    if not text:
        return ""
    from deepbias.match import _word_lemma
    toks = re.findall(r"[a-zA-Z0-9]+", text.lower())
    lemmas: list[str] = []
    for w in toks:
        if w in STOPWORDS:
            continue
        if not w.isalpha():
            lemmas.append(w)
            continue
        lemmas.append(_word_lemma(w))
    return " ".join(lemmas) or text.lower()


# ---------------- clustering ----------------

def _cluster(canon_keys: list[str], emb_lookup: dict[str, np.ndarray]) -> list[list[int]]:
    """Union-find over canon_keys; returns list of clusters (each a list of
    indices into canon_keys).

    Stages, in order of application (each builds on the previous unions):
      Stage 3: derivational-stem equality (modern ~ modernist ~ modernism)
               + SAME_WORDING_EMBED_FLOOR guard, same reasoning as Stage 4:
               sharing a stem is wording evidence, not proof of meaning
               (e.g. 'realism'/'reality' both stem to 'real' but measure
               only ~0.63 cosine), so it gets the same low-but-nonzero bar.
      Stage 4: word-set containment (ISO subset of ISO 9001) + the same
               embedding floor, to guard against e.g. 'red wine' subset
               'wine'.
      Stage 5: pure embedding similarity above EMBED_THRESH, no shared
               wording required at all.
    """
    n = len(canon_keys)
    if n == 0:
        return []
    sets = [set(k.split()) for k in canon_keys]
    embs = np.stack([emb_lookup[k] for k in canon_keys])

    # Derivational stems on the already-lemmatized canonical keys, so the
    # suffix stripper sees plurals/tenses already collapsed.
    from deepbias.match import _strip_derivational
    stems = [
        " ".join(_strip_derivational(tok) for tok in k.split())
        for k in canon_keys
    ]

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # Stage 3: derivational stem equality, guarded by SAME_WORDING_EMBED_FLOOR.
    by_stem: dict[str, list[int]] = {}
    for i, s in enumerate(stems):
        if s:
            by_stem.setdefault(s, []).append(i)
    for idxs in by_stem.values():
        if len(idxs) > 1:
            for a in range(len(idxs)):
                for b in range(a + 1, len(idxs)):
                    i, j = idxs[a], idxs[b]
                    if float(np.dot(embs[i], embs[j])) >= SAME_WORDING_EMBED_FLOOR:
                        union(i, j)

    # Stage 4: containment with embed-floor guard
    for i in range(n):
        for j in range(i + 1, n):
            wi, wj = sets[i], sets[j]
            if not wi or not wj:
                continue
            smaller = wi if len(wi) <= len(wj) else wj
            larger = wj if len(wj) >= len(wi) else wi
            if len(smaller) > MAX_PHRASE_LEN:
                continue
            if (smaller & larger) != smaller:
                continue
            if float(np.dot(embs[i], embs[j])) >= SAME_WORDING_EMBED_FLOOR:
                union(i, j)

    # Stage 5: pure embedding similarity
    for i in range(n):
        for j in range(i + 1, n):
            if float(np.dot(embs[i], embs[j])) >= EMBED_THRESH:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def _dist_from_clusters(canon_keys: list[str],
                        members_by_canon: dict[str, list[str]],
                        cluster_idx_groups: list[list[int]],
                        total_outputs: int) -> list[dict]:
    out = []
    for idxs in cluster_idx_groups:
        canonicals = [canon_keys[i] for i in idxs]
        members: list[str] = []
        for c in canonicals:
            members.extend(members_by_canon[c])
        count = len(members)
        # surface form counts
        sc: dict[str, int] = {}
        for s in members:
            sc[s] = sc.get(s, 0) + 1
        # representative = most-common surface form (lexicographic tiebreak)
        rep = sorted(sc.items(), key=lambda x: (-x[1], x[0]))[0][0]
        out.append({
            "canonical": rep,
            "count": count,
            "rate": count / total_outputs if total_outputs else 0.0,
            "surface_forms": sc,
            "lemma_keys": canonicals,
        })
    out.sort(key=lambda d: -d["count"])
    return out


def _entropy_bits(dist: list[dict]) -> float:
    return -sum(
        d["rate"] * math.log2(d["rate"])
        for d in dist if d["rate"] > 0
    )


# ---------------- per-entry pipeline (worker) ----------------

def _init_worker(emb_lookup: dict[str, np.ndarray]) -> None:
    global _EMB_LOOKUP
    _EMB_LOOKUP = emb_lookup


def _process_entry(args: tuple) -> dict:
    """Worker: cluster one entry's baseline + framing outputs."""
    eid, bucket, tag, base_outputs, frame_outputs = args
    assert _EMB_LOOKUP is not None

    def by_canon(outputs: list[str]) -> dict[str, list[str]]:
        d: dict[str, list[str]] = {}
        for o in outputs:
            c = canonical(o)
            d.setdefault(c, []).append(o)
        return d

    def dist_for(outputs: list[str]) -> list[dict]:
        if not outputs:
            return []
        members_by_canon = by_canon(outputs)
        keys = list(members_by_canon.keys())
        groups = _cluster(keys, _EMB_LOOKUP)
        return _dist_from_clusters(keys, members_by_canon, groups, len(outputs))

    base_dist = dist_for(base_outputs)
    frame_dist = dist_for(frame_outputs)

    # framing metrics referencing baseline mode.  We use the same
    # surface_match primitive as the within-stage cluster judge and the
    # Deep/Shallow cross-stage comparison, so 'ISO 9001' (baseline) and
    # 'ISO' (framing), or 'modernist' (baseline) and 'modernism' (framing),
    # correctly resolve to the same answer.
    from deepbias.match import surface_match
    framed_rate = 0.0
    flip_target = None
    flip_target_rate = 0.0
    if base_dist:
        base_canonical = base_dist[0]["canonical"]
        base_lemmas = base_dist[0]["lemma_keys"]

        def _is_baseline_mode_cluster(d: dict) -> bool:
            """A framing cluster matches the baseline mode iff its canonical
            or any of its lemma_keys matches the baseline mode's canonical
            or any of its lemma_keys via `surface_match`."""
            candidates_d = [d["canonical"]] + list(d.get("lemma_keys", []))
            candidates_b = [base_canonical] + list(base_lemmas)
            for ca in candidates_d:
                for cb in candidates_b:
                    if surface_match(ca, cb):
                        return True
            return False

        match_count = sum(d["count"] for d in frame_dist
                          if _is_baseline_mode_cluster(d))
        framed_rate = match_count / len(frame_outputs) if frame_outputs else 0.0
        # top non-mode framing cluster
        for d in frame_dist:
            if not _is_baseline_mode_cluster(d):
                flip_target = d["canonical"]
                flip_target_rate = d["rate"]
                break

    def metrics(dist: list[dict], n_total: int) -> dict:
        H = _entropy_bits(dist)
        Hmax = math.log2(n_total) if n_total > 1 else 1.0
        return {
            "distribution": dist,
            "mode": dist[0]["canonical"] if dist else None,
            "mode_rate": dist[0]["rate"] if dist else 0.0,
            "n_distinct_clusters": len(dist),
            "entropy_bits": H,
            "entropy_normalized": (H / Hmax) if Hmax > 0 else 0.0,
        }

    return {
        "entry_id": eid,
        "bucket": bucket,
        "tag": tag,
        "n_baseline": len(base_outputs),
        "n_framing": len(frame_outputs),
        "baseline": metrics(base_dist, len(base_outputs)),
        "framing": {
            **metrics(frame_dist, len(frame_outputs)),
            "framed_rate": framed_rate,
            "framing_flip_target": flip_target,
            "framing_flip_target_rate": flip_target_rate,
        },
    }


# ---------------- main ----------------

def _load_jsonl(p: Path) -> list[dict]:
    """Read a raw-sample file (.jsonl or .jsonl.gz). Released raw files use
    `id`, the samplers write `entry_id`; both are accepted."""
    opener = gzip.open if p.suffix == ".gz" else open
    rows = []
    with opener(p, "rt") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if "entry_id" not in r:
                    r["entry_id"] = str(r["id"])
                rows.append(r)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses-raw", type=Path, required=True)
    ap.add_argument("--framing-raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default=None,
                    help="'cuda', 'cpu', or None for auto-select.")
    ap.add_argument("--encode-batch-size", type=int, default=256)
    ap.add_argument("--workers", type=int,
                    default=max(1, (os.cpu_count() or 4) - 1))
    args = ap.parse_args()

    print(f"[cluster] loading raw outputs ...", flush=True)
    raw_base = _load_jsonl(args.responses_raw)
    raw_frame = _load_jsonl(args.framing_raw)
    aux = {}  # optional per-entry bucket/tag metadata, unused in the release

    # When the upstream run was launched with --brace-format (for
    # safety-filtered models that often refuse), each row already has parsed_answer
    # (lowercased, brace-stripped, refusal-filtered, and number-word
    # normalised by deepbias.refusal_filter).  Use that as the
    # cluster-judge input where present; refusals (parsed_answer=None) are
    # skipped so they don't get merged into a "refusal cluster".  For
    # variants without --brace-format we fall back to the raw response text.
    def _select_response(r: dict) -> str | None:
        if "parsed_answer" in r:
            return (r.get("parsed_answer") or "").strip() or None
        return (r.get("response") or "").strip() or None

    base_by_eid: dict[str, list[str]] = {}
    for r in raw_base:
        resp = _select_response(r)
        if resp:
            base_by_eid.setdefault(r["entry_id"], []).append(resp)
    frame_by_eid: dict[str, list[str]] = {}
    for r in raw_frame:
        resp = _select_response(r)
        if resp:
            frame_by_eid.setdefault(r["entry_id"], []).append(resp)

    all_eids = sorted(set(base_by_eid.keys()) | set(frame_by_eid.keys()))
    print(f"[cluster] entries: {len(all_eids)}  "
          f"(with_baseline={len(base_by_eid)}, with_framing={len(frame_by_eid)})", flush=True)

    # Collect all unique canonicals globally (so embedding pass is one call)
    print("[cluster] computing canonical forms ...", flush=True)
    all_canonicals: set[str] = set()
    work_items: list[tuple] = []
    for eid in tqdm(all_eids, desc="canonicalize"):
        base = base_by_eid.get(eid, [])
        frame = frame_by_eid.get(eid, [])
        for o in base + frame:
            c = canonical(o)
            if c:
                all_canonicals.add(c)
        bucket = (aux.get(eid) or {}).get("bucket")
        tag = (aux.get(eid) or {}).get("tag")
        work_items.append((eid, bucket, tag, base, frame))
    print(f"[cluster] unique canonicals: {len(all_canonicals):,}", flush=True)

    # Batch encode all canonicals once
    print(f"[cluster] loading embedder ({EMBED_MODEL}) ...", flush=True)
    model = SentenceTransformer(EMBED_MODEL, device=args.device)
    canonical_list = sorted(all_canonicals)
    print(f"[cluster] encoding {len(canonical_list):,} canonicals "
          f"(batch_size={args.encode_batch_size}) ...", flush=True)
    embs = model.encode(
        canonical_list,
        batch_size=args.encode_batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    emb_lookup = {c: embs[i] for i, c in enumerate(canonical_list)}
    del embs, model  # free GPU memory before spawning workers

    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(f"[cluster] clustering {len(work_items)} entries "
          f"with {args.workers} workers ...", flush=True)
    with args.out.open("w") as fout, \
         ProcessPoolExecutor(max_workers=args.workers,
                              initializer=_init_worker,
                              initargs=(emb_lookup,)) as ex:
        futures = [ex.submit(_process_entry, wi) for wi in work_items]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="cluster"):
            rec = fut.result()
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[cluster] wrote {args.out}  ({len(work_items)} rows)", flush=True)


if __name__ == "__main__":
    main()
