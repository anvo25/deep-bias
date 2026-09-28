"""Step 5: Collapse step4's rewrites down to one representative per
distinct question via embedding-based dedup.

step4 rewrites are much more numerous than they are distinct: at full
scale, 6,864 successful rewrites collapsed to ~72% unique by exact string
match alone (e.g. "Choose a random number" showed up 47 times, "Choose a
random medical intervention" 67 times, both legitimate, different pre-rewrite
SFT rows genuinely converging on the same generalized question). This step
makes that collapse explicit and keeps exactly one representative per
distinct question.

Mechanism (same shape as step3's dedup -- rank once, then a hard similarity
cutoff, no relevance/diversity blend, just "is this a duplicate"):
  1. Embed every rewrite string (MiniLM, same model as the rest of the
     pipeline).
  2. Walk candidates in priority order by family_size_estimate descending
     -- the instance most strongly grounded in a real, large SFT cluster
     wins when duplicates compete.
  3. Keep an item only if its cosine similarity to every already-kept item
     is below --sim-threshold (default 0.9, matching step3's dedup
     threshold); otherwise it's folded into whichever kept item it's
     closest to.

Known risk worth knowing about, not hiding: a 0.9 cosine threshold on short
rewrite strings can merge things that are textually close but mean
something different (e.g. "Choose a random password" vs "Choose a random
password type" -- different expected answers). That's why every surviving
record keeps merged_rewrites (the actual distinct text variants folded into
it, not just a count) -- inspect step5_preview.txt for any duplicate_count
that looks large or looks like it merged questions that aren't really the
same, and lower --sim-threshold if you find one.

Below 0.95, embedding similarity alone stops being reliable in either
direction: real measurement on the full 4,725-item output found near-dupes
at 0.90-0.95 ("choose a random binary array" / "...binary string") sitting
right alongside clearly-different pairs at the same cosine ("choose a
random poet" / "...poem", disjoint number ranges), and the mix stays real
(not just noise) down through 0.80. Raising --sim-threshold to catch the
former would wrongly merge the latter. Instead, --judge-low opens an LLM
judge tier: candidates whose best cosine similarity to an already-kept item
falls in [--judge-low, --sim-threshold) get a real per-pair LLM decision
(gpt-5.6-luna, see dedup_judge.py) instead of a fixed cutoff. Below
--judge-low, items are kept automatically, no LLM call.

Outputs (inside ckpt_dir/step5/):
  step5_deduped.jsonl   one record per surviving representative, sorted by
                        duplicate_count descending: row_idx, rewrite,
                        example_answers, family_size_estimate,
                        duplicate_count, total_family_size,
                        merged_row_idxs, merged_rewrites, reason, evidence
  step5_preview.txt      human-readable, biggest duplicate groups first
  step5_summary.json     pool size in, unique count out, top duplicate groups

Usage:
  python step5_deduplicate.py --device cuda:1
  python step5_deduplicate.py --sim-threshold 0.85   # more aggressive merge
"""
import argparse
import hashlib
import json
import os
from collections import Counter

import numpy as np
from tqdm import tqdm

DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# 0.9 was tried first but merged some genuinely different questions (e.g.
# "choose a random password" absorbed "...password type", a different
# expected answer); 0.8 was worse (e.g. "password checker", a tool, became
# the representative for "password"). 0.95 was the first value that fixed
# both without losing the real duplicate collapses (medical intervention,
# graph type, sampling method, etc still merge correctly). See
# step5_preview.txt for the actual before/after comparison.
DEFAULT_SIM_THRESHOLD = 0.95
# Below this, embedding similarity is treated as "clearly different, don't
# bother asking" -- see the module docstring for the real-data band
# analysis that set this floor.
DEFAULT_JUDGE_LOW = 0.80
# Judge calls are independent (see _greedy_dedup's phase B), so this can go
# much higher than step4's concurrency=8 -- gpt-5.6-luna at low reasoning
# effort on a ~20-word pair is cheap and fast per call. Matches this
# project's other closed-model runs (e.g. gpt_5_6_sol.env), which also use
# CONCURRENCY=100 against OpenAI directly.
DEFAULT_JUDGE_CONCURRENCY = 100


def _fingerprint(texts: list, embed_model_name: str) -> dict:
    h = hashlib.sha256()
    h.update(embed_model_name.encode("utf-8"))
    h.update(str(len(texts)).encode("utf-8"))
    n = len(texts)
    positions = sorted(set(list(range(0, n, max(1, n // 200))) + ([n - 1] if n else [])))
    for i in positions:
        h.update((texts[i] or "")[:200].encode("utf-8", errors="ignore"))
    return {"n_texts": n, "embed_model": embed_model_name, "sha256": h.hexdigest()}


def _embed(texts: list, embed_model_name: str, cache_path: str, device: str, batch_size: int = 512):
    fp = _fingerprint(texts, embed_model_name)
    meta_path = cache_path + ".meta.json"
    if os.path.exists(cache_path) and os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            cached_fp = json.load(f)
        if cached_fp == fp:
            emb = np.load(cache_path)
            if len(emb) == len(texts):
                print(f"[Step 5] SKIP embedding -- loaded cached {cache_path}")
                return emb
        print("[Step 5] Cached embeddings don't match current rewrites -- recomputing.")

    from sentence_transformers import SentenceTransformer
    print(f"[Step 5] Embedding {len(texts):,} rewrites with {embed_model_name} on {device} ...")
    model = SentenceTransformer(embed_model_name, device=device)
    emb = model.encode(texts, batch_size=batch_size, show_progress_bar=True,
                        convert_to_numpy=True, normalize_embeddings=True)
    np.save(cache_path, emb)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(fp, f)
    return emb


def _pick_device(requested: str) -> str:
    import torch
    if requested != "auto":
        return requested
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        return "cuda:1"
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def _greedy_dedup(embeddings: np.ndarray, priority: np.ndarray, sim_threshold: float,
                   texts: list = None, judge_fn=None, judge_low: float = None,
                   judge_concurrency: int = DEFAULT_JUDGE_CONCURRENCY):
    """Returns (kept_idxs, assigned_to array where assigned_to[i] is the
    kept idx that i is a duplicate of -- kept items are always assigned to
    themselves, judge_stats).

    Two phases when judge_fn is given:

    Phase A (fast, vectorized, no LLM calls): exactly the original
    sim_threshold-only greedy walk. Every item's best cosine similarity to
    the *then-kept* set is recorded (including items that end up kept
    only because they didn't cross sim_threshold) -- this is the same
    pass whether or not judging is enabled, so results are identical to
    the pre-judge algorithm when judge_fn is None.

    Phase B (concurrent LLM calls, only when judge_fn is given): every
    phase-A-kept item whose best-match score falls in [judge_low,
    sim_threshold) is a "band candidate" with a partner already fixed by
    phase A -- these calls have no dependency on each other, so they run
    concurrently. Confirmed-duplicate pairs are merged via union-find
    (same pattern cluster_rejudge.py already uses elsewhere in this
    pipeline), representative per merged group = highest priority member.

    This trades strict left-to-right sequencing (each candidate always
    compared against the live, ever-shrinking kept set) for concurrency:
    a candidate is judged against its single nearest phase-A neighbor,
    which if that neighbor itself later merges away is resolved correctly
    through union-find, but a rejected candidate is never re-offered a
    second-best partner. In practice this matches the original design's
    own scope (it only ever compared against the single best match too),
    just resolved as a batch instead of one-by-one.
    """
    n = len(embeddings)
    order = np.argsort(-priority)
    max_sim_to_kept = np.full(n, -1.0, dtype=np.float32)
    best_kept_of = np.full(n, -1, dtype=np.int64)
    assigned_to = np.full(n, -1, dtype=np.int64)
    # Similarity to (and index of) the nearest *already-kept* item at the
    # moment each item was decided, captured before that item's own
    # self-similarity (1.0, always the max) overwrites max_sim_to_kept[idx]
    # and best_kept_of[idx] a few lines below to point at itself -- band
    # membership and partner lookup both read these snapshots, never
    # max_sim_to_kept/best_kept_of directly once the loop has moved on.
    kept_best_sim = np.full(n, -1.0, dtype=np.float32)
    kept_best_partner = np.full(n, -1, dtype=np.int64)
    kept = []
    for idx in tqdm(order, desc="[Step 5] greedy dedup (threshold pass)", unit="item"):
        idx = int(idx)
        sim_here = float(max_sim_to_kept[idx])
        if sim_here >= sim_threshold:
            continue  # duplicate of assigned_to[idx], already recorded
        kept.append(idx)
        kept_best_sim[idx] = sim_here
        kept_best_partner[idx] = best_kept_of[idx]
        sims = embeddings @ embeddings[idx]
        better = sims > max_sim_to_kept
        assigned_to[better] = idx
        best_kept_of[better] = idx
        max_sim_to_kept = np.maximum(max_sim_to_kept, sims)

    judge_stats = {"n_judged": 0, "n_judge_merged": 0}
    if judge_fn is None or judge_low is None:
        return kept, assigned_to, judge_stats

    candidates = [idx for idx in kept if judge_low <= kept_best_sim[idx] < sim_threshold]
    judge_stats["n_judged"] = len(candidates)

    parent = {idx: idx for idx in kept}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if priority[ra] >= priority[rb]:  # higher-priority root wins
            parent[rb] = ra
        else:
            parent[ra] = rb

    def _ask(idx: int):
        partner = int(kept_best_partner[idx])
        return idx, partner, judge_fn(texts[idx], texts[partner])

    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=judge_concurrency) as pool:
        futures = [pool.submit(_ask, idx) for idx in candidates]
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc="[Step 5] greedy dedup (judge pass)", unit="pair"):
            idx, partner, is_dup = fut.result()
            if is_dup:
                judge_stats["n_judge_merged"] += 1
                union(idx, partner)

    final_kept = [idx for idx in kept if find(idx) == idx]
    final_assigned_to = np.array([find(int(a)) for a in assigned_to], dtype=np.int64)
    return final_kept, final_assigned_to, judge_stats


def run(
    ckpt_dir: str = None,
    in_name: str = "step4/step4_generated.jsonl",
    out_name: str = "step5/step5_deduped.jsonl",
    preview_name: str = "step5/step5_preview.txt",
    summary_name: str = "step5/step5_summary.json",
    sim_threshold: float = DEFAULT_SIM_THRESHOLD,
    embed_model: str = DEFAULT_EMBED_MODEL,
    device: str = "auto",
    limit: int = None,
    use_judge: bool = True,
    judge_low: float = DEFAULT_JUDGE_LOW,
    judge_concurrency: int = DEFAULT_JUDGE_CONCURRENCY,
) -> str:
    ckpt_dir = ckpt_dir or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "work", "dataset")
    ckpt_dir = os.path.normpath(ckpt_dir)
    in_path = os.path.join(ckpt_dir, in_name)
    out_path = os.path.join(ckpt_dir, out_name)
    preview_path = os.path.join(ckpt_dir, preview_name)
    summary_path = os.path.join(ckpt_dir, summary_name)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    rows = []
    with open(in_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="[Step 5] loading step4 rewrites", unit="row"):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    print(f"[Step 5] {len(rows):,} rewrites loaded from {in_path}")

    if not rows:
        print("[Step 5] WARNING: nothing to dedup.")
        open(out_path, "w").close()
        return out_path

    device = _pick_device(device)
    cache_path = os.path.join(ckpt_dir, "step5", "step5_embeddings.npy")
    texts = [r.get("rewrite") or "" for r in rows]
    embeddings = _embed(texts, embed_model, cache_path, device).astype(np.float32)

    priority = np.array([r.get("family_size_estimate", 0) or 0 for r in rows], dtype=np.float32)

    judge_fn = None
    judge_cache = None
    if use_judge:
        from dedup_judge import DedupJudgeCache
        judge_cache_path = os.path.join(ckpt_dir, "step5", "step5_judge_cache.json")
        judge_cache = DedupJudgeCache(judge_cache_path, ckpt_dir=ckpt_dir)
        if not judge_cache.api_key:
            print("[Step 5] WARNING: --use-judge requested but no OpenAI API key found "
                  "Falling back to threshold-only dedup for the "
                  f"[{judge_low}, {sim_threshold}) band (items kept, not merged).")
        judge_fn = judge_cache.judge
        print(f"[Step 5] LLM judge ENABLED for cosine band [{judge_low}, {sim_threshold}) "
              f"-- model={judge_cache.model}, cache={judge_cache_path}")
    else:
        print(f"[Step 5] LLM judge disabled -- pure {sim_threshold} cosine cutoff, "
              f"matching pre-judge behavior.")

    kept, assigned_to, judge_stats = _greedy_dedup(
        embeddings, priority, sim_threshold,
        texts=texts, judge_fn=judge_fn, judge_low=judge_low if use_judge else None,
        judge_concurrency=judge_concurrency,
    )
    print(f"[Step 5] {len(kept):,} unique / {len(rows):,} total "
          f"({len(kept)/len(rows)*100:.1f}% unique) at sim_threshold={sim_threshold}")
    if use_judge:
        print(f"[Step 5] judge: {judge_stats['n_judged']:,} pairs asked, "
              f"{judge_stats['n_judge_merged']:,} merged "
              f"({judge_cache.n_cache_hits:,} cache hits, {judge_cache.n_calls:,} API calls, "
              f"{judge_cache.n_errors:,} errors)")

    deduped = []
    for kept_idx in tqdm(kept, desc="[Step 5] building deduped records", unit="rep"):
        group_idxs = np.where(assigned_to == kept_idx)[0]
        group_idxs = [i for i in group_idxs if i != kept_idx]
        rep = rows[kept_idx]
        merged_row_idxs = [rows[i].get("row_idx") for i in group_idxs]
        merged_rewrites = sorted(set(rows[i].get("rewrite") for i in group_idxs) - {rep.get("rewrite")})
        total_family_size = (rep.get("family_size_estimate", 0) or 0) + \
            sum(rows[i].get("family_size_estimate", 0) or 0 for i in group_idxs)
        deduped.append({
            "row_idx":               rep.get("row_idx"),
            "rewrite":               rep.get("rewrite"),
            "example_answers":       rep.get("example_answers"),
            "reason":                rep.get("reason"),
            "evidence":              rep.get("evidence"),
            "family_size_estimate":  rep.get("family_size_estimate"),
            "duplicate_count":       1 + len(group_idxs),
            "total_family_size":     total_family_size,
            "merged_row_idxs":       merged_row_idxs,
            "merged_rewrites":       merged_rewrites,
        })

    deduped.sort(key=lambda d: d["duplicate_count"], reverse=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for d in tqdm(deduped, desc="[Step 5] writing deduped", unit="row"):
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    with open(preview_path, "w", encoding="utf-8") as f:
        f.write(f"=== step5 preview -- {len(deduped):,} unique from {len(rows):,} total "
                f"(sim_threshold={sim_threshold}) ===\n\n")
        for d in deduped:
            f.write(f"row_idx {d['row_idx']} (duplicate_count={d['duplicate_count']}, "
                    f"total_family_size={d['total_family_size']}): {d['rewrite']}\n")
            f.write(f"    example_answers: {d['example_answers']}\n")
            if d["merged_rewrites"]:
                f.write(f"    merged variant text(s): {d['merged_rewrites']}\n")
            f.write("\n")

    dup_counts = Counter()
    for d in deduped:
        dup_counts[d["rewrite"]] = d["duplicate_count"]
    summary = {
        "n_in":              len(rows),
        "n_unique":           len(deduped),
        "pct_unique":         round(len(deduped) / len(rows) * 100, 1),
        "sim_threshold":      sim_threshold,
        "embed_model":        embed_model,
        "use_judge":          use_judge,
        "judge_low":          judge_low if use_judge else None,
        "judge_pairs_asked":  judge_stats["n_judged"] if use_judge else 0,
        "judge_pairs_merged": judge_stats["n_judge_merged"] if use_judge else 0,
        "top_duplicate_groups": dup_counts.most_common(20),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[Step 5] Done. {len(deduped):,} unique -> {out_path}")
    print(f"[Step 5] Preview -> {preview_path}")
    print(f"[Step 5] Summary -> {summary_path}")
    return out_path


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt-dir", default=None)
    p.add_argument("--in-name", default="step4/step4_generated.jsonl")
    p.add_argument("--out-name", default="step5/step5_deduped.jsonl")
    p.add_argument("--preview-name", default="step5/step5_preview.txt")
    p.add_argument("--summary-name", default="step5/step5_summary.json")
    p.add_argument("--sim-threshold", type=float, default=DEFAULT_SIM_THRESHOLD,
                    help=f"cosine similarity threshold for merging rewrites (default {DEFAULT_SIM_THRESHOLD}, "
                         "same value used for step3's dedup)")
    p.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    p.add_argument("--device", default="auto",
                    help="'auto' picks cuda:1 if a second GPU exists, else cuda:0, else cpu")
    p.add_argument("--limit", type=int, default=None, help="only load the first N step4 rows (smoke test)")
    p.add_argument("--use-judge", dest="use_judge", action="store_true", default=True,
                    help="LLM-judge the [--judge-low, --sim-threshold) cosine band instead of "
                         "keeping every item in it unmerged (default: on)")
    p.add_argument("--no-judge", dest="use_judge", action="store_false",
                    help="disable the LLM judge tier -- pure cosine cutoff, old behavior")
    p.add_argument("--judge-low", type=float, default=DEFAULT_JUDGE_LOW,
                    help=f"lower bound of the cosine band sent to the LLM judge (default {DEFAULT_JUDGE_LOW})")
    p.add_argument("--judge-concurrency", type=int, default=DEFAULT_JUDGE_CONCURRENCY,
                    help=f"parallel judge API calls (default {DEFAULT_JUDGE_CONCURRENCY}, calls are "
                         "independent -- see _greedy_dedup's phase B)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        ckpt_dir=args.ckpt_dir,
        in_name=args.in_name,
        out_name=args.out_name,
        preview_name=args.preview_name,
        summary_name=args.summary_name,
        sim_threshold=args.sim_threshold,
        embed_model=args.embed_model,
        device=args.device,
        limit=args.limit,
        use_judge=args.use_judge,
        judge_low=args.judge_low,
        judge_concurrency=args.judge_concurrency,
    )
