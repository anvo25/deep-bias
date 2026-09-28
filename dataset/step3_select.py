"""Step 3: Diverse selection over step2's candidates -- rank by relevance,
then hard-exclude near-duplicates.

Replaces an earlier KMeans-clustering version of this step. The problem
with fixed-k clustering here: you have to guess k, but what actually
matters is "up to N diverse, relevant items", not "k clusters" -- guess k
too small and distinct topics get merged; too large and one topic
fragments into several near-identical clusters, each burning a separate
step4 LLM call on the same underlying pattern (we saw this concretely: an
exact-duplicate physics question showed up 4x in a 99-row smoke sample).

This used to blend relevance and diversity into one score (Maximal
Marginal Relevance / MMR), but since --select-n defaults to ranking the
ENTIRE pool (no early cutoff), that blend never actually excludes anything
on its own, only a hard similarity cutoff does. So it's simpler and more
honest to just do that directly:

  1. Sort candidates once by step2's tfidf_score, highest first.
  2. Walk that fixed order. Keep a candidate unless it's too similar
     (>= --dedup-sim-threshold cosine similarity, on MiniLM sentence
     embeddings) to something already kept -- in which case it's dropped
     outright, not soft-penalized.

Only the (already small, LLM-free) step2 pool is embedded, not the full
Dolci corpus. Runs on a single GPU (defaults to cuda:1, override with
--device).

For each selected representative, its nearest neighbors in the pool (by
the same embedding similarity) are attached as "evidence" for step4's LLM
call, and a cheap "family_size_estimate" (count of pool rows within
--family-sim-threshold cosine similarity) is recorded as a rough proxy for
how common the underlying pattern is in Dolci -- both come for free from
the same similarity vector the greedy loop already computes each step, no
extra pass needed.

By default --select-n ranks the ENTIRE step2 pool, not some fixed count.
There's no LLM cost in this step (only embeddings + numpy), so there's no
reason to guess a count here and re-run this step every time you want to
try a different one. Run this once; step4's --limit then controls how deep
into this one ranking you actually spend LLM calls, cheaply, without ever
re-embedding or re-running this step.

Outputs (inside ckpt_dir/step3/):
  step3_selected.jsonl   one JSON object per selected representative, in
                         selection order: selection_rank, row_idx,
                         user_prompt, tfidf_score, tfidf_score_norm,
                         tfidf_best_seed, max_similarity_to_selected,
                         family_size_estimate, auto_label_terms, evidence
                         (nearest-neighbor list)
  step3_preview.txt      human-readable sample of selections + their evidence
  step3_summary.json     pool size, selected count, score/family-size stats

Usage:
  python step3_select.py --device cuda:1          # ranks the whole pool
  python step3_select.py --select-n 50 --device cpu   # smoke test
"""
import argparse
import hashlib
import json
import os

import numpy as np
from tqdm import tqdm

from step2_filter import _top_terms  # reuse the same cheap word-frequency helper

DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PROMPT_CHAR_LIMIT = 2000
DEFAULT_SELECT_N = None  # None = rank the entire pool -- see run()
DEFAULT_EVIDENCE_K = 0
DEFAULT_FAMILY_SIM_THRESHOLD = 0.6
DEFAULT_DEDUP_SIM_THRESHOLD = 0.9


def _fingerprint(rows: list, embed_model_name: str) -> dict:
    h = hashlib.sha256()
    h.update(embed_model_name.encode("utf-8"))
    h.update(str(len(rows)).encode("utf-8"))
    n = len(rows)
    positions = sorted(set(list(range(0, n, max(1, n // 200))) + ([n - 1] if n else [])))
    for i in positions:
        h.update(str(rows[i].get("row_idx")).encode("utf-8"))
        h.update((rows[i].get("user_prompt") or "")[:200].encode("utf-8", errors="ignore"))
    return {"n_rows": n, "embed_model": embed_model_name, "sha256": h.hexdigest()}


def _embed(rows: list, embed_model_name: str, cache_path: str, device: str, batch_size: int = 512):
    fp = _fingerprint(rows, embed_model_name)
    meta_path = cache_path + ".meta.json"
    if os.path.exists(cache_path) and os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            cached_fp = json.load(f)
        if cached_fp == fp:
            emb = np.load(cache_path)
            if len(emb) == len(rows):
                print(f"[Step 3] SKIP embedding -- loaded cached {cache_path}")
                return emb
        print("[Step 3] Cached embeddings don't match current rows -- recomputing.")

    from sentence_transformers import SentenceTransformer
    print(f"[Step 3] Embedding {len(rows):,} rows with {embed_model_name} on {device} ...")
    model = SentenceTransformer(embed_model_name, device=device)
    texts = [(r.get("user_prompt") or "")[:PROMPT_CHAR_LIMIT] for r in rows]
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


def _greedy_select(embeddings: np.ndarray, relevance: np.ndarray, select_n: int,
                    dedup_sim_threshold: float, evidence_k: int, family_sim_threshold: float):
    """Walk candidates once, highest relevance first (a fixed order, not
    recomputed as picks happen). Keep a candidate unless it's already too
    similar (>= dedup_sim_threshold cosine) to something kept earlier in
    that order -- that's the only thing that excludes anything here, no
    relevance/diversity blending.

    embeddings must be L2-normalized (cosine sim = dot product). relevance
    must already be normalized to [0, 1] (only used for the sort order).

    Returns a list of dicts, one per selection, in selection order:
    {idx, max_similarity_to_selected, neighbor_idxs, neighbor_sims,
     family_size_estimate}.
    """
    n = len(embeddings)
    select_n = min(select_n, n)
    order = np.argsort(-relevance)  # fixed priority order, decided once upfront
    max_sim = np.zeros(n, dtype=np.float32)  # running max similarity to anything kept so far
    results = []

    for idx in tqdm(order, desc="[Step 3] greedy selection", unit="row"):
        if len(results) >= select_n:
            break
        if max_sim[idx] >= dedup_sim_threshold:
            continue  # too similar to something already kept -- dropped outright

        sims_to_pick = embeddings @ embeddings[idx]  # (n,) cosine sims, reused for everything below
        family_size = int((sims_to_pick >= family_sim_threshold).sum())

        neighbor_sims = sims_to_pick.copy()
        neighbor_sims[idx] = -np.inf  # exclude self from its own evidence
        neighbor_idxs = np.argsort(-neighbor_sims)[:evidence_k]
        neighbor_idxs = [int(i) for i in neighbor_idxs if np.isfinite(neighbor_sims[i])]

        results.append({
            "idx":                          int(idx),
            "max_similarity_to_selected":   float(max_sim[idx]),
            "family_size_estimate":         family_size,
            "neighbor_idxs":                neighbor_idxs,
            "neighbor_sims":                [float(sims_to_pick[i]) for i in neighbor_idxs],
        })

        max_sim = np.maximum(max_sim, sims_to_pick)

    return results


def run(
    ckpt_dir: str = None,
    in_name: str = "step2/step2_candidates.jsonl",
    out_name: str = "step3/step3_selected.jsonl",
    preview_name: str = "step3/step3_preview.txt",
    summary_name: str = "step3/step3_summary.json",
    select_n: int = DEFAULT_SELECT_N,
    evidence_k: int = DEFAULT_EVIDENCE_K,
    family_sim_threshold: float = DEFAULT_FAMILY_SIM_THRESHOLD,
    dedup_sim_threshold: float = DEFAULT_DEDUP_SIM_THRESHOLD,
    min_score: float = 0.0,
    top_n: int = None,
    embed_model: str = DEFAULT_EMBED_MODEL,
    device: str = "auto",
    limit: int = None,
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
        for line in tqdm(f, desc="[Step 3] loading step2 candidates", unit="row"):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    print(f"[Step 3] {len(rows):,} candidates loaded from {in_path}")

    if min_score > 0.0:
        rows = [r for r in rows if r.get("tfidf_score", 0.0) >= min_score]
        print(f"[Step 3] {len(rows):,} remain after --min-score {min_score}")
    if top_n is not None:
        rows.sort(key=lambda r: r.get("tfidf_score", 0.0), reverse=True)
        rows = rows[:top_n]
        print(f"[Step 3] {len(rows):,} remain after --top-n {top_n}")

    if not rows:
        print("[Step 3] WARNING: nothing to select from.")
        open(out_path, "w").close()
        return out_path

    # None (the default) ranks the ENTIRE pool, not just some fixed count --
    # there's no LLM cost in this step, only step4 costs per item, so "how
    # many to actually send to the LLM" is better decided later via step4's
    # --limit against this one full ranking, rather than re-running this
    # (embedding + greedy selection) step from scratch for every N you want
    # to try.
    effective_select_n = len(rows) if select_n is None else select_n
    print(f"[Step 3] Selecting {effective_select_n:,} of {len(rows):,} candidates "
          f"{'(full pool)' if select_n is None else ''}")

    device = _pick_device(device)
    cache_path = os.path.join(ckpt_dir, "step3", "step3_embeddings.npy")
    embeddings = _embed(rows, embed_model, cache_path, device).astype(np.float32)

    raw_scores = np.array([r.get("tfidf_score", 0.0) for r in rows], dtype=np.float32)
    lo, hi = float(raw_scores.min()), float(raw_scores.max())
    relevance = (raw_scores - lo) / (hi - lo + 1e-12)

    picks = _greedy_select(embeddings, relevance, effective_select_n,
                            dedup_sim_threshold, evidence_k, family_sim_threshold)

    selected_records = []
    for rank, p in enumerate(tqdm(picks, desc="[Step 3] building selected records", unit="pick")):
        rep = rows[p["idx"]]
        evidence = [
            {
                "row_idx":         rows[i].get("row_idx"),
                "user_prompt":     rows[i].get("user_prompt"),
                "tfidf_score":     rows[i].get("tfidf_score"),
                "tfidf_best_seed": rows[i].get("tfidf_best_seed"),
                "similarity":      sim,
            }
            for i, sim in zip(p["neighbor_idxs"], p["neighbor_sims"])
        ]
        auto_label_terms = _top_terms([rep.get("user_prompt")] + [e["user_prompt"] for e in evidence])
        selected_records.append({
            "selection_rank":              rank,
            "row_idx":                     rep.get("row_idx"),
            "user_prompt":                 rep.get("user_prompt"),
            # Present unless step1 was run with --drop-responses. Step 4
            # shows this to the LLM as unreliable context, never as ground
            # truth (it may be a refusal, wrong, or itself an instance of
            # the exact bias this project studies).
            "assistant_response":          rep.get("assistant_response"),
            "tfidf_score":                 rep.get("tfidf_score"),
            "tfidf_score_norm":            float(relevance[p["idx"]]),
            "tfidf_best_seed":             rep.get("tfidf_best_seed"),
            "max_similarity_to_selected":  p["max_similarity_to_selected"],
            "family_size_estimate":        p["family_size_estimate"],
            "auto_label_terms":            auto_label_terms,
            "evidence":                    evidence,
        })

    with open(out_path, "w", encoding="utf-8") as f:
        for r in tqdm(selected_records, desc="[Step 3] writing selections", unit="row"):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(preview_path, "w", encoding="utf-8") as f:
        f.write(f"=== step3 preview -- {len(selected_records):,} selected from {len(rows):,} candidates "
                f"(dedup_sim_threshold={dedup_sim_threshold}) ===\n\n")
        for r in selected_records[:40]:
            f.write(f"#{r['selection_rank']} row_idx={r['row_idx']} tfidf={r['tfidf_score']:.3f} "
                    f"max_sim_to_selected={r['max_similarity_to_selected']:.3f} "
                    f"family~{r['family_size_estimate']} terms={r['auto_label_terms']}\n")
            f.write(f"    PICK :: {(r['user_prompt'] or '').replace(chr(10), ' ')[:160]}\n")
            for e in r["evidence"][:3]:
                f.write(f"    nbr(sim={e['similarity']:.2f}) :: {(e['user_prompt'] or '').replace(chr(10), ' ')[:140]}\n")
            f.write("\n")

    family_sizes = [r["family_size_estimate"] for r in selected_records]
    summary = {
        "n_candidates_in":        len(rows),
        "n_selected":              len(selected_records),
        "evidence_k":              evidence_k,
        "family_sim_threshold":    family_sim_threshold,
        "dedup_sim_threshold":     dedup_sim_threshold,
        "family_size_min":         min(family_sizes) if family_sizes else None,
        "family_size_max":         max(family_sizes) if family_sizes else None,
        "family_size_avg":         sum(family_sizes) / len(family_sizes) if family_sizes else None,
        "embed_model":             embed_model,
        "device":                  device,
        "min_score":               min_score,
        "top_n":                   top_n,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[Step 3] Done. {len(selected_records):,} selected -> {out_path}")
    print(f"[Step 3] family_size_estimate: min={summary['family_size_min']} "
          f"avg={summary['family_size_avg']:.1f} max={summary['family_size_max']}")
    print(f"[Step 3] Preview -> {preview_path}")
    print(f"[Step 3] Summary -> {summary_path}")
    return out_path


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt-dir", default=None)
    p.add_argument("--in-name", default="step2/step2_candidates.jsonl")
    p.add_argument("--out-name", default="step3/step3_selected.jsonl")
    p.add_argument("--preview-name", default="step3/step3_preview.txt")
    p.add_argument("--summary-name", default="step3/step3_summary.json")
    p.add_argument("--select-n", type=int, default=DEFAULT_SELECT_N,
                    help="how many diverse representatives to select (default: the entire pool). "
                         "There's no LLM cost here, only step4 costs per item -- leave this at the "
                         "default and use step4's --limit to control how deep you actually spend "
                         "LLM calls into this ranking, without ever re-running this step.")
    p.add_argument("--evidence-k", type=int, default=DEFAULT_EVIDENCE_K,
                    help="nearest neighbors attached as evidence per selected representative")
    p.add_argument("--family-sim-threshold", type=float, default=DEFAULT_FAMILY_SIM_THRESHOLD,
                    help="cosine similarity threshold for family_size_estimate (a rough 'how common "
                         "is this pattern in the pool' proxy, not used in selection itself)")
    p.add_argument("--dedup-sim-threshold", type=float, default=DEFAULT_DEDUP_SIM_THRESHOLD,
                    help=f"HARD exclusion: once something within this cosine similarity of an "
                         f"already-selected item is seen, it's dropped outright (default "
                         f"{DEFAULT_DEDUP_SIM_THRESHOLD}). This is the only thing that excludes a "
                         "candidate in this step -- selection order is fixed by tfidf_score alone.")
    p.add_argument("--min-score", type=float, default=0.0,
                    help="drop candidates below this tfidf_score before selecting (default: no cutoff)")
    p.add_argument("--top-n", type=int, default=None,
                    help="only consider the top N candidates by tfidf_score (default: all)")
    p.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    p.add_argument("--device", default="auto",
                    help="'auto' picks cuda:1 if a second GPU exists, else cuda:0, else cpu")
    p.add_argument("--limit", type=int, default=None, help="only load the first N step2 rows (smoke test)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        ckpt_dir=args.ckpt_dir,
        in_name=args.in_name,
        out_name=args.out_name,
        preview_name=args.preview_name,
        summary_name=args.summary_name,
        select_n=args.select_n,
        evidence_k=args.evidence_k,
        family_sim_threshold=args.family_sim_threshold,
        dedup_sim_threshold=args.dedup_sim_threshold,
        min_score=args.min_score,
        top_n=args.top_n,
        embed_model=args.embed_model,
        device=args.device,
        limit=args.limit,
    )
