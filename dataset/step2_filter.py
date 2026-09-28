"""Step 2: Cheap, LLM-free filter down to arbitrary/random-choice candidates.

No embeddings, no LLM calls -- this is meant to run in well under a minute
even over the full ~2.15M-row Dolci corpus. Three passes:

  1. Hard filter: keep only rows whose user_prompt contains "random"
     (substring, case-insensitive -- also catches "randomly", "randomize",
     etc). This is a deliberate scope decision: it targets prompts that
     explicitly ask for an arbitrary pick (matching the canonical rewrite
     target "Choose a random <topic>"), not general subjective/preference
     prompts ("what's your favorite ..."). See CODE_NOISE_SUBSTRINGS below
     for why this alone isn't enough.

  2. Code-noise exclusion: "random" is extremely common in programming/stats
     contexts that have nothing to do with the model making an arbitrary
     choice ("write a function using np.random.seed", "explain random
     forests"). Drop rows matching any pattern in CODE_NOISE_SUBSTRINGS.
     This list is a starting point -- inspect step2_preview.txt and tune it.

  3. TF-IDF seed scoring (no filtering, just ranking): fit ONE TfidfVectorizer
     over the surviving pool (ngram_range=(1,3), token_pattern kept
     permissive so single-letter tokens like "a" survive -- needed so
     "choose a random" and "choose ... random" score differently), transform
     SEED_PHRASES through it, and score every row as the MAX cosine
     similarity across all seeds (OR semantics: strong match on any one
     seed phrase is enough, no averaging across seeds). Records which seed
     produced the max (tfidf_best_seed) alongside the score.

No cutoff is applied here on purpose -- step3 (clustering) takes --min-score
/ --top-n so you can pick a threshold after actually looking at the score
distribution in step2_preview.txt / step2_summary.json.

Outputs (inside ckpt_dir/step2/):
  step2_candidates.jsonl   surviving rows + tfidf_score + tfidf_best_seed,
                           sorted by tfidf_score descending
  step2_preview.txt        human-readable sample (top scores + score deciles)
                           so you can eyeball quality before clustering
  step2_summary.json       funnel counts + score percentiles

Usage:
  python step2_filter.py
  python step2_filter.py --limit 100000   # smoke test
"""
import argparse
import json
import os
import re
from collections import Counter

from tqdm import tqdm

# Compound seeds -- verb + "random" together as one phrase, not verb-only.
# A verb-only seed (e.g. "select") scores a document as soon as "select" and
# "random" both appear ANYWHERE in it, even if they're in unrelated sentences
# (a 300-word stats problem that says "select 5 individuals" in one sentence
# and "random number table" in another). Requiring the phrase together is a
# much stronger signal: with ngram_range=(1,3) below, "choose a random" only
# scores high on documents where those words are actually adjacent.
# OR-combined via max across seeds, no weighting.
SEED_PHRASES = [
    "choose a random",
    "pick a random",
    "select a random",
    "name a random",
    "generate a random",
    "give me a random",
    "suggest a random",
    "recommend a random",
    "randomly choose",
    "randomly pick",
    "randomly select",
    "randomly generate",
]

# Starting list -- tune after reading step2_preview.txt. Lowercase substring
# match against the prompt; any hit drops the row. Widened from a real
# 20k-row smoke test: the original list only caught literal code syntax
# ("```", "import ", "def ") and missed the far more common plain-English
# code-request phrasing ("implement a function", "design a class"), which
# dominated the top-scored results despite containing "random" incidentally.
CODE_NOISE_SUBSTRINGS = [
    # literal code syntax / library calls
    "```", "import ", ".py", "np.random", "random.seed", "random.random",
    "math.random", "randint", "random forest", "random variable",
    "random access memory", "pseudo-random", "pseudorandom",
    "random number generator", "def ", "srand(", "rand()",
    # plain-English code-request framings (the actual dominant noise source)
    "function that generates", "write a program", "write a function",
    "write a method", "implement a function", "implement a method",
    "implement the function", "create a function", "create a method",
    "design a class", "design a function", "craft a python", "craft a function",
    "the function should", "you are tasked with creating a function",
    "you are tasked with implementing", "algorithm that", "algorithm to",
    "random walk", "random seed", "returns a new list", "takes a list",
    "shuffle the list", "shuffle_list", "randomize_list", "given a list of integers",
    # language/library names strongly correlated with code requests
    " python ", " java ", " javascript", " kotlin", " c++", " typescript",
    "sql query", " css ", " html ",
    # statistics/physics word-problems (superficially "random" via
    # "probability", "randomly selected", not an arbitrary-choice prompt)
    "probability that", "standard deviation", "normally distributed",
    "population consists",
    # more code-request phrasings found in a real 104-candidate review
    # (step2_candidates.jsonl on a 20k-row sample) -- word order varies a
    # lot, this list will never be exhaustive, step4's LLM check is the
    # real backstop for whatever still slips through
    "code snippet", "write a code", "fix this code", "ocr result",
    "implement this using", "utilize the random module", "numpy array",
    "using the random module", "create a program", "printed to the console",
    "print to the console",
]

_STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "and", "is", "for", "on", "with",
    "that", "this", "it", "as", "be", "are", "was", "or", "at", "by", "from",
    "me", "my", "you", "your", "i", "we", "please", "can", "would", "like",
    "random", "randomly",
}
_WORD_RE = re.compile(r"[a-zA-Z']+")


def _is_random_row(user_prompt: str) -> bool:
    return "random" in (user_prompt or "").lower()


def _is_code_noise(user_prompt: str) -> bool:
    text = (user_prompt or "").lower()
    return any(pat in text for pat in CODE_NOISE_SUBSTRINGS)


def _top_terms(texts: list, n: int = 6) -> list:
    counts = Counter()
    for t in texts:
        for w in _WORD_RE.findall((t or "").lower()):
            if len(w) > 2 and w not in _STOPWORDS:
                counts[w] += 1
    return [w for w, _ in counts.most_common(n)]


def run(
    ckpt_dir: str = None,
    in_name: str = "step1/step1_rows.jsonl",
    out_name: str = "step2/step2_candidates.jsonl",
    preview_name: str = "step2/step2_preview.txt",
    summary_name: str = "step2/step2_summary.json",
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
        for line in tqdm(f, desc="[Step 2] loading step1 rows", unit="row"):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    n_total = len(rows)
    print(f"[Step 2] {n_total:,} rows loaded from {in_path}")

    random_rows = [
        r for r in tqdm(rows, desc="[Step 2] scanning for random/randomly", unit="row")
        if _is_random_row(r.get("user_prompt"))
    ]
    print(f"[Step 2] {len(random_rows):,} rows contain \"random\"/\"randomly\"")

    kept = [
        r for r in tqdm(random_rows, desc="[Step 2] dropping code/technical noise", unit="row")
        if not _is_code_noise(r.get("user_prompt"))
    ]
    n_dropped_code = len(random_rows) - len(kept)
    print(f"[Step 2] {n_dropped_code:,} dropped as code/technical noise -- {len(kept):,} remain")

    if not kept:
        print("[Step 2] WARNING: nothing survived the filters -- nothing to score.")
        open(out_path, "w").close()
        return out_path

    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np

    texts = [r.get("user_prompt") or "" for r in kept]
    print(f"[Step 2] Fitting TF-IDF over {len(texts):,} prompts (ngram_range=(1,3)) ...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        token_pattern=r"(?u)\b\w+\b",  # keep single-letter tokens (e.g. "a") -- see module docstring
        max_features=200_000,
        sublinear_tf=True,
        stop_words=None,
    )
    doc_matrix = vectorizer.fit_transform(texts)
    seed_matrix = vectorizer.transform(SEED_PHRASES)

    sims = (doc_matrix @ seed_matrix.T).toarray()  # (n_docs, n_seeds), rows are unit-norm so this is cosine sim
    best_seed_idx = sims.argmax(axis=1)
    best_score = sims[np.arange(len(kept)), best_seed_idx]

    for r, score, seed_idx in zip(kept, best_score, best_seed_idx):
        r["tfidf_score"] = float(score)
        r["tfidf_best_seed"] = SEED_PHRASES[seed_idx]

    kept.sort(key=lambda r: r["tfidf_score"], reverse=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for r in tqdm(kept, desc="[Step 2] writing candidates", unit="row"):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    scores = np.array([r["tfidf_score"] for r in kept])
    percentiles = {str(p): float(np.percentile(scores, p)) for p in (0, 10, 25, 50, 75, 90, 99, 100)}
    seed_counts = Counter(r["tfidf_best_seed"] for r in kept)

    # --- human-readable preview: top-N plus a sample across score deciles ---
    with open(preview_path, "w", encoding="utf-8") as f:
        f.write(f"=== step2 preview -- {len(kept):,} candidates ===\n")
        f.write(f"score percentiles: {json.dumps(percentiles, indent=2)}\n")
        f.write(f"best_seed counts: {dict(seed_counts.most_common())}\n\n")

        f.write("--- top 40 by score ---\n")
        for i, r in enumerate(kept[:40]):
            prompt = (r.get("user_prompt") or "").replace("\n", " ")[:200]
            f.write(f"[{i:>3}] score={r['tfidf_score']:.3f} seed={r['tfidf_best_seed']!r} :: {prompt}\n")

        f.write("\n--- sample across score deciles (to see where quality drops off) ---\n")
        n = len(kept)
        for decile in range(0, 100, 10):
            idx = min(n - 1, int(n * decile / 100))
            r = kept[idx]
            prompt = (r.get("user_prompt") or "").replace("\n", " ")[:200]
            f.write(f"[p{decile:>2}, rank {idx:>6}] score={r['tfidf_score']:.3f} "
                    f"seed={r['tfidf_best_seed']!r} :: {prompt}\n")

    summary = {
        "n_total_rows":           n_total,
        "n_contains_random":      len(random_rows),
        "n_dropped_code_noise":   n_dropped_code,
        "n_candidates":           len(kept),
        "score_percentiles":      percentiles,
        "best_seed_counts":       dict(seed_counts.most_common()),
        "seed_phrases":           SEED_PHRASES,
        "code_noise_substrings":  CODE_NOISE_SUBSTRINGS,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[Step 2] Done. {len(kept):,} candidates -> {out_path}")
    print(f"[Step 2] Score percentiles: {percentiles}")
    print(f"[Step 2] Preview -> {preview_path}")
    print(f"[Step 2] Summary -> {summary_path}")
    return out_path


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt-dir", default=None)
    p.add_argument("--in-name", default="step1/step1_rows.jsonl")
    p.add_argument("--out-name", default="step2/step2_candidates.jsonl")
    p.add_argument("--preview-name", default="step2/step2_preview.txt")
    p.add_argument("--summary-name", default="step2/step2_summary.json")
    p.add_argument("--limit", type=int, default=None, help="only load the first N step1 rows (smoke test)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        ckpt_dir=args.ckpt_dir,
        in_name=args.in_name,
        out_name=args.out_name,
        preview_name=args.preview_name,
        summary_name=args.summary_name,
        limit=args.limit,
    )
