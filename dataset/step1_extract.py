"""Step 1: Load and cache the Dolci-Instruct-SFT dataset.

Downloads from HuggingFace on first run (usually a no-op -- if the dataset is
already in your local HF datasets cache, this just reads it, no network);
every run after that loads from this pipeline's own flat JSONL cache instead
of re-walking HF's nested schema, since that Python-level extraction loop is
the actually slow part (a few minutes over ~2M rows) and step2 gets re-run a
lot while tuning thresholds.

Self-contained copy of the old stage1_load.py (legacy/src_top_unused/stages)
so this pipeline doesn't reach outside its own folder. Row schema:
row_idx, user_prompt, assistant_response, source_dataset, domain. The
response is kept because step 4 shows it to the rewriter and the Table 1
evidence search matches against it (pass --drop-responses to omit it).

Outputs (inside ckpt_dir/step1/):
  step1_rows.jsonl           one JSON object per line, full-run cache
  step1_rows.limitN.jsonl    written instead when --limit N is passed, so a
                              smoke test never shadows the full-run cache
  step1_summary.json         dataset name, row count, a few sample rows

Usage:
  python step1_extract.py                 # full run (cache or extract)
  python step1_extract.py --limit 20000    # smoke test -- stops after 20k valid rows
"""
import argparse
import json
import os

from tqdm import tqdm

_DEFAULT_DATASET = "allenai/Dolci-Instruct-SFT"
# Pinned. Every prompt-family id in the released dataset is a row index into
# the train split at exactly this revision (2,152,112 rows). If the upstream
# dataset is ever updated, an unpinned load would silently shift every id.
_DEFAULT_REVISION = "bd3c8f3a9b2cc5a9682e44b96ddd0bb2ff027221"


def _load_split(hf_load, ds_id, ds_config=None):
    """Load split='train'; if not found, concatenate all available splits."""
    from datasets import concatenate_datasets
    args = [ds_id] + ([ds_config] if ds_config else [])
    try:
        return hf_load(*args, split="train", revision=_DEFAULT_REVISION)
    except ValueError as e:
        if "Unknown split" not in str(e):
            raise
        print("[Step 1] No 'train' split -- concatenating all available splits ...")
        ds_dict = hf_load(*args, revision=_DEFAULT_REVISION)
        return concatenate_datasets(list(ds_dict.values()))


def _extract_first_turn(example: dict):
    """Extract first (user, assistant) turn from an example.

    Handles common conversation field names/formats. Returns
    (user_prompt, assistant_response) or (None, None).
    """
    for field in ("conversations", "messages", "conversation", "turns"):
        conv = example.get(field)
        if conv and isinstance(conv, list):
            return _parse_conversation(conv)

    for u_key in ("prompt", "instruction", "input", "question", "user"):
        for a_key in ("response", "output", "completion", "answer", "assistant"):
            if u_key in example and a_key in example:
                return str(example[u_key]), str(example[a_key])

    return None, None


def _parse_conversation(conv: list):
    user_msg = None
    for turn in conv:
        if not isinstance(turn, dict):
            continue
        role = (turn.get("role") or turn.get("from") or "").lower()
        content = turn.get("content") or turn.get("value") or turn.get("text") or ""
        if role in ("user", "human") and user_msg is None:
            user_msg = str(content)
        elif role in ("assistant", "gpt", "bot", "model") and user_msg is not None:
            return user_msg, str(content)
    return None, None


def run(ckpt_dir: str = None, dataset_name: str = None, limit: int = None,
        out_name: str = "step1/step1_rows.jsonl", keep_responses: bool = True) -> list:
    ckpt_dir = ckpt_dir or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "work", "dataset")
    ckpt_dir = os.path.normpath(ckpt_dir)
    dataset_name = dataset_name or _DEFAULT_DATASET

    # A --limit run gets its own cache filename, never the canonical
    # out_name -- otherwise a smoke test with --limit would write a small
    # file to the same path a full run reads from, and a later full run
    # would silently "SKIP -- loading from cache" and hand back just the
    # limited rows.
    if limit is not None:
        base, ext = os.path.splitext(out_name)
        out_name = f"{base}.limit{limit}{ext}"
    jsonl_cache = os.path.join(ckpt_dir, out_name)
    os.makedirs(os.path.dirname(jsonl_cache), exist_ok=True)

    if os.path.exists(jsonl_cache):
        print(f"[Step 1] SKIP -- loading rows from cache {jsonl_cache} ...")
        rows = []
        with open(jsonl_cache, encoding="utf-8") as f:
            for line in tqdm(f, desc="[Step 1] loading cached rows", unit="row"):
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
                    if limit is not None and len(rows) >= limit:
                        break
        print(f"[Step 1] {len(rows):,} rows loaded")
        _write_summary(ckpt_dir, dataset_name, jsonl_cache, rows, limit)
        return rows

    print(f"[Step 1] Loading dataset {dataset_name} (first run, will cache to {jsonl_cache}) ...")
    from datasets import load_dataset as hf_load
    if ":" in dataset_name.split("/")[-1]:
        ds_id, ds_config = dataset_name.rsplit(":", 1)
        ds = _load_split(hf_load, ds_id, ds_config)
    else:
        ds = _load_split(hf_load, dataset_name)

    # assistant_response is ~60% of the raw text bytes but is kept by default:
    # step 4 shows it to the rewriter as unreliable context, and the Table 1
    # evidence search matches against it. --drop-responses omits it.
    tmp = jsonl_cache + ".tmp"
    n_written = 0
    pbar = tqdm(enumerate(ds), total=len(ds), desc="[Step 1] extracting (user, assistant) turns", unit="row")
    with open(tmp, "w", encoding="utf-8") as fh:
        for idx, example in pbar:
            user_prompt, assistant_response = _extract_first_turn(example)
            if not user_prompt or not assistant_response:
                continue
            row = {
                "row_idx":            idx,
                "user_prompt":        user_prompt,
                "source_dataset":     example.get("source_dataset", example.get("source", "")),
                "domain":             example.get("domain", example.get("source_dataset", "unknown")),
            }
            if keep_responses:
                row["assistant_response"] = assistant_response
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_written += 1
            if limit is not None and n_written >= limit:
                pbar.close()
                break
    os.replace(tmp, jsonl_cache)
    print(f"[Step 1] Extracted {n_written:,} rows -> {jsonl_cache}"
          f"{' (stopped early: --limit ' + str(limit) + ')' if limit is not None else ''}")

    rows = []
    with open(jsonl_cache, encoding="utf-8") as f:
        for line in tqdm(f, desc="[Step 1] loading rows into memory", unit="row"):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    print(f"[Step 1] Done. {len(rows):,} rows in memory.")
    _write_summary(ckpt_dir, dataset_name, jsonl_cache, rows, limit)
    return rows


def _write_summary(ckpt_dir: str, dataset_name: str, jsonl_cache: str, rows: list, limit) -> None:
    """A quick, human-readable sanity file -- row count + a couple of real
    sample rows, so you can eyeball that extraction actually worked without
    opening the (potentially multi-GB) full jsonl."""
    summary = {
        "dataset_name":   dataset_name,
        "cache_path":     jsonl_cache,
        "n_rows_loaded":  len(rows),
        "limit_applied":  limit,
        "sample_rows":    rows[:3],
    }
    summary_path = os.path.join(ckpt_dir, "step1", "step1_summary.json")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[Step 1] Summary -> {summary_path}")


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt-dir", default=None)
    p.add_argument("--dataset-name", default=None)
    p.add_argument("--limit", type=int, default=None,
                    help="smoke test: stop extraction after N valid rows, written to its own "
                         "step1_rows.limitN.jsonl so it never shadows the full-run cache")
    p.add_argument("--out-name", default="step1/step1_rows.jsonl")
    p.add_argument("--drop-responses", dest="keep_responses", action="store_false",
                    help="do not store assistant_response. Saves ~60%% of the file size, but "
                         "step 4 (rewrite) and the Table 1 evidence search both need it")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    rows = run(ckpt_dir=args.ckpt_dir, dataset_name=args.dataset_name, limit=args.limit,
               out_name=args.out_name, keep_responses=args.keep_responses)
    print(f"[Step 1] Sample row: {rows[0] if rows else None}")
