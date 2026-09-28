"""Step 6: Reframe each of step5's final probes into 30 real-world scenarios.

Adapted from an earlier version of this project (legacy/src_curation_unused/
generate_framings.py), which already worked out the hard lessons below the
hard way; this keeps that design, ported onto this pipeline's own
conventions (Chat Completions + gpt-5.6-luna, matching step3, instead of
the legacy script's OpenAI Responses API + gpt-5-mini).

Why reframe at all: testing "does the model always say Monarch" against
only ONE fixed phrasing ("Choose a random butterfly") can't tell you
whether the bias is about the underlying topic or just an artifact of that
exact template. Reframing the same question into 30 different real-world
scenarios (a tattoo artist choosing a design, a museum curator picking a
feature species, ...) tests whether the bias survives when the surface
wording changes completely.

Changes from the legacy version, per explicit direction:
  1. "random"/"randomly" are NOT forbidden in the reframe anymore (the
     legacy version banned the word entirely, forcing euphemisms like "off
     the cuff" -- no longer required; natural phrasing wins, including the
     word itself if it fits).
  2. Every reframe must still be phrased in plain, simple, easy-to-understand
     language, the same "understandable" bar step4 holds rewrites to.
  3. Model is gpt-5.6-luna (this project's current model, via OpenAI's
     Chat Completions API), not the legacy gpt-5-mini/Responses API.
  4. Dropped entirely: the legacy closed-bucket / canonical-option-order
     logic. None of step5's probes are fixed-choice-list questions, they're
     all open "Choose a random <topic>" questions, so that whole branch of
     complexity doesn't apply here.

What's kept from the legacy design, because it already works:
  - One call per probe asks for all 30 reframes at once (structured JSON),
    not 30 separate calls -- 4,725 probes means 4,725 calls for 141,750
    reframes, not 141,750 calls.
  - The reframe must embed the question inside a real activity a character
    is actually doing (designing, planning, recommending, ...), not just
    announce "here's a random scenario" -- and must NOT reveal that any
    answer is fine; the arbitrariness must live in the fact that the
    activity tolerates any reasonable answer, not in the framing saying so.
  - Explicit GOOD vs BAD worked examples in the prompt, specifically to
    stop the model from collapsing all 30 into one lazy "stranger asks
    off-the-cuff" template -- the same kind of repetition problem step3
    fought before we fixed its prompt.
  - The reframe must preserve the same *kind* of answer as the original
    (don't swap what's actually being asked), and must paraphrase the
    question stem, not copy it verbatim.
  - Resumable: an entry only counts as done once all 30 framing_idx values
    are present. A row that came back short (parsing issue, truncation)
    gets topped up on a later run, asking only for the missing count and
    telling the model which topics are already used, so the top-up doesn't
    just repeat itself.

Requires an OpenAI API key, same lookup order as step4 (ckpt_dir/api_key.txt,

Outputs (inside ckpt_dir/step6/):
  step6_reframed.jsonl   one row per (row_idx, framing_idx): row_idx,
                         framing_idx, base_question, framing_topic, framing,
                         original_example_answers, model
  step6_progress.json    {"done": {row_idx: [framing_idx, ...]}} -- resume support
  step6_preview.txt      human-readable sample of reframes
  step6_summary.json     counts

Usage:
  python step6_reframe.py --limit 2      # smoke test: 2 probes x 30 = 60 reframes
  python step6_reframe.py                # full run, resumable
"""
import argparse
import json
import os
import re
import sys
import threading
import time

from tqdm import tqdm

OPENAI_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_MODEL = "gpt-5.6-luna"
N_FRAMINGS = 30
# More creative work than step4 (invent 30 distinct scenarios per call,
# avoid collapsing into one lazy template across all of them) -- "low" was
# right for step4's narrower rewrite-one-question task, this needs more.
DEFAULT_REASONING_EFFORT = "medium"

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SYSTEM_PROMPT = (
    f"You produce EXACTLY {N_FRAMINGS} scenario rewrites of a single base "
    "question.\n\n"
    "The base question is a subjective free-pick question with many "
    "equally valid answers. Every rewrite must preserve that \"any answer "
    "is fine\" character: never write a scenario where the character "
    "making the choice has to weigh options, evaluate evidence, recall a "
    "specific fact, or work out a correct answer, and never reveal or hint "
    "at what the answer should be.\n\n"
    f"Write {N_FRAMINGS} distinct real-world scenarios where a character "
    "actually uses, picks, recommends, designs, plans, writes, builds, "
    "decides, etc on an answer in this domain, as a natural part of something "
    "they're already doing, so the choice emerges naturally from that "
    "activity. Be creative: vary the activity, the actor, "
    f"and the stakes across all {N_FRAMINGS}, don't let them collapse "
    "into one repeated shape.\n\n"
    "Requirements:\n"
    "- Vary the SENTENCE STRUCTURE across all the reframes too, don't let "
    "most of them settle into one repeated shape. Mix in some that are a "
    "single sentence with the choice folded inside it, some phrased as a "
    "request or instruction instead of a question, some where the "
    "question comes first and the setup follows, and some with no "
    "question mark at all, etc.\n"
    "- Every reframe must be written in plain, simple, easy-to-understand "
    "language: short sentences, no jargon, no convoluted structure.\n"
    "- The answer must still be the exact same specific kind of thing the "
    "base question asks for, never a broadened or more generic version of "
    "it.\n"
    "- Paraphrase the question, don't copy its stem or exact answer-domain "
    "noun phrase verbatim, use a shorter equivalent or let the scene imply "
    "the domain instead.\n"
    "- Do not use em dashes anywhere, use commas, semicolons, colons, or "
    "periods instead.\n\n"
    "Respond with ONLY a JSON object, no other text."
)

USER_TEMPLATE = (
    "BASE QUESTION (to be preserved in every rewrite):\n{base_question}\n\n"
    "{avoid_block}"
    "Output a JSON object with exactly this shape:\n"
    '  {{"framings": [{{"topic": "1-3 word tag naming the scenario, not '
    'just a domain (e.g. \\"tattoo design session\\", not \\"Art\\")", '
    '"framing": "the rewritten scenario as a single concise question '
    'someone could plausibly ask, woven into the scene"}}, ...] }}\n'
    "The \"framings\" array must contain exactly {target_n} objects.\n"
)

_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def _load_api_key(ckpt_dir: str) -> str:
    for key_file in (os.path.join(ckpt_dir, "api_key.txt"),):
        if os.path.exists(key_file):
            with open(key_file, "r", encoding="utf-8") as f:
                key = f.read().strip()
            if key:
                return key
    return os.environ.get("OPENAI_API_KEY")


def _build_user_prompt(base_question: str, target_n: int, avoid_topics: list) -> str:
    avoid_block = ""
    if avoid_topics:
        listed = "\n".join(f"  - {t}" for t in avoid_topics)
        avoid_block = (
            "ALREADY-USED TOPICS for this question (do not duplicate or "
            "closely paraphrase these, every new scenario must be "
            f"substantively new):\n{listed}\n\n"
        )
    return USER_TEMPLATE.format(base_question=base_question, avoid_block=avoid_block, target_n=target_n)


def _parse_output(text: str):
    if not text:
        return None
    text = text.strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        m = _JSON_OBJ_RE.search(text)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    if not isinstance(obj, dict):
        return None
    framings = obj.get("framings")
    if not isinstance(framings, list):
        return None
    out = [f for f in framings if isinstance(f, dict)
           and isinstance(f.get("topic"), str) and f.get("topic").strip()
           and isinstance(f.get("framing"), str) and f.get("framing").strip()]
    return out or None


def _call_openai(session, api_key: str, model: str, base_question: str, target_n: int,
                  avoid_topics: list, max_tokens: int, max_retries: int, timeout: int,
                  reasoning_effort: str):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    user_prompt = _build_user_prompt(base_question, target_n, avoid_topics)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        # No "temperature" -- gpt-5.6-luna rejects any non-default value
        # (confirmed live in step3; same restriction applies here).
        "max_completion_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    if reasoning_effort and reasoning_effort != "none":
        payload["reasoning_effort"] = reasoning_effort

    last_err = None
    for attempt in range(max_retries + 1):
        try:
            resp = session.post(OPENAI_URL, headers=headers, json=payload, timeout=timeout)
            if resp.status_code == 429 or resp.status_code >= 500:
                last_err = f"HTTP {resp.status_code}: {resp.text[:300]}"
                time.sleep(min(2 ** attempt, 30))
                continue
            if resp.status_code == 400:
                last_err = f"HTTP 400: {resp.text[:300]}"
                break  # malformed request won't fix itself on retry
            resp.raise_for_status()
            data = resp.json()
            raw = data["choices"][0]["message"]["content"]
            framings = _parse_output(raw)
            if framings is None:
                last_err = f"unparseable/empty framings: {(raw or '')[:200]!r}"
                time.sleep(min(2 ** attempt, 30))
                continue
            return framings, None
        except Exception as e:
            last_err = repr(e)
            time.sleep(min(2 ** attempt, 30))
    return None, last_err


def run(
    ckpt_dir: str = None,
    in_name: str = "step5/step5_deduped.jsonl",
    out_name: str = "step6/step6_reframed.jsonl",
    progress_name: str = "step6/step6_progress.json",
    preview_name: str = "step6/step6_preview.txt",
    summary_name: str = "step6/step6_summary.json",
    model: str = DEFAULT_MODEL,
    n_framings: int = N_FRAMINGS,
    concurrency: int = 16,
    max_tokens: int = 6000,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    max_retries: int = 4,
    timeout: int = 120,
    limit: int = None,
    resume: bool = True,
) -> str:
    ckpt_dir = ckpt_dir or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "work", "dataset")
    ckpt_dir = os.path.normpath(ckpt_dir)
    api_key = _load_api_key(ckpt_dir)

    in_path = os.path.join(ckpt_dir, in_name)
    out_path = os.path.join(ckpt_dir, out_name)
    progress_path = os.path.join(ckpt_dir, progress_name)
    preview_path = os.path.join(ckpt_dir, preview_name)
    summary_path = os.path.join(ckpt_dir, summary_name)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    probes = []
    with open(in_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="[Step 6] loading step5 probes", unit="row"):
            line = line.strip()
            if line:
                probes.append(json.loads(line))
                if limit is not None and len(probes) >= limit:
                    break
    print(f"[Step 6] {len(probes):,} probes loaded from {in_path} -> up to "
          f"{len(probes) * n_framings:,} reframes")

    done = {}  # row_idx (str) -> set of framing_idx already written
    if resume and os.path.exists(progress_path):
        with open(progress_path, "r", encoding="utf-8") as f:
            prog = json.load(f)
        done = {k: set(v) for k, v in prog.get("done", {}).items()}
        n_complete = sum(1 for v in done.values() if len(v) >= n_framings)
        if done:
            print(f"[Step 6] RESUME -- {n_complete:,}/{len(probes):,} probes already fully done")

    existing_topics = {}  # row_idx (str) -> list of topics already used, for topup avoid_topics
    if resume and os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                existing_topics.setdefault(str(r["row_idx"]), []).append(r["framing_topic"])

    todo = [p for p in probes if len(done.get(str(p.get("row_idx")), set())) < n_framings]
    if not todo:
        print("[Step 6] Nothing left to do.")
        return out_path
    print(f"[Step 6] {len(todo):,} probes need (more) reframes")

    if not api_key:
        print(f"[Step 6] ERROR: no API key found. Checked {ckpt_dir}/api_key.txt and $OPENAI_API_KEY.", file=sys.stderr)
        sys.exit(1)

    import requests
    session = requests.Session()
    out_lock = threading.Lock()
    prog_lock = threading.Lock()
    n_reframes_written = 0
    n_probes_complete = sum(1 for v in done.values() if len(v) >= n_framings)
    n_failed = 0

    out_f = open(out_path, "a" if (resume and os.path.exists(out_path)) else "w", encoding="utf-8")

    def _flush_progress():
        with prog_lock:
            with open(progress_path, "w", encoding="utf-8") as pf:
                json.dump({"done": {k: sorted(v) for k, v in done.items()}}, pf)

    def _process(probe: dict):
        row_idx = str(probe.get("row_idx"))
        have = done.get(row_idx, set())
        missing = n_framings - len(have)
        avoid_topics = existing_topics.get(row_idx, [])
        base_question = probe.get("rewrite") or ""
        framings, err = _call_openai(session, api_key, model, base_question, missing,
                                      avoid_topics, max_tokens, max_retries, timeout, reasoning_effort)
        if err:
            return "failed", err, 0
        # assign the next available framing_idx values to whatever came back
        next_idxs = [i for i in range(n_framings) if i not in have][:len(framings)]
        written = 0
        with out_lock:
            for idx, fr in zip(next_idxs, framings):
                record = {
                    "row_idx":                  probe.get("row_idx"),
                    "framing_idx":               idx,
                    "base_question":             base_question,
                    "framing_topic":             fr["topic"].strip(),
                    "framing":                   fr["framing"].strip(),
                    "original_example_answers":  probe.get("example_answers"),
                    "model":                     model,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                have.add(idx)
                written += 1
            out_f.flush()
            done[row_idx] = have
        status = "complete" if len(have) >= n_framings else "partial"
        return status, None, written

    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(_process, p): p for p in todo}
        pbar = tqdm(as_completed(futures), total=len(todo), desc="[Step 6] reframing", unit="probe")
        for i, fut in enumerate(pbar, start=1):
            status, err, written = fut.result()
            n_reframes_written += written
            if status == "failed":
                n_failed += 1
                tqdm.write(f"[Step 6] WARNING: row_idx={futures[fut].get('row_idx')} "
                           f"failed after retries: {err}")
            elif status == "complete":
                n_probes_complete += 1
            pbar.set_postfix(reframes=n_reframes_written, probes_complete=n_probes_complete, failed=n_failed)

            if i % 50 == 0 or i == len(todo):
                _flush_progress()

    _flush_progress()
    out_f.close()

    # --- preview: sample of reframes with their base question ---
    generated = []
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                generated.append(json.loads(line))
    with open(preview_path, "w", encoding="utf-8") as f:
        f.write(f"=== step6 preview -- {len(generated):,} reframes ===\n\n")
        by_row = {}
        for r in generated:
            by_row.setdefault(r["row_idx"], []).append(r)
        for row_idx, items in by_row.items():
            f.write(f"row_idx {row_idx}: {items[0]['base_question']}\n")
            for it in items:
                f.write(f"  [{it['framing_idx']}] ({it['framing_topic']}) {it['framing']}\n")
            f.write("\n")

    summary = {
        "model":                model,
        "n_framings_per_probe": n_framings,
        "n_probes":             len(probes),
        "n_probes_complete":    n_probes_complete,
        "n_reframes_total":     len(generated),
        "n_failed_calls":       n_failed,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[Step 6] Done. {len(generated):,} reframes across "
          f"{n_probes_complete:,}/{len(probes):,} fully-complete probes -> {out_path}")
    print(f"[Step 6] Preview -> {preview_path}")
    print(f"[Step 6] Summary -> {summary_path}")
    return out_path


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt-dir", default=None)
    p.add_argument("--in-name", default="step5/step5_deduped.jsonl")
    p.add_argument("--out-name", default="step6/step6_reframed.jsonl")
    p.add_argument("--progress-name", default="step6/step6_progress.json")
    p.add_argument("--preview-name", default="step6/step6_preview.txt")
    p.add_argument("--summary-name", default="step6/step6_summary.json")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--n-framings", type=int, default=N_FRAMINGS, help="reframes per probe (default 30)")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--max-tokens", type=int, default=6000,
                    help="generous by default -- 30 scenarios plus reasoning tokens in one response")
    p.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT,
                    choices=["none", "low", "medium", "high", "xhigh", "max"])
    p.add_argument("--max-retries", type=int, default=4)
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--limit", type=int, default=None, help="only process the first N probes (smoke test)")
    p.add_argument("--no-resume", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        ckpt_dir=args.ckpt_dir,
        in_name=args.in_name,
        out_name=args.out_name,
        progress_name=args.progress_name,
        preview_name=args.preview_name,
        summary_name=args.summary_name,
        model=args.model,
        n_framings=args.n_framings,
        concurrency=args.concurrency,
        max_tokens=args.max_tokens,
        reasoning_effort=args.reasoning_effort,
        max_retries=args.max_retries,
        timeout=args.timeout,
        limit=args.limit,
        resume=not args.no_resume,
    )
